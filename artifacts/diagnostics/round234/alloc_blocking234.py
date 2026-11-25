#!/usr/bin/env python3
"""R234b: does allocating a large FP32 temporary block the host while the device is busy?

Evidence chain so far: the ~3.4 s/step that R231 could not attribute sits inside the
attention/MLP parent spans. R228-R233 disproved weight wait, allocator-vs-copy
churn, audit syncs and guard syncs; R234 measured per-op host dispatch at 0.013 ms,
so dispatch cannot explain 26-35 ms gaps either.

What remains is that the *host itself parks* inside those regions: R232 measured the
MLP call at 83.4 ms host wall vs 88.3 ms device span, i.e. host and device advance
together. The MLP allocates two 300 MB FP32 temporaries per block
(`gate.float()`, `up.float()`), so the candidate mechanism is an allocator
stream-synchronisation on large blocks.

This benchmark (no model, no weights) measures:
  1. host wall of allocating a 300 MB NPU tensor with the device idle vs busy;
  2. the same for repeated alloc/free so the caching pool grows;
  3. whether a queued backlog really drains before the allocation returns.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

MB = 1024 * 1024
TEMP_BYTES = 300 * MB          # one gate.float()/up.float() FP32 temporary
REPEATS = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backlog-ops", type=int, default=2000)
    args = ap.parse_args()
    if not 0 <= args.device <= 7:
        raise SystemExit("device must be an observed logical card 0..7")

    import torch
    import torch_npu as _tn

    torch.npu.set_device(args.device)
    props = torch.npu.get_device_properties(args.device)
    n = TEMP_BYTES // 4
    x = torch.zeros(1024, dtype=torch.float16, device="npu")
    y = torch.ones(1024, dtype=torch.float16, device="npu")

    def alloc_tmp():
        t = torch.empty(n, dtype=torch.float32, device="npu")
        return t

    def measure_alloc(with_backlog):
        samples = []
        for _ in range(REPEATS):
            torch.npu.synchronize()
            if with_backlog:
                for _ in range(args.backlog_ops):
                    x.add_(y)
            t0 = time.perf_counter()
            tmp = alloc_tmp()
            host_ms = (time.perf_counter() - t0) * 1000.0
            torch.npu.synchronize()
            samples.append(host_ms)
            del tmp
        return samples

    # grow the cached pool once with the device idle
    warm = alloc_tmp()
    del warm
    torch.npu.synchronize()

    idle = measure_alloc(False)
    busy = measure_alloc(True)
    torch.npu.synchronize()

    backlog_device_ms = None
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.backlog_ops):
        x.add_(y)
    end.record()
    torch.npu.synchronize()
    backlog_device_ms = start.elapsed_time(end)

    report = {
        "schema_version": "r234b.alloc_blocking.v1",
        "host": "ts-217",
        "measurement_only": True,
        "no_model_weights_loaded": True,
        "no_model_import": True,
        "device": {"requested_id": args.device,
                   "actual_id": int(torch.npu.current_device()),
                   "name": str(getattr(props, "name", "unknown"))},
        "runtime": {"torch": torch.__version__,
                    "torch_npu": getattr(_tn, "__version__", "unknown")},
        "temp_bytes": TEMP_BYTES,
        "backlog_ops": args.backlog_ops,
        "backlog_device_ms": round(backlog_device_ms, 3),
        "alloc_host_ms_idle": {"samples": idle, "median": round(statistics.median(idle), 4)},
        "alloc_host_ms_with_backlog": {"samples": busy,
                                       "median": round(statistics.median(busy), 4)},
        "implied": {
            "extra_ms_when_busy": round(statistics.median(busy) - statistics.median(idle), 4),
            "backlog_drain_explains": bool(
                statistics.median(busy) - statistics.median(idle) > 0.5 * backlog_device_ms),
            "note": "if allocation does not wait for the backlog, the allocator is not "
                    "synchronising and cannot explain the MLP host wall",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT " + json.dumps({
        "backlog_device_ms": report["backlog_device_ms"],
        "alloc_idle_ms": report["alloc_host_ms_idle"]["median"],
        "alloc_busy_ms": report["alloc_host_ms_with_backlog"]["median"],
        "extra_ms": report["implied"]["extra_ms_when_busy"],
        "drain_explains": report["implied"]["backlog_drain_explains"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())