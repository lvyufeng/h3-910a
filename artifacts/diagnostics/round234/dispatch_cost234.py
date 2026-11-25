#!/usr/bin/env python3
"""R234: how expensive is dispatching one small NPU op from the host?

R228-R233 disproved every external mechanism behind the 3.43 s/step that is still
unattributed inside the attention/MLP parent spans: weight-prefetch wait, allocator
churn, audit host syncs, and the guarded linear's `float(peak)` host sync. R232 did
show that large ops dispatch cheaply (qkv 0.24 ms host for 17 ms device, fc1 0.23 ms
host for 23 ms device), so the remaining candidate is that the *many small* ops in
those regions are host-dispatch limited: device idle waiting for the next launch.

This benchmark measures, with no model and no weights, the host wall needed to
enqueue one op at several sizes, and the achieved device time per op. The ratio
host/device for small tensors is the quantity that decides whether the parent spans
are host-bound.

Measurement only; nothing is adopted.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

SIZES = [
    ("tiny_64x64", (64, 64), 1000),
    ("elem_1k", (1024,), 1000),
    ("elem_224x128", (224, 128), 1000),
    ("elem_1m", (1024, 1024), 300),
    ("elem_8m", (2048, 4096), 100),
    ("elem_37m_attn_shaped", (10448, 3584), 40),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not 0 <= args.device <= 7:
        raise SystemExit("device must be an observed logical card 0..7")

    import torch
    import torch_npu as _tn

    torch.npu.set_device(args.device)
    props = torch.npu.get_device_properties(args.device)
    report = {
        "schema_version": "r234.dispatch_cost.v1",
        "host": "ts-217",
        "measurement_only": True,
        "no_model_weights_loaded": True,
        "no_model_import": True,
        "device": {"requested_id": args.device,
                   "actual_id": int(torch.npu.current_device()),
                   "name": str(getattr(props, "name", "unknown"))},
        "runtime": {"torch": torch.__version__,
                    "torch_npu": getattr(_tn, "__version__", "unknown")},
        "cases": {},
    }

    for name, shape, repeats in SIZES:
        x = torch.zeros(shape, dtype=torch.float16, device="npu")
        y = torch.ones(shape, dtype=torch.float16, device="npu")

        def enqueue(n):
            # pure async dispatch: no synchronize inside the timed window
            for _ in range(n):
                x.add_(y)

        # warm up the dispatch path
        enqueue(10)
        torch.npu.synchronize()

        host_samples = []
        for _ in range(5):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            enqueue(repeats)
            host_ms = (time.perf_counter() - t0) * 1000.0 / repeats
            torch.npu.synchronize()
            host_samples.append(host_ms)

        # device time per op measured with events over a short queued run
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            x.add_(y)
        end.record()
        torch.npu.synchronize()
        device_ms = start.elapsed_time(end) / repeats

        host_ms = statistics.median(host_samples)
        report["cases"][name] = {
            "shape": list(shape),
            "elements": int(x.numel()),
            "host_dispatch_ms_per_op": round(host_ms, 4),
            "device_ms_per_op": round(device_ms, 4),
            "host_over_device_x": round(host_ms / device_ms, 2) if device_ms else None,
            "ops_per_second_host_limited": round(1000.0 / host_ms, 1) if host_ms else None,
        }
        print("CASE", json.dumps({name: report["cases"][name]}), flush=True)
        del x, y
        torch.npu.empty_cache()

    tiny = report["cases"].get("tiny_64x64", {}).get("host_dispatch_ms_per_op")
    report["implications"] = {
        "ops_dispatchable_per_ms": round(1.0 / tiny, 2) if tiny else None,
        "gap_model": (
            "If a region issues N small ops and each costs host_dispatch_ms while the "
            "device needs device_ms, the parent span includes (host-device)*N of device "
            "idle. attn/mlp issue on the order of 10-40 small ops per block."),
        "examples": {
            "attn_26.7ms_gap_vs_1ms_dispatch": round(26.7 / 1.0, 1),
            "mlp_34.9ms_gap_vs_1ms_dispatch": round(34.9 / 1.0, 1),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY", json.dumps(report["cases"], ensure_ascii=False)[:800], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())