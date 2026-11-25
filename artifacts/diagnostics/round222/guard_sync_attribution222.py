#!/usr/bin/env python3
"""R222: attribute the ~11 ms/call in-model gap left open by R221.

R221 measured the guarded linear standalone at 7.08 ms (`out_proj`) / 18.43 ms
(`fc2`), while R147 measured 18.31 / 29.93 ms in-model. That ~11.2-11.5 ms gap
is neither the range guard nor the GEMM. Candidate causes:

  (H1) the guard's host-side `float(peak)` drains every stream, including the
       layerwise weight-prefetch stream (~646 MB per prefetch, 32.3 GB/step);
  (H2) the row-parallel TP all-reduce is serial with the GEMM.

This script tests H1 without any model, weights, or source change:

  * queue device-to-device copies on a side stream (allocation-free, so no
    allocator/stream registration issue) and then time the guard's host sync.
    If the sync waits for that foreign-stream work, H1 is supported; if it
    returns at idle latency, H1 is disproved for this runtime.
  * measure one real pinned H2D copy to convert "pending bytes" into ms.
  * check whether a device-side scale computation with no host sync stays
    bitwise identical to the deployed host-sync path on a scale==1 input.

Timing plus a bitwise comparison only. Nothing is adopted, no approval implied.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

M = 10448
PREFETCH_BYTES = 646_000_000  # 32.3 GB/step over ~50 layerwise prefetches
SIDE_COPIES = 5
SIDE_BUFFER_BYTES = 4 * 1024**3  # 4 GiB per buffer, 5 queued copies
CASES = [("attn_out_proj_guarded", 3584, 5376, "float16"),
         ("mlp_fc2_guarded", 7168, 5376, "float32")]


def summ(v):
    v = [float(x) for x in v]
    return {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not 0 <= args.device <= 7:
        raise SystemExit("device must be an observed logical card 0..7")
    if not 1 <= args.steps <= 100:
        raise SystemExit("invalid sample count")

    import torch
    import torch_npu as _tn

    torch.npu.set_device(args.device)
    props = torch.npu.get_device_properties(args.device)
    device = torch.device(f"npu:{args.device}")
    side = torch.npu.Stream()
    result = {
        "schema_version": "r222.guard_sync_attribution.v1",
        "host": "ts-217",
        "measurement_only": True,
        "no_model_weights_loaded": True,
        "no_model_import": True,
        "no_source_modification": True,
        "candidate_installed": False,
        "numerical_approval_granted": False,
        "device": {"requested_id": args.device,
                   "actual_id": int(torch.npu.current_device()),
                   "name": str(getattr(props, "name", "unknown"))},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "torch_npu": getattr(_tn, "__version__", "unknown")},
        "m_tokens": M,
        "prefetch_bytes_per_event": PREFETCH_BYTES,
        "h2d_bandwidth": {},
        "cross_stream_queue": {"queued_copies": SIDE_COPIES,
                               "buffer_bytes": SIDE_BUFFER_BYTES},
        "host_sync_blocking": {},
        "guard_timing": {},
        "bitwise": {},
        "interpretation_limits": (
            "The side stream carries pure d2d copies, not the real prefetch graph; the "
            "test answers whether the sync drains foreign-stream work at all. Real H2D "
            "prefetch is slower per byte, so a measured wait here is a lower bound on the "
            "in-model cross-stream stall, and a measured non-wait disproves H1."),
    }

    # ---- one real pinned H2D copy, for the bytes->ms conversion --------------
    n_float = PREFETCH_BYTES // 4
    try:
        host_src = torch.empty(n_float, dtype=torch.float32, pin_memory=True)
        pinned = True
    except BaseException:
        host_src = torch.empty(n_float, dtype=torch.float32)
        pinned = False
    host_src.normal_(0.0, 0.01)
    h2d_dst = torch.empty(n_float, dtype=torch.float32, device=device)
    torch.npu.synchronize()

    def h2d_once():
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        h2d_dst.copy_(host_src, non_blocking=True)
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end)

    for _ in range(2):
        h2d_once()
    h2d_samples = [h2d_once() for _ in range(max(5, args.steps // 2))]
    h2d_ms = statistics.median(h2d_samples)
    result["h2d_bandwidth"] = {
        "pinned": pinned, "bytes": PREFETCH_BYTES, "copy_ms": summ(h2d_samples),
        "effective_GB_per_s": round(PREFETCH_BYTES / (h2d_ms * 1e-3) / 1e9, 1),
        "prefetch_ms_implied_per_event": round(h2d_ms, 3),
    }
    del h2d_dst
    torch.npu.empty_cache()

    # ---- side-stream d2d queue --------------------------------------------
    n_buf = SIDE_BUFFER_BYTES // 4
    a = torch.empty(n_buf, dtype=torch.float32, device=device)
    b = torch.empty(n_buf, dtype=torch.float32, device=device)
    a.fill_(1.0)
    torch.npu.synchronize()

    def queue_side_copies():
        with torch.npu.stream(side):
            for _ in range(SIDE_COPIES):
                b.copy_(a, non_blocking=True)

    def time_side_queue():
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(side):
            start.record()
            for _ in range(SIDE_COPIES):
                b.copy_(a, non_blocking=True)
            end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end)

    for _ in range(2):
        time_side_queue()
    queue_samples = [time_side_queue() for _ in range(5)]
    queue_ms = statistics.median(queue_samples)
    result["cross_stream_queue"].update({
        "queued_work_ms": summ(queue_samples),
        "d2d_GB_per_s": round(SIDE_COPIES * SIDE_BUFFER_BYTES /
                              (queue_ms * 1e-3) / 1e9, 1),
    })

    def probe_sync(product, pending):
        """Host wall of the deployed guard's sync point only."""
        def fn():
            peak_t = product.abs().amax(dim=tuple(range(product.ndim)))
            return float(peak_t)

        if pending:
            queue_side_copies()
        wall0 = time.perf_counter()
        peak = fn()
        wall_ms = (time.perf_counter() - wall0) * 1000.0
        torch.npu.synchronize()
        return wall_ms, peak

    for name, k, n, in_dtype in CASES:
        dtype = torch.float16 if in_dtype == "float16" else torch.float32
        torch.manual_seed(222)
        x = (torch.randn(M, k, dtype=dtype, device=device) * 0.05)
        w = (torch.randn(n, k, dtype=torch.float16, device=device) * 0.05)
        w_t = w.t()
        del w
        product = x.float()

        def guard_hostsync(x_in):
            prod = x_in.float()
            peak = float(prod.abs().amax(dim=tuple(range(prod.ndim))))
            if not math.isfinite(peak):
                raise FloatingPointError("nonfinite peak")
            exponent = max(0, math.ceil(math.log2(peak / 32752.0))) if peak > 32752.0 else 0
            scale = float(2 ** exponent)
            half_input = (prod / scale).to(torch.float16)
            out = torch.mm(half_input.reshape(-1, half_input.shape[-1]), w_t,
                           out_dtype=torch.float32)
            return out * scale

        def guard_devscale(x_in):
            """No host sync: exponent is a device scalar tensor."""
            prod = x_in.float()
            peak_t = prod.abs().amax(dim=tuple(range(prod.ndim)))
            ratio = torch.clamp(peak_t / 32752.0, min=1e-38)
            exponent_t = torch.clamp(torch.ceil(torch.log2(ratio)), min=0.0)
            scale_t = torch.pow(torch.full_like(exponent_t, 2.0), exponent_t)
            half_input = (prod / scale_t).to(torch.float16)
            out = torch.mm(half_input.reshape(-1, half_input.shape[-1]), w_t,
                           out_dtype=torch.float32)
            return out * scale_t

        for _ in range(args.warmup):
            probe_sync(product, False)
        idle = [probe_sync(product, False)[0] for _ in range(args.steps)]
        for _ in range(args.warmup):
            probe_sync(product, True)
        busy = [probe_sync(product, True)[0] for _ in range(args.steps)]

        def wall(fn):
            t = time.perf_counter()
            out = fn()
            return (time.perf_counter() - t) * 1000.0, out

        for _ in range(args.warmup):
            wall(lambda: guard_hostsync(x))
        idle_guard = [wall(lambda: guard_hostsync(x))[0] for _ in range(args.steps)]

        def busy_guard_hostsync():
            queue_side_copies()
            return wall(lambda: guard_hostsync(x))[0]

        for _ in range(args.warmup):
            busy_guard_hostsync()
        busy_guard = [busy_guard_hostsync() for _ in range(args.steps)]

        def busy_guard_devscale():
            queue_side_copies()
            t = time.perf_counter()
            out = guard_devscale(x)
            torch.npu.synchronize()  # drains queue + gemm; reported separately
            return (time.perf_counter() - t) * 1000.0, out

        for _ in range(args.warmup):
            busy_guard_devscale()
        busy_dev = [busy_guard_devscale()[0] for _ in range(args.steps)]

        torch.npu.synchronize()
        ref = guard_hostsync(x)
        dev = guard_devscale(x)
        torch.npu.synchronize()
        ref_words = ref.view(torch.int16)
        dev_words = dev.view(torch.int16)
        same = bool(torch.equal(ref_words, dev_words))
        differing = int((ref_words != dev_words).sum())
        peak_seen = float(x.abs().max())
        result["bitwise"][name] = {
            "path_a": "deployed host-sync guard",
            "path_b": "device-scale guard (no host sync)",
            "scale_branch": "scale==1" if peak_seen <= 32752.0 else "scale>1",
            "observed_peak": peak_seen,
            "bitwise_equal": same, "differing_words": differing,
            "word_count": int(ref_words.numel()),
            "approval": "none; comparison only",
        }
        del ref, dev, ref_words, dev_words

        result["host_sync_blocking"][name] = {
            "idle_wall_ms": summ(idle),
            "with_side_queue_wall_ms": summ(busy),
            "added_wall_ms_median": statistics.median(busy) - statistics.median(idle),
            "side_queue_work_ms": queue_ms,
        }
        result["guard_timing"][name] = {
            "host_wall_idle_ms": summ(idle_guard),
            "host_wall_with_side_queue_ms": summ(busy_guard),
            "host_wall_added_ms": statistics.median(busy_guard) - statistics.median(idle_guard),
            "device_scale_with_side_queue_ms": summ(busy_dev),
            "device_scale_minus_hostsync_ms": statistics.median(busy_dev) - statistics.median(busy_guard),
        }
        print("CASE", json.dumps({
            "case": name,
            "h2d_646MB_ms": round(h2d_ms, 2),
            "h2d_GBps": result["h2d_bandwidth"]["effective_GB_per_s"],
            "side_queue_ms": round(queue_ms, 2),
            "sync_idle_ms": round(statistics.median(idle), 3),
            "sync_with_queue_ms": round(statistics.median(busy), 3),
            "guard_idle_ms": round(statistics.median(idle_guard), 3),
            "guard_with_queue_ms": round(statistics.median(busy_guard), 3),
            "devscale_with_queue_ms": round(statistics.median(busy_dev), 3),
            "bitwise_equal": same,
        }, ensure_ascii=False), flush=True)
        del x, w_t, product
        torch.npu.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY", json.dumps({"out": args.out, "h2d_ms": round(h2d_ms, 2),
                                 "queue_ms": round(queue_ms, 2)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())