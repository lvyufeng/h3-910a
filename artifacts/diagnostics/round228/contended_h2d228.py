#!/usr/bin/env python3
"""R228: does H2D bandwidth collapse under 8-rank concurrency and compute?

Why: R226 located 3.37 s/step in the attention/MLP parent spans and hypothesised
exposed weight-prefetch wait. The bound is arithmetic: exposed wait cannot exceed
the copy work itself, and one 646 MB prefetch x 50 takes 1.52 s/step at the idle
21.3 GB/s measured in R222. For the hypothesis to explain 3.37 s, in-model copies
must run at <= ~9.6 GB/s, i.e. 2.2x slower than idle.

This measurement uses no model and no weights: 8 ranks each copy 646 MB of pinned
host memory to their device, (A) with every rank copying concurrently, (B) while a
sustained HBM-bound matmul runs on the same device. Both use device events on the
copy stream; nothing about the model is touched.

Launch: torch_npu_run --standalone --nproc-per-node 8 contended_h2d228.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

BYTES = 646_000_000
COPIES = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bytes", type=int, default=BYTES)
    ap.add_argument("--copies", type=int, default=COPIES)
    ap.add_argument("--matmul-ms", type=float, default=4000.0,
                    help="approx duration of queued compute work for the contended phase")
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)
    device = torch.device(f"npu:{local}")

    n_float = args.bytes // 4
    src = torch.empty(n_float, dtype=torch.float32, pin_memory=True)
    src.normal_(0.0, 0.01)
    dst = torch.empty(n_float, dtype=torch.float32, device=device)
    copy_stream = torch.npu.Stream()
    torch.npu.synchronize()

    def timed_copies(n):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(copy_stream):
            start.record()
            for _ in range(n):
                dst.copy_(src, non_blocking=True)
            end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) / n  # ms per copy

    # warm up the copy path and the compute path
    timed_copies(2)
    a = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    torch.mm(a, a)
    torch.npu.synchronize()

    def contended_copies(n, matmul_ms):
        # queue sustained HBM-bound compute on the default stream first
        per_mm = 2048 * 2048 * 2048 * 2 / 90e12 * 1000.0  # ms, ~90 TFLOPS assumption
        reps = max(1, int(matmul_ms / max(per_mm, 0.1)))
        t0 = torch.npu.Event(enable_timing=True)
        for _ in range(reps):
            torch.mm(a, a)
        # copies contend with that compute for HBM/host bandwidth
        sample = timed_copies(n)
        t0.record()
        torch.npu.synchronize()
        return sample, reps

    dist.barrier()
    idle = [timed_copies(args.copies) for _ in range(3)]
    dist.barrier()
    contended, reps = contended_copies(args.copies, args.matmul_ms)
    dist.barrier()

    idle_ms = statistics.median(idle)
    report = {
        "schema_version": "r228.contended_h2d.v1",
        "host": "ts-217",
        "measurement_only": True,
        "no_model_weights_loaded": True,
        "no_model_import": True,
        "rank": rank, "world_size": dist.get_world_size(),
        "bytes_per_copy": args.bytes, "copies_per_sample": args.copies,
        "idle_copy_ms": {"samples": idle, "median": idle_ms,
                         "GB_per_s": round(args.bytes / (idle_ms * 1e-3) / 1e9, 2)},
        "contended_copy_ms": {
            "median": contended, "reps_of_2048_mm": reps,
            "GB_per_s": round(args.bytes / (contended * 1e-3) / 1e9, 2),
            "slowdown_x": round(contended / idle_ms, 2),
        },
        "implications": {
            "idle_copy_work_per_step_s": round(50 * idle_ms / 1000, 3),
            "contended_copy_work_per_step_s": round(50 * contended / 1000, 3),
            "r226_unattributed_attn_mlp_s": 3.37,
            "weight_wait_can_explain_all_of_r226": bool(50 * contended / 1000 >= 3.37),
        },
    }
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, report["idle_copy_ms"]["median"])
    if rank == 0:
        print("RESULT " + json.dumps({
            "idle_ms_all_ranks": [round(x, 2) for x in gathered],
            "idle_GBps": report["idle_copy_ms"]["GB_per_s"],
            "contended_ms": round(contended, 2),
            "contended_GBps": report["contended_copy_ms"]["GB_per_s"],
            "slowdown_x": report["contended_copy_ms"]["slowdown_x"],
            "copy_work_per_step_contended_s": report["implications"]["contended_copy_work_per_step_s"],
        }), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())