#!/usr/bin/env python3
"""R238: are the four in-model GEMM shapes ~3x slower when 8 ranks run concurrently?

R237 corrected the attribution: qkv 41.6 ms/block and fc1 54.4 ms/block in-model versus
12.47 and 19.04 ms for the same shapes measured standalone in R146 (2.9-3.3x), while
out_proj/fc2 match "standalone mm + TP all-reduce" (1.22x). Before calling that a
model-path problem, check the cheaper explanation: contention when all eight cards issue
the same GEMMs simultaneously.

No model, no weights. Same shapes/dtypes as R146; device events around each mm.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

M = 10448
CASES = [
    ("qkv_proj_true", 5376, 10752, "float16"),
    ("mlp_fc1_true", 5376, 14336, "float32"),
    ("attn_out_proj_true", 3584, 5376, "float32"),
    ("mlp_fc2_true", 7168, 5376, "float32"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)
    device = torch.device(f"npu:{local}")

    rows = {}
    for name, k, n, out_dtype in CASES:
        x = torch.randn(M, k, dtype=torch.float16, device=device) * 0.05
        w = torch.randn(n, k, dtype=torch.float16, device=device) * 0.05
        w_t = w.t()
        od = torch.float16 if out_dtype == "float16" else torch.float32
        def fn():
            return torch.mm(x, w_t, out_dtype=od)
        for _ in range(args.warmup):
            fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(args.steps):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.npu.synchronize()
            samples.append(start.elapsed_time(end))
        flops = 2 * M * k * n
        med = statistics.median(samples)
        rows[name] = {"world_size": world, "k": k, "n": n, "out_dtype": out_dtype,
                      "median_ms": round(med, 3), "min_ms": round(min(samples), 3),
                      "max_ms": round(max(samples), 3),
                      "TFLOPS_per_card": round(flops / (med * 1e-3) / 1e12, 1)}
        del x, w, w_t
        torch.npu.empty_cache()

    report = {"schema_version": "r238.concurrent_gemm.v1", "host": "ts-217",
              "measurement_only": True, "no_model_weights_loaded": True,
              "devices": world, "rank": rank, "rows": rows,
              "r146_single_process_reference_ms": {"qkv_proj_true": 12.47,
                                                   "mlp_fc1_true": 19.04,
                                                   "attn_out_proj_true": 2.73,
                                                   "mlp_fc2_true": 12.21}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.stem}_rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RANK", json.dumps({"rank": rank, "world": world,
                              "qkv_ms": rows["qkv_proj_true"]["median_ms"],
                              "fc1_ms": rows["mlp_fc1_true"]["median_ms"],
                              "out_proj_ms": rows["attn_out_proj_true"]["median_ms"],
                              "fc2_ms": rows["mlp_fc2_true"]["median_ms"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
