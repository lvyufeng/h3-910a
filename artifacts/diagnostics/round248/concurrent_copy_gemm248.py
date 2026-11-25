#!/usr/bin/env python3
"""R248: does the block's weight H2D traffic slow the GEMMs?

Corrected attribution (R237 retracted, R240 cross-checked) gives these in-model vs
standalone ratios on the four linears:

    qkv      17.07 / 12.47 = 1.37x
    fc1      23.40 / 19.04 = 1.23x
    out_proj 18.22 / (2.73 + 12.20 all-reduce) = 1.22x
    fc2      30.05 / (12.21 + 12.20 all-reduce) = 1.23x

R238 showed eight-rank concurrency alone does not inflate these GEMMs (<=0.4%). What is
still different in-model is the 646 MB per-block weight copy running on the copy stream
next to the GEMMs. This benchmark measures each true shape with and without a concurrent
646 MB pinned H2D copy on all eight ranks, keeping the copy off the compute stream.

No model, no weights: only torch buffers of the real byte sizes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

M = 10448
CASES = [
    ("qkv_proj_true", 5376, 10752, "float16"),
    ("mlp_fc1_true", 5376, 14336, "float32"),
    ("attn_out_proj_true", 3584, 5376, "float32"),
    ("mlp_fc2_true", 7168, 5376, "float32"),
]
COPY_BYTES = 646 * 1024 * 1024   # one block's streamed weights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local)
    device = torch.device(f"npu:{local}")
    copy_stream = torch.npu.Stream(device=device)

    host_src = torch.empty(COPY_BYTES, dtype=torch.uint8, pin_memory=True)
    dev_dst = torch.empty(COPY_BYTES, dtype=torch.uint8, device=device)

    rows = {}
    for name, k, n, out_dtype in CASES:
        x = torch.randn(M, k, dtype=torch.float16, device=device) * 0.05
        w = torch.randn(n, k, dtype=torch.float16, device=device) * 0.05
        od = torch.float16 if out_dtype == "float16" else torch.float32
        wt = w.t()

        def run_once():
            return torch.mm(x, wt, out_dtype=od)

        for _ in range(3):
            run_once()
        torch.npu.synchronize()

        def measure(with_copy):
            samples = []
            for _ in range(args.reps):
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                if with_copy:
                    with torch.npu.stream(copy_stream):
                        dev_dst.copy_(host_src, non_blocking=True)
                start.record()
                run_once()
                end.record()
                torch.npu.synchronize()
                samples.append(start.elapsed_time(end))
            return samples

        base = measure(False)
        contended = measure(True)
        flops = 2 * M * k * n
        med_b = statistics.median(base)
        med_c = statistics.median(contended)
        rows[name] = {
            "world_size": world, "k": k, "n": n, "out_dtype": out_dtype,
            "solo_ms": round(med_b, 3), "with_646mb_copy_ms": round(med_c, 3),
            "ratio": round(med_c / med_b, 3),
            "solo_TFLOPS_per_card": round(flops / (med_b * 1e-3) / 1e12, 1),
            "contended_TFLOPS_per_card": round(flops / (med_c * 1e-3) / 1e12, 1),
        }
        del x, w, wt
        torch.npu.empty_cache()
    del dev_dst

    report = {
        "schema_version": "r248.gemm_vs_weight_copy.v1", "host": "ts-217",
        "measurement_only": True, "no_model_weights_loaded": True,
        "rank": rank, "world": world, "copy_bytes": COPY_BYTES,
        "r146_standalone_reference_ms": {"qkv_proj_true": 12.47, "mlp_fc1_true": 19.04,
                                         "attn_out_proj_true": 2.73, "mlp_fc2_true": 12.21},
        "in_model_ms": {"qkv_proj_true": 17.07, "mlp_fc1_true": 23.40,
                        "attn_out_proj_true": 18.22, "mlp_fc2_true": 30.05},
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.stem}_rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RANK", json.dumps({"rank": rank, "world": world, "rows": {
        k: {"solo": v["solo_ms"], "contended": v["with_646mb_copy_ms"], "ratio": v["ratio"]}
        for k, v in rows.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())