#!/usr/bin/env python3
"""R229: can allocator/copy churn explain the 34.6 ms/block MLP remainder?

R228 disproved the weight-wait explanation: copies are hidden. But the MLP parent
span minus fc1/fc2 is still 34.6 ms/block (1.73 s/step) for what is only

    hidden, _ = self.fc1(x)
    gate, up = hidden.chunk(2, dim=-1)
    hidden = torch.nn.functional.silu(gate.float()) * up.float()
    out, _ = self.fc2(hidden)

i.e. two FP32 casts, one silu and one multiply on (10448, 7168) tensors: about
1.8 GB of traffic, which should cost 2-3 ms at HBM bandwidth.

This benchmark replays exactly that elementwise chain, with no model and no
weights, under four conditions to separate bandwidth from allocator behaviour:

  idle            nothing else running
  copy_alloc      a 646 MB layerwise-prefetch-sized H2D copy runs on a side
                  stream each iteration, with a fresh allocation every time
  copy_reuse      same copy, buffer reused (no allocator traffic)
  alloc_only      fresh 646 MB allocation each iteration, no copy

Measurement only; nothing is adopted.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

M = 10448
HALF = 7168          # ffn_hidden_size per half of the fused fc1 output
PREFETCH_BYTES = 646_000_000


def summarise(values):
    values = [float(v) for v in values]
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not 0 <= args.device <= 7:
        raise SystemExit("device must be an observed logical card 0..7")

    import torch
    import torch_npu as _tn

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    props = torch.npu.get_device_properties(args.device)
    side = torch.npu.Stream()

    torch.manual_seed(229)
    gate = (torch.randn(M, HALF, dtype=torch.float16, device=device) * 0.05)
    up = (torch.randn(M, HALF, dtype=torch.float16, device=device) * 0.05)
    torch.npu.synchronize()

    n_float = PREFETCH_BYTES // 4
    try:
        host_src = torch.empty(n_float, dtype=torch.float32, pin_memory=True)
        pinned = True
    except BaseException:
        host_src = torch.empty(n_float, dtype=torch.float32)
        pinned = False
    host_src.normal_(0.0, 0.01)
    reuse_dst = torch.empty(n_float, dtype=torch.float32, device=device)

    fused = (torch.randn(M, 2 * HALF, dtype=torch.float16, device=device) * 0.05)
    gate_view = fused[:, :HALF]      # same non-contiguous chunk views the model uses
    up_view = fused[:, HALF:]

    def elementwise_contig():
        return torch.nn.functional.silu(gate.float()) * up.float()

    def elementwise_chunked():
        return torch.nn.functional.silu(gate_view.float()) * up_view.float()

    def elementwise_chunked_contiguous():
        return torch.nn.functional.silu(gate_view.contiguous().float()) * up_view.contiguous().float()

    HOOKS = {"contig": elementwise_contig, "chunked": elementwise_chunked,
             "chunked_contig": elementwise_chunked_contiguous}

    def timed(fn, pre=None, post=None):
        for _ in range(args.warmup):
            if pre:
                pre()
            fn()
            if post:
                post()
        torch.npu.synchronize()
        samples = []
        for _ in range(args.steps):
            if pre:
                pre()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            if post:
                post()
            torch.npu.synchronize()
            samples.append(start.elapsed_time(end))
            del out
        return summarise(samples)

    result = {
        "schema_version": "r229b.mlp_alloc.v1",
        "host": "ts-217",
        "measurement_only": True,
        "no_model_weights_loaded": True,
        "no_model_import": True,
        "no_source_modification": True,
        "candidate_installed": False,
        "device": {"requested_id": args.device,
                   "actual_id": int(torch.npu.current_device()),
                   "name": str(getattr(props, "name", "unknown"))},
        "runtime": {"torch": torch.__version__,
                    "torch_npu": getattr(_tn, "__version__", "unknown")},
        "replayed_op": "silu(gate.float()) * up.float() on (10448, 7168) FP16 inputs",
        "element_bytes_read": 2 * M * HALF * 2,
        "fp32_temporaries_bytes": 3 * M * HALF * 4,
        "prefetch_bytes": PREFETCH_BYTES,
        "pinned": pinned,
        "idle_by_layout": {name: timed(fn) for name, fn in HOOKS.items()},
    }

    # fresh GPU allocation each iteration (mirrors gpu_weight = torch.empty(...))
    def alloc_fresh():
        return torch.empty(n_float, dtype=torch.float32, device=device)

    def alloc_only_pre():
        result.setdefault("_scratch", [])
        result["_scratch"].append(alloc_fresh())

    # copy on the side stream into a fresh allocation, then free after sync
    def copy_alloc_pre():
        dst = alloc_fresh()
        with torch.npu.stream(side):
            dst.copy_(host_src, non_blocking=True)
        result.setdefault("_scratch", []).append(dst)

    def copy_reuse_pre():
        with torch.npu.stream(side):
            reuse_dst.copy_(host_src, non_blocking=True)

    def drop_scratch():
        result["_scratch"] = []

    fn = HOOKS["chunked"]
    result["chunked_copy_reuse"] = timed(fn, pre=copy_reuse_pre)
    drop_scratch()
    result["chunked_copy_alloc"] = timed(fn, pre=copy_alloc_pre, post=drop_scratch)
    drop_scratch()
    result["chunked_alloc_only"] = timed(fn, pre=alloc_only_pre, post=drop_scratch)
    drop_scratch()
    result.pop("_scratch", None)

    idle_med = result["idle_by_layout"]["chunked"]["median"]
    for key in ("chunked_copy_reuse", "chunked_copy_alloc", "chunked_alloc_only"):
        result[key]["slowdown_x"] = round(result[key]["median"] / idle_med, 2)
        result[key]["extra_ms"] = round(result[key]["median"] - idle_med, 2)
    result["r228_mlp_remainder_ms_per_block"] = 34.6
    result["interpretation"] = {
        "idle_vs_r228_remainder_ratio": round(34.6 / idle_med, 2),
        "allocator_churn_explains_remainder": bool(result["chunked_alloc_only"]["median"] >= 0.5 * 34.6),
        "note": "the replicated chain is only the elementwise part; the in-model "
                "remainder also contains chunk/residual/Python dispatch",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps({
        "idle_by_layout_ms": {k: round(v["median"], 3) for k, v in result["idle_by_layout"].items()},
        "chunked_copy_reuse_ms": round(result["chunked_copy_reuse"]["median"], 3),
        "chunked_copy_alloc_ms": round(result["chunked_copy_alloc"]["median"], 3),
        "chunked_alloc_only_ms": round(result["chunked_alloc_only"]["median"], 3),
        "r228_remainder_ms": 34.6,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())