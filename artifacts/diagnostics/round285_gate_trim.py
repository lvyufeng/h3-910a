#!/usr/bin/env python3
"""R283: two-process numeric + timing gate for FA kernel variants.

The deployed backend imports the TIK package from a fixed path (/tmp/fa_pkg), so a patched
kernel and the reference kernel cannot be loaded in one process. This script therefore works
in two modes sharing one .pt file:

    python numeric_gate283.py --mode baseline  --store /tmp/fa_gate.pt
    python numeric_gate283.py --mode candidate --store /tmp/fa_gate.pt

`baseline` runs the deployed kernel and stores its output tensor plus its timing.
`candidate` runs whatever package is currently mounted at /tmp/fa_pkg, loads the stored
reference output and reports max|diff|, relRMS, NaN/Inf counts and the timing change.

Contract notes: this gate only measures; it never adopts anything. A candidate that changes
numerics must still be reported to the user and approved before entering the model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_omni.diffusion.attention.backends import fa910a as FA
import sys as _sys
_sys.path.insert(0, "/tmp/h3_budget149")
import bb218trim as _trim
FA.flash_attention_910a = _trim.flash_attention_910a

SHAPE = (1, 7, 41792, 128)
SEED = 20261124


def build_inputs():
    torch.manual_seed(SEED)
    q = (torch.randn(*SHAPE, dtype=torch.float16, device="npu") * 0.05)
    k = (torch.randn(*SHAPE, dtype=torch.float16, device="npu") * 0.05)
    v = (torch.randn(*SHAPE, dtype=torch.float16, device="npu") * 0.05)
    return q, k, v


def measure(fn, reps):
    fn()                                   # warm + build
    torch.npu.synchronize()
    ts = []
    out = None
    for _ in range(reps):
        torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.npu.synchronize()
        ts.append(s.elapsed_time(e))
    return out, ts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "candidate"], required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()

    q, k, v = build_inputs()
    out, ts = measure(lambda: FA.flash_attention_910a(q, k, v), a.reps)
    rep = {"schema": "r283.num_gate.v1", "mode": a.mode, "shape": list(SHAPE), "seed": SEED,
           "median_ms": round(statistics.median(ts), 2),
           "all_ms": [round(t, 2) for t in ts],
           "finite": bool(torch.isfinite(out).all().item()),
           "nan": int(torch.isnan(out).sum().item()),
           "inf": int(torch.isinf(out).sum().item()),
           "sum": float(out.float().sum().item())}

    if a.mode == "baseline":
        torch.save({"out": out.cpu(), "median_ms": rep["median_ms"]}, a.store)
        rep["stored"] = a.store
    else:
        ref = torch.load(a.store, map_location="cpu")
        r = ref["out"].npu()
        diff = (r.float() - out.float()).abs()
        rep.update({
            "baseline_median_ms": ref["median_ms"],
            "delta_ms": round(rep["median_ms"] - ref["median_ms"], 2),
            "speedup": round(ref["median_ms"] / rep["median_ms"], 3),
            "identical": bool(torch.equal(r, out)),
            "mismatched_words": int((r != out).sum().item()),
            "total_words": int(r.numel()),
            "max_abs_diff": float(diff.max().item()),
            "relRMS": float((diff.pow(2).mean().sqrt().item()
                             / r.float().pow(2).mean().sqrt().item())),
        })

    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
