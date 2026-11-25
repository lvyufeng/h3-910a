#!/usr/bin/env python3
"""R265: bitwise-preserving dual-stream FA experiment (head split) vs single-stream.

Single launch inside one kernel is vector-then-cube serial (S1: vec 54.8%, cube MAC 22.7%).
Hypothesis: two concurrent launches covering disjoint head groups, each with half the core
budget, may interleave one kernel's vector phase with the other's cube phase.

No numerics change is involved: heads are independent, so the concatenated output must be
bit-identical to the single launch. That is asserted here.
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


def timed(fn, reps=3):
    ts = []
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    torch.manual_seed(20261124)
    B, H, N, D = 1, 7, 41792, 128
    q = (torch.randn(B, H, N, D, dtype=torch.float16, device="npu") * 0.05)
    k = (torch.randn(B, H, N, D, dtype=torch.float16, device="npu") * 0.05)
    v = (torch.randn(B, H, N, D, dtype=torch.float16, device="npu") * 0.05)
    rep = {"schema": "r265.fa_dual_stream.v1", "shape": [B, H, N, D],
           "tiles": [int(__import__("os").environ.get("H3_FA_BR_CAP", 128)),
                     int(__import__("os").environ.get("H3_FA_BC_CAP", 304))]}

    # ---- single stream, full 30-core budget -----------------------------------
    full, t_single = timed(lambda: FA.flash_attention_910a(q, k, v), a.reps)
    torch.npu.synchronize()
    rep["single_stream_ms"] = round(statistics.median(t_single), 2)
    rep["single_stream_all_ms"] = [round(t, 2) for t in t_single]

    # ---- dual stream, head split 4 + 3, half core budget each ----------------
    real_budget = FA._runtime_core_budget
    try:
        FA._runtime_core_budget = lambda: 15
        s1, s2 = torch.npu.Stream(), torch.npu.Stream()

        def run_dual():
            with torch.npu.stream(s1):
                o1 = FA.flash_attention_910a(q[:, :4], k[:, :4], v[:, :4])
            with torch.npu.stream(s2):
                o2 = FA.flash_attention_910a(q[:, 4:], k[:, 4:], v[:, 4:])
            torch.npu.synchronize()
            torch.npu.current_stream().wait_stream(s1)
            torch.npu.current_stream().wait_stream(s2)
            return torch.cat([o1, o2], dim=1)

        dual, t_dual = timed(run_dual, a.reps)
        rep["dual_stream_ms"] = round(statistics.median(t_dual), 2)
        rep["dual_stream_all_ms"] = [round(t, 2) for t in t_dual]
        rep["speedup"] = round(rep["single_stream_ms"] / rep["dual_stream_ms"], 3)
        rep["bit_identical"] = bool(torch.equal(full, dual))
        if not rep["bit_identical"]:
            diff = (full.float() - dual.float()).abs()
            rep["max_abs_diff"] = float(diff.max().item())
            rep["mismatched_words"] = int((full != dual).sum().item())
    except BaseException as exc:  # pragma: no cover
        rep["dual_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        FA._runtime_core_budget = real_budget

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
