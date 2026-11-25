#!/usr/bin/env python3
"""R256 S2: Triton-CANN flash-attention forward prototype for the real DiT shape.

Purpose: feasibility measurement only. Deployed kernel is TIK-generated and takes
152.2 ms/call at B=1,H=7,N=41792,D=128; S1 showed it is vector-bound (vec 54.8%,
cube MAC 22.7%). This prototype asks whether a Triton-CANN implementation with
software-pipelined tiles can raise cube occupancy enough to matter.

Semantics mirrored from the deployed kernel: dense (non-causal) attention, FP16 Q/K/V,
FP32 online softmax, FP32 PV accumulation, FP16 output, no 1/sqrt(d) (caller scales Q).
Numerics are NOT expected to match the TIK kernel bitwise; this is a speed probe only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _fa_fwd(Q, K, V, O, N: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_mask = offs_m < N

    q_ptrs = Q + pid_bh * N * D + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    for start_n in range(0, N, BN):
        offs_n = start_n + tl.arange(0, BN)
        col_mask = offs_n < N
        k_ptrs = K + pid_bh * N * D + offs_n[None, :] * D + offs_d[:, None]
        v_ptrs = V + pid_bh * N * D + offs_n[:, None] * D + offs_d[None, :]
        k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

        s = tl.dot(q, k).to(tl.float32)
        s = tl.where(col_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = O + pid_bh * N * D + offs_m[:, None] * D + offs_d[None, :]
    tl.store(o_ptrs, out.to(tl.float16), mask=row_mask[:, None])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--n", type=int, default=41792)
    ap.add_argument("--h", type=int, default=7)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--arch", default="",
                    help="override Triton/ bishengir target arch, e.g. Ascend910B4")
    args = ap.parse_args()

    if args.arch:
        # bishengir accepts Ascend910B1/B2/B3/B4 but torch_npu reports plain "Ascend910B"
        import triton.runtime.driver as _drv
        try:
            _drv.active.utils.get_arch = lambda: args.arch
        except BaseException:
            pass

    torch.manual_seed(20261125)
    b, h, n, d = 1, args.h, args.n, args.d
    q = torch.randn(b, h, n, d, dtype=torch.float16, device="npu") * 0.05
    k = torch.randn(b, h, n, d, dtype=torch.float16, device="npu") * 0.05
    v = torch.randn(b, h, n, d, dtype=torch.float16, device="npu") * 0.05
    o = torch.empty(b, h, n, d, dtype=torch.float16, device="npu")

    grid = (triton.cdiv(n, args.bm), b * h)
    report = {"schema": "r256.triton_fa.v1", "measurement_only": True,
              "semantics": "dense, FP16 in, FP32 softmax/PV, FP16 out, no 1/sqrt(d)",
              "shape": {"b": b, "h": h, "n": n, "d": d},
              "tiles": {"BM": args.bm, "BN": args.bn, "num_stages": args.stages},
              "grid": list(grid), "triton": triton.__version__}

    t0 = time.perf_counter()
    _fa_fwd[grid](q, k, v, o, N=n, D=d, BM=args.bm, BN=args.bn, num_stages=args.stages)
    torch.npu.synchronize()
    report["compile_s"] = round(time.perf_counter() - t0, 2)

    for _ in range(2):
        _fa_fwd[grid](q, k, v, o, N=n, D=d, BM=args.bm, BN=args.bn, num_stages=args.stages)
    torch.npu.synchronize()

    times = []
    for _ in range(args.reps):
        torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        _fa_fwd[grid](q, k, v, o, N=n, D=d, BM=args.bm, BN=args.bn, num_stages=args.stages)
        e.record()
        torch.npu.synchronize()
        times.append(s.elapsed_time(e))

    flops = 4 * n * n * d * h
    med = statistics.median(times)
    report.update({
        "median_ms": round(med, 2), "best_ms": round(min(times), 2),
        "times_ms": [round(t, 2) for t in times],
        "TFLOPS_per_card": round(flops / (med * 1e-3) / 1e12, 1),
        "tik_reference_ms": 152.19,
        "speedup_vs_tik": round(152.19 / med, 3) if med else None,
        "output_finite": bool(torch.isfinite(o).all().item()),
        "output_sum": float(o.float().sum().item()),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
