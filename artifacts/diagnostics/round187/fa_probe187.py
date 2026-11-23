#!/usr/bin/env python3
"""R187: what causal-attention throughput can this hardware actually reach?

The 10 s target needs the FA kernel to go from the currently achieved rate to about
69 TFLOPS/card. Before investing in a new kernel it matters whether that number is reachable
at all, so this measures the library attention path on the exact shapes the model uses:
B=1, H=7, N=41792, D=128, causal, FP16, one card.

Read-only: synthetic tensors, no model weights, no change to the model.
"""
from __future__ import annotations
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

OUT = Path('/tmp/h3_fa187/probe187.json')
B, H, N, D = 1, 7, 41792, 128
WARMUP, REPS = 3, 8

# measured in R149/R150 for the same work
CURRENT_FA_S_PER_STEP = 7.454
BLOCKS = 50


def flops_full(b, h, n, d):
    """dense attention: QK^T plus PV"""
    return 4.0 * b * h * n * n * d


def flops_causal(b, h, n, d):
    """about half the tiles are masked away"""
    return 2.0 * b * h * n * n * d


def timed(fn, reps=REPS):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), min(times)


def main():
    torch.npu.set_device(0)
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')

    rows = []

    # 1. the CANN fusion attention op, which is what the model already uses
    try:
        import torch_npu
        # sparse_mode=2 is the causal mask in this CANN version; the pre/next tockens form
        # was rejected with 161001 (param invalid)
        fn_fusion = lambda: torch_npu.npu_fusion_attention(
            q, k, v, head_num=H, input_layout='BNSD',
            scale=D ** -0.5, sparse_mode=2)
        med, best = timed(fn_fusion)
        rows.append({'path': 'npu_fusion_attention(causal bounds)', 'median_ms': round(med, 2),
                     'best_ms': round(best, 2)})
    except Exception as exc:  # noqa: BLE001
        rows.append({'path': 'npu_fusion_attention(causal bounds)', 'error': repr(exc)[:300]})

    # 2. plain non-causal fusion attention, for the dense reference number
    try:
        fn_dense = lambda: torch_npu.npu_fusion_attention(
            q, k, v, head_num=H, input_layout='BNSD', scale=D ** -0.5)
        med, best = timed(fn_dense)
        rows.append({'path': 'npu_fusion_attention(dense)', 'median_ms': round(med, 2),
                     'best_ms': round(best, 2)})
    except Exception as exc:  # noqa: BLE001
        rows.append({'path': 'npu_fusion_attention(dense)', 'error': repr(exc)[:300]})

    # 3. torch SDPA, one head at a time: this build materialises the score matrix, so all
    #    seven heads at once needs 45.55 GiB and does not fit in 32 GiB
    try:
        q1, k1, v1 = q[:, :1], k[:, :1], v[:, :1]
        fn_sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(
            q1, k1, v1, attn_mask=None, dropout_p=0.0, is_causal=True)
        med, best = timed(fn_sdpa)
        med, best = med * H, best * H   # scale one head up to the seven the model needs
        rows.append({'path': 'sdpa(is_causal=True)', 'median_ms': round(med, 2),
                     'best_ms': round(best, 2)})
    except Exception as exc:  # noqa: BLE001
        rows.append({'path': 'sdpa(is_causal=True)', 'error': repr(exc)[:300]})

    f_full = flops_full(B, H, N, D)
    f_causal = flops_causal(B, H, N, D)
    for r in rows:
        if 'median_ms' not in r:
            continue
        s = r['median_ms'] / 1000.0
        r['dense_TFLOPS_per_card'] = round(f_full / s / 1e12, 2)
        r['causal_TFLOPS_per_card'] = round(f_causal / s / 1e12, 2)
        # one block per rank costs this, and there are 50 blocks per step
        r['projected_step_s_for_50_blocks'] = round(s * BLOCKS, 2)
        r['speedup_vs_current_7.454s'] = round(CURRENT_FA_S_PER_STEP / (s * BLOCKS), 2)

    out = {
        'scope': 'R187: achievable causal attention throughput on this hardware at the model FA shapes',
        'read_only_measurement': True, 'nothing_adopted': True, 'model_math_unchanged': True,
        'synthetic_inputs': True, 'device': 'npu:0', 'dtype': 'float16',
        'shape': {'B': B, 'H': H, 'N': N, 'D': D, 'causal': True},
        'flops_dense_TF': round(f_full / 1e12, 2),
        'flops_causal_TF': round(f_causal / 1e12, 2),
        'current_measured': {'fa_s_per_step': CURRENT_FA_S_PER_STEP, 'blocks': BLOCKS,
                             'achieved_TFLOPS_per_card': 41.1,
                             'note': 'from R149/R150 for the same work'},
        'rows': rows,
        'target_TFLOPS_per_card_for_10s': 69.0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'flops_dense_TF': out['flops_dense_TF'], 'flops_causal_TF': out['flops_causal_TF']}))
    for r in rows:
        print(' ', r.get('path'), '->', r.get('median_ms', r.get('error')),
              'dense TFLOPS/card', r.get('dense_TFLOPS_per_card'),
              'causal TFLOPS/card', r.get('causal_TFLOPS_per_card'),
              'projected step', r.get('projected_step_s_for_50_blocks'))


if __name__ == '__main__':
    main()
