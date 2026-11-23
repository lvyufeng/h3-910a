#!/usr/bin/env python3
"""R197: can the official AscendC prompt_flash_attention replace the legacy TIK FA kernel?

The model currently runs a legacy TIK-templated FA kernel whose per-tile serialisation R196 traced
to a single Sij buffer plus disable_sync=False barriers. libopapi exposes aclnnPromptFlashAttention
and torch_npu exposes npu_prompt_flash_attention, both modern AscendC implementations. If either
serves this shape, the win is a library swap instead of a multi-day kernel rewrite.

Read-only: synthetic tensors, nothing adopted.
"""
from __future__ import annotations
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

OUT = Path('/tmp/h3_pfa197/pfa197.json')
B, H, N, D = 1, 7, 41792, 128
WARMUP, REPS = 2, 5


def timed(fn):
    for _ in range(WARMUP):
        fn()
    import torch
    torch.npu.synchronize()
    ts = []
    for _ in range(REPS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts), ts


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    q = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    scale = D ** -0.5

    rows = []

    # the torch_npu docstring/kwarg form; head_num + input_layout are the documented names
    # exact schema (from torch.ops.npu.npu_prompt_flash_attention.default._schema):
    #   (query, key, value, *, ..., int num_heads=1, float scale_value=1.,
    #    int pre_tokens=2**31-1, int next_tokens=0, str input_layout="BSH",
    #    int num_key_value_heads=0, int[]? actual_seq_lengths_kv, int sparse_mode=0) -> Tensor
    def pfa(**kw):
        return torch_npu.npu_prompt_flash_attention(q, k, v, **kw)

    attempts = [
        ('pfa BNSD sparse_mode=2 causal',
         lambda: pfa(num_heads=H, input_layout='BNSD', scale_value=scale,
                     pre_tokens=2147483647, next_tokens=0, sparse_mode=2)),
        ('pfa BNSD sparse_mode=0 dense',
         lambda: pfa(num_heads=H, input_layout='BNSD', scale_value=scale,
                     pre_tokens=2147483647, next_tokens=0, sparse_mode=0)),
        ('pfa BSH sparse_mode=2 causal',
         lambda: torch_npu.npu_prompt_flash_attention(
             q.transpose(1, 2).reshape(B, N, H * D),
             k.transpose(1, 2).reshape(B, N, H * D),
             v.transpose(1, 2).reshape(B, N, H * D),
             num_heads=H, input_layout='BSH', scale_value=scale,
             pre_tokens=2147483647, next_tokens=0, sparse_mode=2)),
    ]

    for name, fn in attempts:
        try:
            r = fn()
            if r is None:
                rows.append({'path': name, 'error': 'op missing'}); continue
            out = r[0] if isinstance(r, (tuple, list)) else r
            med, best, ts = timed(fn)
            flops_causal = 2.0 * B * H * N * N * D
            rows.append({'path': name, 'median_ms': round(med, 2), 'best_ms': round(best, 2),
                         'causal_TFLOPS_per_card': round(flops_causal / (med/1000) / 1e12, 2),
                         'projected_fa_s_per_step': round(med * 50 / 1000, 3),
                         'vs_current_7_454': round(7.454 / (med*50/1000), 3),
                         'finite': bool(torch.isfinite(out).all().item())})
        except Exception as exc:  # noqa: BLE001
            rows.append({'path': name, 'error': repr(exc)[:240]})

    out = {
        'scope': 'R197: whether the official AscendC prompt_flash_attention can replace the TIK FA kernel',
        'read_only_measurement': True, 'nothing_adopted': True, 'model_math_unchanged': True,
        'shape': {'B': B, 'H': H, 'N': N, 'D': D, 'causal': True},
        'current_tik_kernel': {'fa_s_per_step': 7.454, 'blocks': 50,
                               'per_call_ms': round(7.454/50*1000, 2)},
        'aclnn_symbols_present': ['aclnnPromptFlashAttention', 'aclnnPromptFlashAttentionGetWorkspaceSize'],
        'attempts': rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    for r in rows:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
