#!/usr/bin/env python3
"""R197b: isolate why aclnnPromptFlashAttentionV3 fails with 561103.

41792 = 64 x 653, so it is not a multiple of 128. This sweeps aligned and unaligned lengths to
separate "the op cannot serve this length" from "the op cannot run at all", and times whatever
works so the result can be compared with the TIK kernel's 149 ms per call.
"""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path

OUT = Path('/tmp/h3_pfa197/pfa197b.json')
B, H, D = 1, 7, 128
SHAPES = [4096, 8192, 16384, 41792, 43776]


def timed(fn, warm=1, reps=3):
    import torch
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    rows = []
    for n in SHAPES:
        row = {'N': n, 'N_aligned_128': (n % 128 == 0)}
        try:
            q = torch.randn((B, H, n, D), dtype=torch.float16, device='npu')
            k = torch.randn((B, H, n, D), dtype=torch.float16, device='npu')
            v = torch.randn((B, H, n, D), dtype=torch.float16, device='npu')
            fn = lambda: torch_npu.npu_prompt_flash_attention(
                q, k, v, num_heads=H, input_layout='BNSD', scale_value=D ** -0.5,
                pre_tokens=2147483647, next_tokens=0, sparse_mode=2)
            out = fn()
            torch.npu.synchronize()
            med = timed(fn)
            # causal work: about half of the dense 4*B*H*N^2*D
            causal_tf = 2.0 * B * H * n * n * D / 1e12
            row.update({'ok': True, 'median_ms': round(med, 2),
                        'causal_TFLOPS_per_card': round(causal_tf / (med / 1000), 2),
                        'finite': bool(torch.isfinite(out).all().item())})
            del q, k, v, out
            torch.npu.empty_cache()
        except Exception as exc:  # noqa: BLE001
            row.update({'ok': False, 'error': repr(exc)[:160]})
        rows.append(row)
        print(json.dumps(row), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'scope': 'R197b: which sequence lengths aclnnPromptFlashAttentionV3 accepts',
                               'shape': {'B': B, 'H': H, 'D': D},
                               'reference_tik_kernel_ms_at_41792': 149.08,
                               'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
