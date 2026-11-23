#!/usr/bin/env python3
"""R197c: the one untried variant - sparse_mode with an explicit causal atten_mask.

R197 declared the official prompt_flash_attention unusable after 561103 at every length, but the
calls omitted atten_mask. causal modes sometimes require it. This settles param-vs-binary.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

OUT = Path('/tmp/h3_pfa197/pfa197c.json')
B, H, N, D = 1, 7, 4096, 128


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    scale = D ** -0.5

    # causal mask in the uint8/bool form these ops expect: True/1 means masked out
    i = torch.arange(N, device='npu')
    mask_bool = (i[None, :] > i[:, None]).to(torch.uint8) * 255   # upper triangle = masked
    mask_bool = mask_bool.unsqueeze(0).unsqueeze(0)

    rows = []
    variants = [
        ('sparse_mode=2 + atten_mask(uint8)', dict(atten_mask=mask_bool, sparse_mode=2)),
        ('sparse_mode=3 + atten_mask(uint8)', dict(atten_mask=mask_bool, sparse_mode=3)),
        ('sparse_mode=1 + atten_mask(uint8)', dict(atten_mask=mask_bool, sparse_mode=1)),
        ('default sparse_mode=0 + atten_mask', dict(atten_mask=mask_bool, sparse_mode=0)),
        ('sparse_mode=2 bool mask', dict(atten_mask=mask_bool.bool(), sparse_mode=2)),
    ]
    for name, extra in variants:
        try:
            fn = lambda: torch_npu.npu_prompt_flash_attention(
                q, k, v, num_heads=H, input_layout='BNSD', scale_value=scale,
                pre_tokens=2147483647, next_tokens=0, **extra)
            out = fn()
            torch.npu.synchronize()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            rows.append({'variant': name, 'ok': True, 'ms': round(ms, 2),
                         'finite': bool(torch.isfinite(out).all().item())})
        except Exception as exc:  # noqa: BLE001
            rows.append({'variant': name, 'ok': False, 'error': repr(exc)[:150]})
        print(json.dumps(rows[-1]), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'scope': 'R197c: sparse_mode variants with an explicit atten_mask',
                               'N': N, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
