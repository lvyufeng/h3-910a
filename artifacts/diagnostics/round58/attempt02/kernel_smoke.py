#!/usr/bin/env python3
"""Compile-first smoke verification for custom software BF16 and fused math."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch_npu
from fused_vector58 import launch

ROOT = Path('/tmp/h3_round58_vec01')
OUT = ROOT / 'kernel_smoke02.json'
if OUT.exists():
    raise RuntimeError('Refuse smoke evidence overwrite')
torch.npu.set_device(0)
r = {'cases': [], 'scope': 'TIK primitive smoke only; no full model or speed claim'}

def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    e = x.float() - y.float()
    return {'raw_bits_equal': x.dtype == y.dtype and torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'finite': bool(torch.isfinite(x.float()).all()), 'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / y.float().square().mean().sqrt().clamp_min(1e-12))}

try:
    with torch.inference_mode():
        # Finite raw FP32 exponent/mantissa values, including exact BF16 ties.
        rng = np.random.default_rng(58)
        bits = rng.integers(0, 0x7f7fffff, size=(8, 128), dtype=np.uint32)
        bits[1] |= np.uint32(0x80000000)
        bits[0, :8] = np.array([0, 0x80000000, 0x3f808000, 0x3f818000,
                               0xbf808000, 0xbf818000, 0x00018000, 0x007f8000], dtype=np.uint32)
        x = torch.from_numpy(bits.view(np.float32).copy())
        got, _ = launch('round', x.to('npu'), torch.ones(128, device='npu'))
        torch.npu.synchronize()
        item = {'kind': 'finite_bf16_round', **metric(got, x.bfloat16().float())}
        assert item['raw_bits_equal']
        r['cases'].append(item)
        torch.manual_seed(58)
        xc = torch.randn(8, 128).bfloat16()
        xc[0].zero_(); xc[1].fill_(1.); xc[2, 0] = 20736
        gamma = torch.randn(128).bfloat16()
        residual = torch.full_like(xc, 0.003936767578125)
        for kind in ('rms', 'addrms'):
            merged = (xc.float() + residual.float()).bfloat16() if kind == 'addrms' else xc
            xf = merged.float()
            normalized = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
            ref = (normalized.float() * gamma.float()).bfloat16()
            y, updated = launch(kind, xc.float().to('npu'), gamma.float().to('npu'),
                                residual.float().to('npu') if kind == 'addrms' else None)
            torch.npu.synchronize()
            item = {'kind': kind, 'output': metric(y.bfloat16(), ref), 'residual': metric(updated.bfloat16(), merged)}
            assert item['output']['relative_rms'] < 5e-4 and item['residual']['raw_bits_equal']
            r['cases'].append(item)
        gate = torch.randn(8, 128).bfloat16(); up = torch.randn_like(gate)
        gate[0, :8] = torch.tensor([-100., -80., -0., 0., 1., 20., 80., 100.]).bfloat16()
        ref = torch.nn.functional.silu(gate) * up
        y, _ = launch('swiglu', gate.float().to('npu'), up.float().to('npu'))
        torch.npu.synchronize()
        item = {'kind': 'swiglu', 'output': metric(y.bfloat16(), ref)}
        assert item['output']['relative_rms'] < 5e-4
        r['cases'].append(item)
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)
