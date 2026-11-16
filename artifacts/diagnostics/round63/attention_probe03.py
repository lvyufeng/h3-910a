#!/usr/bin/env python3
"""Actual VAE attention padding/head partition/FA experiment on a separate card."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
os.environ.update(VLLM_OMNI_910A_NO_FUSED_NORM='1', VLLM_OMNI_910A_NO_FUSED_ACT='1',
                  VLLM_OMNI_910A_ATTN_CHUNK='128')
import torch
import torch_npu
from vllm_omni.platforms import current_omni_platform
from vllm_omni import h3_910a_sdpa_shim as shim
from vae_attention_candidates63 import eager, custom_fa_exploratory, fp32_oracle, sanitize

ROOT = Path('/tmp/h3_round63_decode01')
CAP = ROOT / 'capture' / 'rank0'
OUT = ROOT / 'attention_probe03.json'
if OUT.exists():
    raise RuntimeError('Refuse attention proof overwrite')
device = int(os.environ.get('H3_DIAGNOSTIC_DEVICE', '1'))
current_omni_platform.set_device(torch.device('npu', device))
torch.set_num_threads(8)
r = {'cases': {}, 'device': device, 'scope': 'Actual first decoder-tile/layer0 QKV; padding and FA are not assumed lossless.',
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'candidates_sha256': hashlib.sha256((ROOT / 'vae_attention_candidates63.py').read_bytes()).hexdigest()}


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    assert x.shape == y.shape and x.dtype == y.dtype
    e = x.double() - y.double()
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'elements': x.numel(), 'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'word_mismatches': int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),
            'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / y.double().square().mean().sqrt().clamp_min(1e-30))}


def timed(fn):
    first = fn()
    torch.npu.synchronize()
    groups = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(5):
            out = fn()
        torch.npu.synchronize()
        groups.append((time.perf_counter() - t) * 1000 / 5)
    return out, {'groups_ms': groups, 'median_ms': statistics.median(groups), 'first_vs_repeat': metric(out, first)}


try:
    d = torch.load(CAP / 'attention0_inputs.pt', weights_only=True)
    q, k, v = [d[key].to(torch.device('npu', device)).contiguous() for key in ('q', 'k', 'v')]
    b, n, h, width = v.shape
    backing = torch.zeros(b, n, h, 3 * width, dtype=v.dtype, device=v.device)
    vv = backing[..., 2 * width:]
    vv.copy_(v)
    v = vv
    actual = torch.load(CAP / 'attention0_output.pt', weights_only=True)
    functions = {'shim_c128': lambda: sanitize(shim._matmul_sdpa(*[x.transpose(1, 2) for x in (q, k, v)]).transpose(1, 2)),
                 'dense_c256_h32': lambda: eager(q, k, v),
                 'dense_full_h8': lambda: eager(q, k, v, query_chunk=n, head_chunk=8),
                 'dense_full_h1': lambda: eager(q, k, v, query_chunk=n, head_chunk=1),
                 'pad_query16_full_h32': lambda: eager(q, k, v, query_chunk=2048, pad_query=16),
                 'pad16_full_h32': lambda: eager(q, k, v, query_chunk=2048, pad_query=16, pad_keys=16),
                 'pad128_full_h32': lambda: eager(q, k, v, query_chunk=2048, pad_query=128, pad_keys=128),
                 'pad16_c256_h8': lambda: eager(q, k, v, query_chunk=256, head_chunk=8, pad_query=16, pad_keys=16),
                 'custom_fa_exploratory': lambda: custom_fa_exploratory(q, k, v)}
    with torch.inference_mode(), torch.autocast('npu', dtype=torch.float16):
        oracle = fp32_oracle(q, k, v).cpu().contiguous()
        r['native_capture_vs_fp32_oracle'] = metric(actual, oracle)
        for name, fn in functions.items():
            out, timing = timed(fn)
            row = {'timing': timing, 'vs_actual': metric(out, actual), 'vs_fp32_oracle': metric(out, oracle)}
            r['cases'][name] = row
            OUT.write_text(json.dumps(r, indent=2) + '\n')
            print(json.dumps({'name': name, **row}), flush=True)
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print('Evidence:', OUT, flush=True)
