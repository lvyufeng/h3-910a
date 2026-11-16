#!/usr/bin/env python3
"""Actual VAE SDPA-shim replay; keep the failed unpatched probe intact.

Only this diagnostic process changes query chunking/layout. All candidates
retain the native FP16 QK -> FP16 scaled/clamped scores -> FP32 softmax ->
FP16 probabilities -> FP16 PV boundaries, except explicitly labelled FP32
or custom-FA exploratory cases. No production deployment is performed.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time

os.environ['VLLM_OMNI_910A_NO_FUSED_NORM'] = '1'
os.environ['VLLM_OMNI_910A_NO_FUSED_ACT'] = '1'
import torch
import torch_npu
# The old probe omitted this import, so it called unsupported native SDPA.
import vllm_omni
from vllm_omni import h3_910a_sdpa_shim as shim
from vllm_omni.platforms import current_omni_platform
from torch.nn import functional as F

ROOT = Path('/tmp/h3_round63_decode01')
CAP = ROOT / 'capture' / 'rank0'
OUT = ROOT / 'attention_probe02.json'
if OUT.exists():
    raise RuntimeError('Refuse attention proof overwrite')
current_omni_platform.set_device(torch.device('npu', 0))
torch.set_num_threads(8)
r = {'cases': {}, 'scope': 'Actual first decoder-tile/layer0 QKV; no full-tile/video acceptance implied.',
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'shim_sha256': hashlib.sha256(Path(inspect.getfile(shim)).read_bytes()).hexdigest(),
     'sdpa_module': F.scaled_dot_product_attention.__module__,
     'context': {'allow_internal_format': getattr(torch.npu.config, 'allow_internal_format', 'write_only_unavailable'),
                 'platform_set_device_called': True,
                 'matmul_allow_hf32': torch.npu.matmul.allow_hf32,
                 'conv_allow_hf32': torch.npu.conv.allow_hf32}}
assert F.scaled_dot_product_attention is shim._shim


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    assert x.shape == y.shape and x.dtype == y.dtype
    e = x.double() - y.double()
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'elements': x.numel(), 'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'word_mismatches': int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),
            'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / y.double().square().mean().sqrt().clamp_min(1e-30))}


def sanitize(x):
    return torch.where(torch.isfinite(x), x, torch.zeros((), dtype=x.dtype, device=x.device))


def native(q, k, v, chunk=128, layout='original'):
    os.environ['VLLM_OMNI_910A_ATTN_CHUNK'] = str(chunk)
    if layout == 'bsnd_contiguous':
        q, k, v = [x.contiguous() for x in (q, k, v)]
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    if layout == 'bhsd_contiguous':
        q, k, v = [x.contiguous() for x in (q, k, v)]
    return sanitize(F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False).transpose(1, 2))


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
    return out, {'groups_ms': groups, 'median_ms': statistics.median(groups),
                 'first_vs_repeat': metric(out, first)}


try:
    d = torch.load(CAP / 'attention0_inputs.pt', weights_only=True)
    q, k, v = [d[x].to('npu').contiguous() for x in ('q', 'k', 'v')]
    # Captures are dense CPU copies. Recreate the real interleaved value view.
    b, n, h, width = v.shape
    backing = torch.zeros(b, n, h, 3 * width, dtype=v.dtype, device=v.device)
    vv = backing[..., 2 * width:]
    vv.copy_(v)
    assert torch.equal(vv.cpu(), v.cpu())
    v = vv
    metadata = json.loads((CAP / 'decode_profile.json').read_text())
    r['input_meta'] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                            'stride': list(value.stride()), 'storage_offset': value.storage_offset(),
                            'format': torch_npu.get_npu_format(value)}
                       for key, value in zip(('q', 'k', 'v'), (q, k, v))}
    assert metadata['attention0_pack_info'] == {}
    for key in ('q', 'k', 'v'):
        assert r['input_meta'][key]['stride'] == metadata['attention0_input_meta'][key]['stride']
    actual = torch.load(CAP / 'attention0_output.pt', weights_only=True)
    functions = {'shim_original_c128': lambda: native(q, k, v)}
    for chunk in (256, 512, 1024, n):
        functions['shim_original_c%d' % chunk] = lambda chunk=chunk: native(q, k, v, chunk)
    for layout in ('bsnd_contiguous', 'bhsd_contiguous'):
        for chunk in (128, n):
            functions['shim_%s_c%d' % (layout, chunk)] = lambda layout=layout, chunk=chunk: native(q, k, v, chunk, layout)
    baseline = None
    with torch.inference_mode(), torch.autocast('npu', dtype=torch.float16):
        for name, fn in functions.items():
            out, timing = timed(fn)
            if baseline is None:
                baseline = out.detach().cpu().contiguous()
            row = {'timing': timing, 'vs_actual': metric(out, actual), 'vs_same_input_baseline': metric(out, baseline)}
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
