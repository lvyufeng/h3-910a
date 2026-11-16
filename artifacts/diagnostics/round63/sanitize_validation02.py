#!/usr/bin/env python3
"""Exhaustively validate native-half sanitizer and isolate its VAE cost."""
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
from sanitize_vector63 import sanitize as fused
from evidence_io63 import EvidenceWriter

ROOT = Path('/tmp/h3_round63_decode01')
CAP = ROOT / 'capture' / 'rank0'
OUT = ROOT / 'sanitize_validation02.json'
writer = EvidenceWriter(OUT)
r = {'cases': {}, 'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ('sanitize_validation02.py', 'sanitize_vector63.py', 'evidence_io63.py')},
     'scope': 'Documented FP16 finite-select contract and finite actual VAE attention. Native NPU nonfinite-word behavior is separately retained; no full video/service acceptance yet.'}


def native(x):
    return torch.where(torch.isfinite(x), x, torch.zeros((), dtype=x.dtype, device=x.device))


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    assert x.shape == y.shape and x.dtype == y.dtype
    e = x.double() - y.double()
    all_finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'elements': x.numel(), 'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'word_mismatches': int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),
            'max_abs_error': float(e.abs().max()) if all_finite else 'nonfinite',
            'relative_rms': float(e.square().mean().sqrt() / y.double().square().mean().sqrt().clamp_min(1e-30)) if all_finite else 'nonfinite'}


def timed(fn):
    first = fn()
    torch.npu.synchronize()
    groups = []
    for _ in range(3):
        torch.npu.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            out = fn()
        torch.npu.synchronize()
        groups.append((time.perf_counter() - t) * 1000 / 10)
    return out, {'groups_ms': groups, 'median_ms': statistics.median(groups), 'first_vs_repeat': metric(out, first)}


try:
    current_omni_platform.set_device(torch.device('npu', 0))
    torch.set_num_threads(8)
    with torch.inference_mode():
        bits = torch.arange(65536, dtype=torch.int32)
        exhaustive = bits.to(torch.int16).view(torch.float16).to('npu')
        expected_bits = torch.where((bits & 0x7c00) != 0x7c00, bits, torch.zeros_like(bits))
        expected = expected_bits.to(torch.int16).view(torch.float16)
        eager = native(exhaustive)
        out = fused(exhaustive)
        torch.npu.synchronize()
        r['exhaustive'] = {'all_half_bit_patterns': 65536, 'nonfinite_words': 2048,
                            'vs_integer_bit_oracle': metric(out, expected), 'vs_native_npu': metric(out, eager)}
        r['exhaustive']['native_npu_matches_documented_finite_select'] = r['exhaustive']['vs_native_npu']['raw_bits_equal']
        r['exhaustive']['native_output_vs_unfiltered_words'] = metric(eager, exhaustive)
        valid = (bits & 0x7c00) != 0x7c00
        r['exhaustive']['finite_words_vs_native'] = metric(out.cpu()[valid], eager.cpu()[valid])
        assert r['exhaustive']['vs_integer_bit_oracle']['raw_bits_equal']
        assert r['exhaustive']['finite_words_vs_native']['raw_bits_equal']
        writer.write(r)
        data = torch.load(CAP / 'attention0_inputs.pt', weights_only=True)
        q, k, dense_v = [data[key].to('npu').contiguous() for key in ('q', 'k', 'v')]
        backing = torch.zeros(1, 1797, 32, 192, dtype=torch.float16, device='npu')
        v = backing[..., 128:]
        v.copy_(dense_v)
        actual = torch.load(CAP / 'attention0_output.pt', weights_only=True)
        def raw_attention():
            return shim._matmul_sdpa(*[x.transpose(1, 2) for x in (q, k, v)]).transpose(1, 2)
        with torch.autocast('npu', dtype=torch.float16):
            raw = raw_attention()
            r['actual_raw_output_nonfinite_count_cpu'] = int((~torch.isfinite(raw.cpu())).sum())
            assert r['actual_raw_output_nonfinite_count_cpu'] == 0
            r['raw_output_stride'] = list(raw.stride())
            r['raw_output_format'] = torch_npu.get_npu_format(raw)
            for name, fn in (('sanitize_only_native', lambda: native(raw)),
                             ('sanitize_only_tbe', lambda: fused(raw)),
                             ('full_attention_native', lambda: native(raw_attention())),
                             ('full_attention_tbe_sanitize', lambda: fused(raw_attention()))):
                value, timing = timed(fn)
                row = {'timing': timing, 'vs_actual': metric(value, actual)}
                r['cases'][name] = row
                writer.write(r)
                print(json.dumps({'case': name, **row}), flush=True)
                assert row['vs_actual']['raw_bits_equal'] and timing['first_vs_repeat']['raw_bits_equal']
        r['completed'] = True
        r['production_installed'] = False
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print(json.dumps(r, indent=2), flush=True)
    print('Evidence:', OUT, flush=True)
