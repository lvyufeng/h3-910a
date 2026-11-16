#!/usr/bin/env python3
"""Native reference checks before any full text integration; no model changes."""
import hashlib
import json
from pathlib import Path
import time
import torch
import torch_npu
from norm_tail67 import norm_tail
from evidence_io67 import EvidenceWriter

ROOT = Path('/tmp/h3_round67_norm01')
writer = EvidenceWriter(ROOT / 'validation01.json')
r = {'cases': [], 'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('norm_tail67.py', 'validate_tail67.py')},
     'scope': 'Same-input finite random/special fixtures with native FP32 statistics. Exhaustive BF16 word probe is pointwise identity only, not actual normalized model activations.'}
torch.npu.set_device(0)
torch.set_num_threads(4)


def reference(x, gamma, inverse):
    return ((x.float() * inverse).to(torch.bfloat16).float() * gamma.float()).to(torch.bfloat16)


def compare(a, b):
    a, b = a.cpu().contiguous(), b.cpu().contiguous()
    bits = a.view(torch.uint16).to(torch.int32) != b.view(torch.uint16).to(torch.int32)
    error = a.float().double() - b.float().double()
    finite = bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all())
    result = {'raw_bits_equal': not bool(bits.any()), 'different_raw_words': int(bits.sum()), 'elements': a.numel(),
              'all_finite': finite, 'max_abs_error': float(error.abs().max()) if finite else None}
    if bool(bits.any()):
        idx = bits.flatten().nonzero().flatten()[:8]
        result['first_differences'] = [{'index': int(i), 'got_word': int(a.view(torch.uint16).flatten()[i]),
                                        'native_word': int(b.view(torch.uint16).flatten()[i])} for i in idx]
    return result


try:
    generator = torch.Generator(device='cpu').manual_seed(6701)
    with torch.inference_mode():
        for rows, width in ((17, 128), (9, 5120), (2043, 128), (2043, 5120)):
            x = (torch.randn(rows, width, generator=generator) * 2).to(torch.bfloat16)
            x[0, :8] = torch.tensor([0., -0., 1., -1., 2.**-126, -2.**-126, 2.**-133, -2.**-133], dtype=torch.bfloat16)
            gamma = (1 + torch.randn(width, generator=generator) * .1).to(torch.bfloat16)
            gamma[:8] = torch.tensor([1., 1., 0., 0., -1., -1., 1., 1.], dtype=torch.bfloat16)
            x, gamma = x.to('npu'), gamma.to('npu')
            xf = x.float()
            inverse = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
            expected = reference(x, gamma, inverse)
            got = norm_tail(x, gamma, inverse)
            torch.npu.synchronize()
            row = {'shape': [rows, width], 'proof': compare(got, expected), 'warm_s': []}
            r['cases'].append(row)
            writer.write(r)
            if not row['proof']['raw_bits_equal']:
                raise AssertionError('Same-input native finite Norm-tail fixture differs')
            for _ in range(3):
                torch.npu.synchronize()
                t = time.perf_counter()
                got = norm_tail(x, gamma, inverse)
                torch.npu.synchronize()
                row['warm_s'].append(time.perf_counter() - t)
                assert compare(got, expected)['raw_bits_equal']
            print(json.dumps(row), flush=True)
        words = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).reshape(512, 128)
        x = words.to('npu')
        gamma = torch.ones(128, dtype=torch.bfloat16, device='npu')
        inverse = torch.ones(512, 1, dtype=torch.float32, device='npu')
        got, expected = norm_tail(x, gamma, inverse), reference(x, gamma, inverse)
        torch.npu.synchronize()
        finite = torch.isfinite(words.float())
        g, e = got.cpu().contiguous(), expected.cpu().contiguous()
        r['all_bf16_words'] = {'all_inputs': compare(g, e), 'finite_inputs': compare(g[finite], e[finite]),
                               'scope': 'Unit inverse/gamma; NaN/Inf payload equivalence is reported, not silently accepted as a finite-model quality gate.'}
        assert r['all_bf16_words']['finite_inputs']['raw_bits_equal']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print('Evidence:', ROOT / 'validation01.json', flush=True)
