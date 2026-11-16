#!/usr/bin/env python3
"""Actual NPU bounded-prefix checks and explicit overflow/NaN counterexamples."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from norm_prefix68 import norm_prefix
from evidence_io68 import EvidenceWriter

ROOT = Path('/tmp/h3_round68_prefix01')
writer = EvidenceWriter(ROOT / 'validation01.json')
r = {'cases': [], 'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('norm_prefix68.py', 'validate_prefix68.py')}}
torch.npu.set_device(0)
torch.set_num_threads(4)


def proof(a, b):
    a, b = a.cpu().contiguous(), b.cpu().contiguous()
    equal = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return {'raw_bits_equal': equal, 'dtype': str(a.dtype), 'elements': a.numel(),
            'finite': bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()),
            'different_values': int(a.ne(b).sum())}


try:
    generator = torch.Generator(device='cpu').manual_seed(6801)
    with torch.inference_mode():
        for rows, width in ((17, 128), (9, 5120), (2043, 128), (2043, 5120)):
            x = (torch.randn(rows, width, generator=generator) * 2).to(torch.bfloat16)
            residual = (torch.randn(rows, width, generator=generator) * 2).to(torch.bfloat16)
            x[0, :8] = torch.tensor([0., -0., 1., 1.0078125, -1., -1.0078125, 2.**-126, 2.**-133], dtype=torch.bfloat16)
            residual[0, :8] = torch.tensor([-0., -0., 2.**-8, 2.**-8, -2.**-8, -2.**-8, -2.**-126, 0.], dtype=torch.bfloat16)
            x, residual = x.to('npu'), residual.to('npu')
            for active in (False, True):
                expected_updated = (residual.float() + x.float()).to(torch.bfloat16) if active else x
                expected_square = expected_updated.float().square()
                square, updated = norm_prefix(x, residual if active else None)
                torch.npu.synchronize()
                row = {'shape': [rows, width], 'residual': active, 'square': proof(square, expected_square),
                       'updated': proof(updated, expected_updated), 'square_shape_stride': [list(square.shape), list(square.stride())],
                       'native_shape_stride': [list(expected_square.shape), list(expected_square.stride())]}
                r['cases'].append(row)
                writer.write(r)
                assert row['square']['raw_bits_equal'] and row['square']['finite'] and row['updated']['raw_bits_equal']
                assert row['square_shape_stride'] == row['native_shape_stride']
                for _ in range(2):
                    again, again_updated = norm_prefix(x, residual if active else None)
                    torch.npu.synchronize()
                    assert proof(again, expected_square)['raw_bits_equal'] and proof(again_updated, expected_updated)['raw_bits_equal']
            print('Passed shape', rows, width, flush=True)
        all_words = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).reshape(512, 128)
        x = all_words.to('npu')
        square, _ = norm_prefix(x)
        native = x.float().square()
        torch.npu.synchronize()
        bounded = torch.isfinite(all_words.float()) & all_words.float().abs().le(2.**60)
        a, b = square.cpu().contiguous(), native.cpu().contiguous()
        r['all_bf16_words_square'] = {'all_patterns': proof(a, b), 'bounded_finite_inputs_abs_le_2pow60': proof(a[bounded], b[bounded]),
                                     'native_overflow_values': int(torch.isinf(b).sum()), 'candidate_overflow_values': int(torch.isinf(a).sum()),
                                     'scope': 'All65536 BF16 input words; bounded finite subset is the required arithmetic gate, overflow/NaN behavior is retained separately, not accepted as generic equivalence.'}
        assert r['all_bf16_words_square']['bounded_finite_inputs_abs_le_2pow60']['raw_bits_equal']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print('Evidence:', ROOT / 'validation01.json', flush=True)
