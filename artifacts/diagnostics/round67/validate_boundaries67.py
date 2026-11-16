#!/usr/bin/env python3
"""Guarded source02: explicit RNE boundaries and unaligned-view rejection."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from norm_tail_guard67 import norm_tail
from evidence_io67 import EvidenceWriter

ROOT = Path('/tmp/h3_round67_norm01')
writer = EvidenceWriter(ROOT / 'validation02.json')
r = {'cases': [], 'guards': [], 'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
     for name in ('norm_tail_guard67.py', 'validate_boundaries67.py')}}
torch.npu.set_device(0)


def reference(x, gamma, inverse):
    return ((x.float() * inverse).to(torch.bfloat16).float() * gamma.float()).to(torch.bfloat16)


def verify(name, x, gamma, inv):
    x, gamma, inv = x.to('npu'), gamma.to('npu'), inv.to('npu')
    output, native = norm_tail(x, gamma, inv), reference(x, gamma, inv)
    torch.npu.synchronize()
    a, b = output.cpu().contiguous(), native.cpu().contiguous()
    bits = a.view(torch.uint16).to(torch.int32) != b.view(torch.uint16).to(torch.int32)
    row = {'name': name, 'shape': list(x.shape), 'different_raw_words': int(bits.sum()), 'elements': a.numel(),
           'raw_bits_equal': not bool(bits.any())}
    r['cases'].append(row)
    writer.write(r)
    assert row['raw_bits_equal'], name


try:
    with torch.inference_mode():
        midpoint = torch.tensor(1.00390625, dtype=torch.float32)
        below = torch.nextafter(midpoint, torch.tensor(0., dtype=torch.float32))
        above = torch.nextafter(midpoint, torch.tensor(2., dtype=torch.float32))
        x = torch.ones(3, 128, dtype=torch.bfloat16)
        x[:, 1::2] = -1.
        inv = torch.stack((below, midpoint, above)).view(3, 1)
        r['normalized_boundary_inverse_values'] = inv.flatten().tolist()
        verify('normalized_round_below_tie_above_both_signs', x, torch.ones(128, dtype=torch.bfloat16), inv)
        values = (1 + torch.arange(128, dtype=torch.float32) / 128).to(torch.bfloat16)
        x = values[:, None].repeat(1, 128)
        gamma = values.clone()
        product = x.float() * gamma.float()
        low = product.view(torch.int32).bitwise_and(65535)
        r['final_product_boundaries'] = {'exact_midpoints': int(low.eq(32768).sum()),
                                         'below_midpoint_nonexact': int((low.gt(0) & low.lt(32768)).sum()),
                                         'above_midpoint_nonexact': int(low.gt(32768).sum())}
        assert all(value > 0 for value in r['final_product_boundaries'].values())
        verify('gamma_product_all128x128_mantissa_pairs', x, gamma, torch.ones(128, 1, dtype=torch.float32))
        x = -x
        verify('gamma_product_negative128x128_mantissa_pairs', x, gamma, torch.ones(128, 1, dtype=torch.float32))
        good = torch.ones(3, 128, dtype=torch.bfloat16, device='npu')
        gamma = torch.ones(128, dtype=torch.bfloat16, device='npu')
        inv = torch.ones(3, 1, dtype=torch.float32, device='npu')
        bad_x = torch.ones(3 * 128 + 1, dtype=torch.bfloat16, device='npu')[1:].view(3, 128)
        bad_gamma = torch.ones(129, dtype=torch.bfloat16, device='npu')[1:]
        assert bad_x.is_contiguous() and bad_gamma.is_contiguous() and bad_x.data_ptr() % 32 and bad_gamma.data_ptr() % 32
        for name, args in (('contiguous_unaligned_x_rejected', (bad_x, gamma, inv)),
                           ('contiguous_unaligned_gamma_rejected', (good, bad_gamma, inv)),
                           ('wrong_broadcast_shape_same_numel_rejected', (good, gamma, inv.reshape(1, 3)))):
            try:
                norm_tail(*args)
            except ValueError as exc:
                r['guards'].append({'name': name, 'rejected': True, 'message': str(exc)})
            else:
                raise AssertionError('Unsafe shape/layout not rejected: ' + name)
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print(json.dumps(r, indent=2), flush=True)
