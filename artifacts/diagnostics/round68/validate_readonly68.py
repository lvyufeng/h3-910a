#!/usr/bin/env python3
"""New proof ID: independent readonly snapshots and explicit extreme add pairs."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from norm_prefix68 import norm_prefix
from evidence_io68 import EvidenceWriter

ROOT = Path('/tmp/h3_round68_prefix01')
writer = EvidenceWriter(ROOT / 'validation02.json')
r = {'cases': [], 'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('norm_prefix68.py', 'validate_readonly68.py')}}
torch.npu.set_device(0)


def raw_equal(a, b):
    return torch.equal(a.detach().cpu().contiguous().view(torch.uint8), b.detach().cpu().contiguous().view(torch.uint8))


try:
    with torch.inference_mode():
        generator = torch.Generator(device='cpu').manual_seed(6802)
        fixtures = [('bounded', torch.randn(17, 128, generator=generator).to(torch.bfloat16),
                     torch.randn(17, 128, generator=generator).to(torch.bfloat16))]
        pairs = [(0x0000, 0x8000), (0x8000, 0x8000), (0x0001, 0x8001), (0x0080, 0x8080),
                 (0x7f7f, 0x7f7f), (0xff7f, 0xff7f), (0x7f7f, 0xff7f), (0x7f80, 0xff80),
                 (0x7fc1, 0x3f80), (0xffc1, 0xbf80)]
        x = torch.tensor([a for a, _ in pairs], dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)[:, None].repeat(1, 128)
        res = torch.tensor([b for _, b in pairs], dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)[:, None].repeat(1, 128)
        fixtures.append(('extreme_pairs', x, res))
        for name, xc, rc in fixtures:
            for active in (False, True):
                x, residual = xc.to('npu'), rc.to('npu')
                before_x, before_res = x.cpu().clone(), residual.cpu().clone()
                expected_update = (residual.float() + x.float()).to(torch.bfloat16) if active else x.clone()
                expected_square = expected_update.float().square()
                square, updated = norm_prefix(x, residual if active else None)
                torch.npu.synchronize()
                row = {'name': name, 'residual': active, 'x_readonly': raw_equal(x, before_x),
                       'residual_readonly': raw_equal(residual, before_res),
                       'square_raw_bits_equal': raw_equal(square, expected_square),
                       'updated_raw_bits_equal': raw_equal(updated, expected_update),
                       'outputs_aligned_dense': all(value.data_ptr() % 32 == 0 and torch_npu.get_npu_format(value) in (0, 2) for value in (square, updated)),
                       'output_formats': [torch_npu.get_npu_format(square), torch_npu.get_npu_format(updated)]}
                if name == 'extreme_pairs':
                    row['pair_words'] = pairs
                    row['updated_word_per_pair'] = [int(v) for v in updated.cpu()[:, 0].contiguous().view(torch.uint16)]
                    row['native_updated_word_per_pair'] = [int(v) for v in expected_update.cpu()[:, 0].contiguous().view(torch.uint16)]
                r['cases'].append(row)
                assert row['x_readonly'] and row['residual_readonly'] and row['outputs_aligned_dense']
                if name == 'bounded':
                    assert row['square_raw_bits_equal'] and row['updated_raw_bits_equal']
        r['exceptional_add_equivalence_claimed'] = False
        r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print(json.dumps(r, indent=2), flush=True)
