#!/usr/bin/env python3
"""Native BF16 cast comparison, byte oracle and identical-layout warm microtest."""
import hashlib
import json
from pathlib import Path
import time
import torch
import torch_npu
from activation_cast69 import expand_activation
from evidence_io69 import EvidenceWriter

ROOT = Path('/tmp/h3_round69_cast01')
writer = EvidenceWriter(ROOT / 'validation01.json')
r = {'cases': [], 'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('activation_cast69.py', 'validate_cast69.py')}}
torch.npu.set_device(0)
torch.set_num_threads(4)


def raw_equal(a, b):
    return torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


def layout(value):
    return [list(value.shape), list(value.stride()), str(value.dtype), torch_npu.get_npu_format(value)]


try:
    with torch.inference_mode():
        words = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        x = words.to('npu')
        out, native = expand_activation(x), x.float()
        torch.npu.synchronize()
        oracle = (torch.arange(65536, dtype=torch.int64) << 16).to(torch.int32).view(torch.float32)
        r['all_bf16_words'] = {'elements': 65536, 'matches_bit_expansion_oracle': raw_equal(out, oracle),
                                'matches_native_cast': raw_equal(out, native), 'native_layout': layout(native), 'candidate_layout': layout(out)}
        assert r['all_bf16_words']['matches_bit_expansion_oracle'] and r['all_bf16_words']['matches_native_cast']
        generator = torch.Generator(device='cpu').manual_seed(6901)
        for shape in ((1, 2043, 5120), (1, 2043, 1024), (1, 2043, 3200)):
            x = torch.randn(shape, generator=generator).to(torch.bfloat16).to('npu')
            before = x.cpu().clone()
            native, out = x.float(), expand_activation(x)
            torch.npu.synchronize()
            row = {'shape': list(shape), 'elements': x.numel(), 'raw_bits_equal': raw_equal(out, native),
                   'input_readonly': raw_equal(x, before), 'native_layout': layout(native), 'candidate_layout': layout(out), 'native_s': [], 'candidate_s': []}
            assert row['raw_bits_equal'] and row['input_readonly'] and row['native_layout'] == row['candidate_layout']
            for label, function in (('native', lambda: x.float()), ('candidate', lambda: expand_activation(x))):
                for _ in range(4):
                    torch.npu.synchronize()
                    t = time.perf_counter()
                    value = function()
                    torch.npu.synchronize()
                    row[label + '_s'].append(time.perf_counter() - t)
                    assert raw_equal(value, native)
            r['cases'].append(row)
            writer.write(r)
            print(json.dumps(row), flush=True)
        bad = torch.ones(129, dtype=torch.bfloat16, device='npu')[1:]
        try:
            expand_activation(bad)
        except ValueError:
            r['contiguous_unaligned_view_rejected'] = True
        else:
            raise AssertionError('Contiguous misaligned source was not rejected')
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print('Evidence:', ROOT / 'validation01.json', flush=True)
