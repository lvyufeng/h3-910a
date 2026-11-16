#!/usr/bin/env python3
"""First genuine910A gather/FP32-product/finalBF16 smoke; not real model timing."""
import hashlib
import json
from pathlib import Path
import time
import torch
import torch_npu
from evidence_io72 import EvidenceWriter
from vision_rope72 import rope_pair

ROOT = Path('/tmp/h3_round72_rope01')
writer = EvidenceWriter(ROOT / 'micro01.json')
r = {'scope': __doc__, 'cases': {}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
torch.npu.set_device(0)
torch.set_num_threads(4)


def native(q, k, cos, sin):
    q, k = q.float(), k.float()
    c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    def rotate(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), -1)
    return ((q * c) + (rotate(q) * s)).bfloat16(), ((k * c) + (rotate(k) * s)).bfloat16()


def identity(a, b):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'elements': a.numel(),
            'raw_bits_equal': torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
            'different_words': int((a.view(torch.uint16).int() != b.view(torch.uint16).int()).sum()),
            'finite': bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()),
            'max_abs_error': float((a.float()-b.float()).abs().max())}


try:
    for n in (1, 5, 8):
        cpu = torch.linspace(-2, 2, n * 3456, dtype=torch.float32).view(n, 3, 16, 72).bfloat16()
        cpu.view(torch.uint16)[0, 0, 0, 0] = 0
        cpu.view(torch.uint16)[0, 0, 0, 1] = 32768
        cpu.view(torch.uint16)[0, 1, 0, 0] = 32768
        packed = cpu.to('npu:0')
        q, k, v = packed.unbind(1)
        angle = torch.arange(n * 72, dtype=torch.float32).reshape(n, 72) * .03125
        cos, sin = angle.cos().to('npu:0'), angle.sin().to('npu:0')
        cos[0, 0], sin[0, 0] = 1., 0.
        before = packed.detach().cpu().contiguous().clone()
        with torch.inference_mode():
            expected = native(q, k, cos, sin)
            torch.npu.synchronize(); start = time.perf_counter()
            actual = rope_pair(q, k, cos, sin)
            torch.npu.synchronize(); seconds = time.perf_counter() - start
        row = {'first_s_excluded_compile': seconds, 'query': identity(actual[0], expected[0]), 'key': identity(actual[1], expected[1]),
               'query_input_stride': list(q.stride()), 'key_storage_offset': k.storage_offset(),
               'output_stride': list(actual[0].stride()), 'native_output_stride': list(expected[0].stride()),
               'output_format': torch_npu.get_npu_format(actual[0]), 'native_output_format': torch_npu.get_npu_format(expected[0]),
               'packed_readonly': torch.equal(packed.detach().cpu().contiguous().view(torch.uint8), before.view(torch.uint8))}
        r['cases'][str(n)] = row
        writer.write(r)
        print(json.dumps(row), flush=True)
        assert row['query']['raw_bits_equal'] and row['key']['raw_bits_equal'] and row['packed_readonly']
        assert row['output_stride'] == row['native_output_stride'] and row['output_format'] == row['native_output_format']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
