#!/usr/bin/env python3
"""Actual fullpacked model vision RoPE replay, not full encoder performance."""
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from evidence_io72 import EvidenceWriter
from vision_rope_tail72 import rope_pair, tail

ROOT = Path('/tmp/h3_round72_tail01')
CAP = Path('/tmp/h3_round72_capture01')
writer = EvidenceWriter(ROOT / 'validation01.json')
r = {'scope': __doc__, 'layers': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'tail_source_sha256': hashlib.sha256((ROOT / 'vision_rope_tail72.py').read_bytes()).hexdigest()}
torch.npu.set_device(0)
torch.set_num_threads(4)


def raw(value):
    cpu = value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()


def metric(a, b):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    assert a.shape == b.shape and a.dtype == b.dtype
    error = a.float() - b.float()
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'elements': a.numel(),
            'raw_bits_equal': torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
            'finite': bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()),
            'different_words': int((a.view(torch.uint16).int() != b.view(torch.uint16).int()).sum()),
            'max_abs_error': float(error.abs().max())}


def native(q, k, cos, sin):
    q, k = q.float(), k.float()
    def rotate(value):
        return torch.cat((-value[..., 36:], value[..., :36]), dim=-1)
    c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return ((q*c)+(rotate(q)*s)).bfloat16(), ((k*c)+(rotate(k)*s)).bfloat16()


def timed(function, *args):
    torch.npu.synchronize(); start = time.perf_counter()
    result = function(*args)
    torch.npu.synchronize(); seconds = time.perf_counter() - start
    return result, seconds


try:
    for layer in (0, 8, 16, 26):
        path = CAP / ('rope_layer%02d.pt' % layer)
        meta = json.loads(path.with_suffix('.json').read_text())
        assert path.stat().st_size == meta['file_bytes'] < 128 * 1024 * 1024
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['file_sha256']
        d = torch.load(path, map_location='cpu', weights_only=True)
        assert d['packed_qkv'].shape == (8064, 3, 16, 72) and d['packed_qkv'].dtype == torch.bfloat16
        packed = d['packed_qkv'].to('npu:0')
        q, k, v = packed.unbind(1)
        cos, sin = d['cos'].to('npu:0'), d['sin'].to('npu:0')
        assert q.stride() == k.stride() == (3456,72,1) and q.storage_offset() == 0 and k.storage_offset() == 1152
        assert cos.dtype == sin.dtype == torch.float32 and cos.shape == sin.shape == (8064,72)
        assert all(torch.isfinite(value.float()).all() for value in d.values())
        assert max(meta['max_abs_values'].values()) < 2**120
        assert meta['max_abs_values']['cos'] <= 1 and meta['max_abs_values']['sin'] <= 1
        before = [raw(value) for value in (packed, cos, sin)]
        with torch.inference_mode():
            reference, reference_first = timed(native, q, k, cos, sin)
            actual, excluded = timed(rope_pair, q, k, cos, sin)
            row = {'layer': layer, 'native_query_vs_capture': metric(reference[0],d['query_out']),
                   'native_key_vs_capture': metric(reference[1],d['key_out']),
                   'candidate_query': metric(actual[0],reference[0]), 'candidate_key': metric(actual[1],reference[1]),
                   'first_native_s_excluded': reference_first, 'first_candidate_s_excluded_compile': excluded,
                   'output_strides': [list(value.stride()) for value in actual],
                   'native_output_strides': [list(value.stride()) for value in reference],
                   'output_formats': [torch_npu.get_npu_format(value) for value in actual],
                   'native_output_formats': [torch_npu.get_npu_format(value) for value in reference],
                   'input_before': before, 'warm': []}
            for proof in ('native_query_vs_capture','native_key_vs_capture','candidate_query','candidate_key'):
                assert row[proof]['raw_bits_equal'] and row[proof]['finite']
            assert row['output_strides'] == row['native_output_strides'] and row['output_formats'] == row['native_output_formats']
            for _ in range(3):
                original, original_s = timed(native, q, k, cos, sin)
                candidate, candidate_s = timed(rope_pair, q, k, cos, sin)
                proofs = [metric(candidate[i], reference[i]) for i in (0,1)]
                assert all(proof['raw_bits_equal'] and proof['finite'] for proof in proofs)
                row['warm'].append({'native_s': original_s, 'tail_s': candidate_s, 'candidate_proofs': proofs})
            # Interpret the full packed bytes using the WRONG dense pitch. This
            # counterexample must disagree, proving actual neighbour/V coverage.
            wrong_q = packed.reshape(-1)[:8064*1152].reshape(8064,16,72)
            wrong = native(wrong_q, k, cos, sin)[0]
            row['wrong_dense_pitch_query'] = metric(wrong, reference[0])
            assert not row['wrong_dense_pitch_query']['raw_bits_equal']
            row['unaligned_coefficient_rejected'] = False
            bad = torch.empty(cos.numel()+1, dtype=torch.float32, device='npu:0')[1:].reshape(cos.shape)
            bad.copy_(cos)
            q32=q.float(); rotated=torch.cat((-q32[...,36:],q32[...,:36]),-1)
            try:
                tail(q32, rotated, bad, sin)
            except ValueError:
                row['unaligned_coefficient_rejected'] = True
            assert row['unaligned_coefficient_rejected']
        row['input_after'] = [raw(value) for value in (packed, cos, sin)]
        row['packed_and_coefficients_readonly'] = row['input_before'] == row['input_after']
        assert row['packed_and_coefficients_readonly']
        row['native_median_s'] = statistics.median(item['native_s'] for item in row['warm'])
        row['tail_median_s'] = statistics.median(item['tail_s'] for item in row['warm'])
        r['layers'].append(row); writer.write(r)
        print(json.dumps({'layer':layer,'native_s':row['native_median_s'],'tail_s':row['tail_median_s'],'full_bits_equal':True,'wrong_pitch_fails':True}),flush=True)
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
