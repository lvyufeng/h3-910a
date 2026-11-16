#!/usr/bin/env python3
"""Verify fused layout/scaling bits and benchmark full actual VAE FA wrapper.

No production deployment; FA's numerical difference from legacy is independent
of pack/unpack identity and must remain visible in the receipt.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
os.environ.update(VLLM_OMNI_910A_NO_FUSED_NORM='1', VLLM_OMNI_910A_NO_FUSED_ACT='1')
import torch
import torch_npu
from vllm_omni.platforms import current_omni_platform
from packed_fa63 import PackedFA
from vae_attention_candidates63 import custom_fa_exploratory, sanitize
from evidence_io63 import EvidenceWriter

ROOT = Path('/tmp/h3_round63_decode01')
CAP = ROOT / 'capture' / 'rank0'
OUT = ROOT / 'pack_validation01.json'
writer = EvidenceWriter(OUT)
r = {'cases': {}, 'scope': 'Actual VAE QKV pack/unpack and full dense FA wrapper. Layout equivalence to existing FA, NOT FA equivalence to legacy attention/video lossless.',
     'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in ('pack_validation.py', 'pack_vector63.py', 'packed_fa63.py', 'evidence_io63.py')}}


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
        data = torch.load(CAP / 'attention0_inputs.pt', weights_only=True)
        q, k, dense_v = [data[key].to('npu').contiguous() for key in ('q', 'k', 'v')]
        backing = torch.zeros(1, 1797, 32, 192, dtype=torch.float16, device='npu')
        v = backing[..., 128:]
        v.copy_(dense_v)
        actual = torch.load(CAP / 'attention0_output.pt', weights_only=True)
        r['input_strides'] = {key: list(value.stride()) for key, value in zip(('q', 'k', 'v'), (q, k, v))}
        n, padded, heads, d = 1797, 2048, 32, 64
        ref = []
        for i, value in enumerate((q, k, v)):
            x = torch.nn.functional.pad(value.transpose(1, 2).contiguous(), (0, 0, 0, padded - n))
            if i == 0:
                x = x * .125
            ref.append(x.reshape(1, heads, padded // 16, 16, 4, 16).permute(0, 1, 4, 2, 3, 5).contiguous())
        reference_packed = torch.stack(ref)
        t = time.perf_counter()
        original_plan = PackedFA(br=128, bc=256)
        r['initial_build_or_cache_s'] = time.perf_counter() - t
        r['original_plan_objects'] = original_plan.objects
        packed = original_plan.pack(q, k, v)
        torch.npu.synchronize()
        r['pack_vs_torch'] = metric(packed, reference_packed)
        writer.write(r)
        print(json.dumps({'pack_vs_torch': r['pack_vs_torch']}), flush=True)
        assert r['pack_vs_torch']['raw_bits_equal'], 'Do not time/accept an incorrect layout transform'
        # Round-trip arbitrary actual K values through the inverse map, with no
        # attention or projection to conceal a permutation/tail-store error.
        roundtrip = original_plan.unpack(reference_packed[1])
        torch.npu.synchronize()
        r['unpack_vs_k'] = metric(roundtrip, k)
        assert r['unpack_vs_k']['raw_bits_equal']
        try:
            original_plan.pack(q, k, dense_v)
        except ValueError:
            r['wrong_v_stride_rejected_before_launch'] = True
        else:
            raise AssertionError('A stride-unsafe V descriptor was accepted')
        # Keep all contexts/precision boundaries fixed across wrapper variants.
        with torch.autocast('npu', dtype=torch.float16):
            legacy_fa, timing = timed(lambda: custom_fa_exploratory(q, k, v))
            r['cases']['existing_wrapper_br128_bc256'] = {'timing': timing, 'vs_native_capture': metric(legacy_fa, actual)}
            out, timing = timed(lambda: sanitize(original_plan(q, k, v)))
            proof = metric(out, legacy_fa)
            r['cases']['packed_wrapper_br128_bc256'] = {'timing': timing, 'vs_existing_same_tile_fa': proof,
                                                        'vs_native_capture': metric(out, actual)}
            assert proof['raw_bits_equal'], 'Packing must not change identical-kernel FA output'
            faster_plan = PackedFA(br=64, bc=1024)
            r['wider_plan_objects'] = faster_plan.objects
            out, timing = timed(lambda: sanitize(faster_plan(q, k, v)))
            # Same-input/same-binary oracle for wider FA, still outside timing.
            onz = torch.empty(faster_plan.nz_shape, dtype=torch.float16, device='npu')
            lse = torch.empty((1, heads, padded), dtype=torch.float32, device='npu')
            faster_plan._launch('attention', (reference_packed[0], reference_packed[1], reference_packed[2], onz, lse))
            torch.npu.synchronize()
            oracle = onz.permute(0, 1, 3, 4, 2, 5).reshape(1, heads, padded, d)[..., :n, :].transpose(1, 2).contiguous()
            same_binary = metric(out, oracle)
            r['cases']['packed_wrapper_br64_bc1024'] = {'timing': timing, 'vs_torch_pack_same_binary': same_binary,
                                                       'vs_existing_default_tile_fa': metric(out, legacy_fa),
                                                       'vs_native_capture': metric(out, actual)}
            assert same_binary['raw_bits_equal']
        r['completed'] = True
        r['production_installed'] = False
        r['legacy_attention_lossless_accepted'] = False
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print(json.dumps(r, indent=2), flush=True)
    print('Evidence:', OUT, flush=True)
