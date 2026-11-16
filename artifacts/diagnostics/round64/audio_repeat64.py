#!/usr/bin/env python3
"""Isolate native audio repeat variance on actual weights and workload geometry.

Input is synthetic, NOT the lost HTTP request's actual denoised audio latent.
No video finite-select is enabled and no production audio code is modified.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
os.environ.pop('VLLM_OMNI_910A_VAE_FINITE_SELECT', None)
import torch
import torch_npu
from vllm_omni.platforms import current_omni_platform
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE
from h3_evidence64 import EvidenceWriter

ROOT = Path('/tmp/h3_round64_service01')
writer = EvidenceWriter(ROOT / 'audio_repeat01.json')
r = {'cases': {}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'scope': 'Native audio VAE, real weights, synthetic fixed FP32 [2,32,207] latent matching service geometry; not actual HTTP latent or candidate video arithmetic.',
     'production_audio_modified': False}


def sha(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()


def metric(x, y):
    x, y = x.cpu().contiguous(), y.cpu().contiguous()
    e = x.double() - y.double()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'finite': finite, 'different_values': int(x.ne(y).sum()), 'elements': x.numel(),
            'max_abs_error': float(e.abs().max()) if finite else 'nonfinite',
            'relative_rms': float(e.square().mean().sqrt() / y.double().square().mean().sqrt().clamp_min(1e-30)) if finite else 'nonfinite'}


def masters_sha(vae):
    h = hashlib.sha256()
    for group in vae._stager._groups:
        h.update(memoryview(group.master.detach().cpu().contiguous().numpy()))
    return h.hexdigest()


try:
    current_omni_platform.set_device(torch.device('npu', 0))
    torch.set_num_threads(8)
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    r['initial_deterministic_algorithms'] = original_deterministic
    r['matmul_allow_hf32'], r['conv_allow_hf32'] = torch.npu.matmul.allow_hf32, torch.npu.conv.allow_hf32
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        vae = MiniMaxH3AudioVAE('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/audio_vae',
                              device=torch.device('npu', 0), load_device=torch.device('cpu'), decode_only=True, trust_remote_code=True)
    finally:
        torch.set_default_dtype(previous)
    r['parameter_dtypes'] = dict(collections.Counter(str(p.dtype) for p in vae.parameters()))
    r['parameter_bytes'] = sum(p.numel() * p.element_size() for p in vae.parameters())
    r['cpu_master_sha_before'] = masters_sha(vae)
    generator = torch.Generator(device='cpu').manual_seed(6401)
    x = (torch.randn(2, 32, 207, generator=generator) * .1).to('npu').contiguous()
    r['input_sha_before'] = sha(x)
    baseline = None
    with torch.inference_mode():
        for name, deterministic, staged in (('resident_native', False, False), ('staged_native', False, True),
                                             ('resident_deterministic', True, False), ('staged_deterministic', True, True)):
            torch.use_deterministic_algorithms(deterministic)
            row = {'deterministic_algorithms': deterministic, 'staged_each_call': staged, 'calls': []}
            outputs = []
            r['cases'][name] = row
            before = {'cpu': sha(torch.random.get_rng_state()), 'npu': sha(torch.npu.get_rng_state())}
            if not staged:
                vae.load_to_device()
                torch.npu.synchronize()
            try:
                for index in range(5):
                    if staged:
                        vae.load_to_device()
                    torch.npu.synchronize()
                    t = time.perf_counter()
                    value = vae.decode_latent(x)
                    torch.npu.synchronize()
                    elapsed = time.perf_counter() - t
                    value = value.cpu().contiguous()
                    if staged:
                        vae.offload_to_cpu()
                    if baseline is None:
                        baseline = value
                    outputs.append(value)
                    row['calls'].append({'seconds': elapsed, 'shape': list(value.shape), 'dtype': str(value.dtype),
                                         'sha256': sha(value), 'vs_first_in_case': metric(value, outputs[0]),
                                         'vs_native_first': metric(value, baseline)})
                    writer.write(r)
                row['median_after_first_s'] = statistics.median(call['seconds'] for call in row['calls'][1:])
                row['distinct_hashes'] = len({call['sha256'] for call in row['calls']})
                row['rng_state_unchanged'] = before == {'cpu': sha(torch.random.get_rng_state()), 'npu': sha(torch.npu.get_rng_state())}
                row['completed'] = True
            finally:
                vae.offload_to_cpu()
            print(json.dumps({'case': name, 'distinct_hashes': row['distinct_hashes'], 'rng_unchanged': row['rng_state_unchanged'],
                              'median_after_first_s': row['median_after_first_s']}), flush=True)
    r['input_sha_after'] = sha(x)
    r['cpu_master_sha_after'] = masters_sha(vae)
    assert r['input_sha_before'] == r['input_sha_after']
    assert r['cpu_master_sha_before'] == r['cpu_master_sha_after']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    if 'original_deterministic' in locals():
        torch.use_deterministic_algorithms(original_deterministic)
    writer.write(r)
    print('Evidence:', ROOT / 'audio_repeat01.json', flush=True)
