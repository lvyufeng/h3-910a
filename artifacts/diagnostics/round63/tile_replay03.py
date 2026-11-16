#!/usr/bin/env python3
"""Resolve VAE ambient construction dtype before complete replay acceptance.

Service loads VAE from_pretrained under the pipeline default half dtype, then
converts storage to FP32. A standalone default-FP32 construction is not the same
state even if all final parameter dtype metadata matches. Do not edit checkpoint.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import MethodType
os.environ.update(VLLM_OMNI_910A_NO_FUSED_NORM='1', VLLM_OMNI_910A_NO_FUSED_ACT='1',
                  VLLM_OMNI_910A_ATTN_CHUNK='128')
for key in ('VLLM_OMNI_910A_VAE_AUDIT_DIR', 'VLLM_OMNI_910A_TEXT_AUDIT_DIR', 'VLLM_OMNI_910A_AUDIT_DIR'):
    os.environ.pop(key, None)
import torch
import torch_npu
from vllm_omni.platforms import current_omni_platform
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

ROOT = Path('/tmp/h3_round63_decode01')
CAP = ROOT / 'capture' / 'rank0'
MODEL = Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/video_vae')
OUT = ROOT / 'tile_replay03.json'
if OUT.exists():
    raise RuntimeError('Refuse tile proof overwrite')
current_omni_platform.set_device(torch.device('npu', 0))
torch.set_num_threads(8)
r = {'cases': {}, 'scope': 'Complete real tile; pin actual loader dtype before accepting local optimization, not whole-video quality.',
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    assert x.shape == y.shape and x.dtype == y.dtype
    e = x.double() - y.double()
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'elements': x.numel(), 'different_values': int(x.ne(y).sum()),
            'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / y.double().square().mean().sqrt().clamp_min(1e-30))}


def run(decoder, x):
    with torch.autocast('npu', dtype=torch.float16):
        return decoder(x)


def inspect_once(decoder, x, actual):
    snapshots, handles = {}, []
    for layer in (0, 17, 35):
        def pre(module, args, layer=layer):
            native = torch.load(CAP / ('transformer_blocks_%d_norm1_input.pt' % layer), weights_only=True)
            snapshots['norm1_input_%02d' % layer] = metric(args[0], native)
            if layer == 0:
                snapshots['first_patch_tokens'] = metric(args[0][:, :-5], native[:, :-5])
                snapshots['first_suffix_tokens'] = metric(args[0][:, -5:], native[:, -5:])
        handles.append(decoder.transformer_blocks[layer].norm1.register_forward_pre_hook(pre))
    attn = decoder.transformer_blocks[0].attn
    original = attn._perform_attention
    def inspect_attention(self, q, k, v, pack_info):
        captured = torch.load(CAP / 'attention0_inputs.pt', weights_only=True)
        snapshots['qkv'] = {key: metric(value, captured[key]) for key, value in zip(('q', 'k', 'v'), (q, k, v))}
        return original(q, k, v, pack_info)
    attn._perform_attention = MethodType(inspect_attention, attn)
    try:
        out = run(decoder, x)
        torch.npu.synchronize()
        return {'checkpoints': snapshots, 'vs_actual': metric(out, actual)}
    finally:
        for handle in handles:
            handle.remove()
        attn._perform_attention = original


def measured(decoder, x, actual):
    run(decoder, x)
    torch.npu.synchronize()
    groups, outputs = [], []
    with torch.autocast('npu', dtype=torch.float16):
        for _ in range(3):
            t = time.perf_counter()
            out = decoder(x)
            torch.npu.synchronize()
            groups.append(time.perf_counter() - t)
            outputs.append(out.detach().cpu().contiguous())
    return {'warm_s': groups, 'median_s': statistics.median(groups),
            'all_repeats_equal': all(metric(value, outputs[0])['raw_bits_equal'] for value in outputs),
            'vs_actual': metric(outputs[-1], actual)}


try:
    # Deliberately reconstruct the worker's half-initialized module. The adapter
    # then widens its loaded parameters to FP32, exactly like production.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    t = time.perf_counter()
    try:
        vae = MiniMaxH3VideoVAE(str(MODEL), device=torch.device('npu', 0), load_device=torch.device('cpu'),
                              decode_only=True, trust_remote_code=True)
    finally:
        torch.set_default_dtype(previous_dtype)
    vae.decoder_component.load_to_device()
    torch.npu.synchronize()
    decoder = vae.model.decoder.eval()
    r['load_and_device_s'] = time.perf_counter() - t
    r['final_parameter_dtypes'] = dict(collections.Counter(str(p.dtype) for p in decoder.parameters()))
    x = torch.load(CAP / 'decoder_tile_input.pt', weights_only=True).to('npu').contiguous()
    actual = torch.load(CAP / 'decoder_tile_output.pt', weights_only=True)
    with torch.inference_mode():
        r['half_constructed_baseline'] = inspect_once(decoder, x, actual)
        r['capture_baseline_exact'] = r['half_constructed_baseline']['vs_actual']['raw_bits_equal']
        OUT.write_text(json.dumps(r, indent=2) + '\n')
        print(json.dumps({'baseline_capture_exact': r['capture_baseline_exact'],
                          'checkpoints': r['half_constructed_baseline']}), flush=True)
        if not r['capture_baseline_exact']:
            raise AssertionError('Half-constructed baseline still differs from actual tile; reject integration claim')
        for chunk in (128, 256, 512):
            os.environ['VLLM_OMNI_910A_ATTN_CHUNK'] = str(chunk)
            row = measured(decoder, x, actual)
            r['cases']['c%d' % chunk] = row
            OUT.write_text(json.dumps(r, indent=2) + '\n')
            print(json.dumps({'chunk': chunk, **row}), flush=True)
            assert row['all_repeats_equal'] and row['vs_actual']['raw_bits_equal']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print('Evidence:', OUT, flush=True)
