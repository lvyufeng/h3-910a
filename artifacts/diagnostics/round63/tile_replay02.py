#!/usr/bin/env python3
"""Reproduce actual VAE tile before accepting same-input chunk/linear A/B.

Use worker's platform initialization (internal formats), explicit service env,
and real captured intermediate checkpoints. This is not a whole-video result.
"""
import collections
import hashlib
import inspect
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
OUT = ROOT / 'tile_replay02.json'
if OUT.exists():
    raise RuntimeError('Refuse VAE tile proof overwrite')
current_omni_platform.set_device(torch.device('npu', 0))
torch.set_num_threads(8)
r = {'cases': {}, 'scope': 'Actual same-input complete 36-layer tile, resident; excludes tile collectives/temporal blend/offload/video quality.',
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'platform_set_device_called': True,
     'context': {'matmul_allow_hf32': torch.npu.matmul.allow_hf32,
                 'conv_allow_hf32': torch.npu.conv.allow_hf32}}


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    assert x.shape == y.shape and x.dtype == y.dtype
    e = x.double() - y.double()
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'elements': x.numel(), 'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / y.double().square().mean().sqrt().clamp_min(1e-30))}


def once(decoder, x):
    with torch.autocast('npu', dtype=torch.float16):
        return decoder(x)


def measured(decoder, x, baseline=None, actual=None):
    torch.npu.synchronize()
    t = time.perf_counter()
    first = once(decoder, x)
    torch.npu.synchronize()
    first_s = time.perf_counter() - t
    outputs, groups = [], []
    with torch.autocast('npu', dtype=torch.float16):
        for _ in range(3):
            torch.npu.synchronize()
            t = time.perf_counter()
            out = decoder(x)
            torch.npu.synchronize()
            groups.append(time.perf_counter() - t)
            outputs.append(out.detach().cpu().contiguous())
    row = {'first_s': first_s, 'warm_s': groups, 'median_warm_s': statistics.median(groups),
           'first_vs_repeat': metric(first, outputs[0]),
           'all_repeats_equal': all(metric(z, outputs[0])['raw_bits_equal'] for z in outputs)}
    if baseline is not None:
        row['vs_baseline'] = metric(outputs[-1], baseline)
    if actual is not None:
        row['vs_actual'] = metric(outputs[-1], actual)
    return outputs[-1], row


try:
    torch.set_default_dtype(torch.float32)
    t = time.perf_counter()
    vae = MiniMaxH3VideoVAE(str(MODEL), device=torch.device('npu', 0), load_device=torch.device('cpu'),
                          decode_only=True, trust_remote_code=True)
    vae.decoder_component.load_to_device()
    torch.npu.synchronize()
    decoder = vae.model.decoder.eval()
    r['load_and_device_s'] = time.perf_counter() - t
    r['parameter_dtypes'] = dict(collections.Counter(str(p.dtype) for p in decoder.parameters()))
    r['parameter_inference_counts'] = dict(collections.Counter(str(p.is_inference()) for p in decoder.parameters()))
    r['decoder_source_sha256'] = hashlib.sha256(Path(inspect.getfile(type(decoder))).read_bytes()).hexdigest()
    x = torch.load(CAP / 'decoder_tile_input.pt', weights_only=True).to('npu').contiguous()
    actual = torch.load(CAP / 'decoder_tile_output.pt', weights_only=True)
    r['input_shape'], r['input_dtype'] = list(x.shape), str(x.dtype)
    handles, snapshots = [], {}
    for layer in (0, 17, 35):
        def pre(module, args, layer=layer):
            snapshots['norm1_input_%02d' % layer] = metric(args[0], torch.load(CAP / ('transformer_blocks_%d_norm1_input.pt' % layer), weights_only=True))
        handles.append(decoder.transformer_blocks[layer].norm1.register_forward_pre_hook(pre))
    attn = decoder.transformer_blocks[0].attn
    original_attn = attn._perform_attention
    def inspect_attention(self, q, k, v, pack_info):
        data = torch.load(CAP / 'attention0_inputs.pt', weights_only=True)
        snapshots['qkv'] = {key: metric(value, data[key]) for key, value in zip(('q', 'k', 'v'), (q, k, v))}
        out = original_attn(q, k, v, pack_info)
        snapshots['attention_out'] = metric(out, torch.load(CAP / 'attention0_output.pt', weights_only=True))
        return out
    attn._perform_attention = MethodType(inspect_attention, attn)
    with torch.inference_mode():
        diagnostic = once(decoder, x)
        torch.npu.synchronize()
        r['baseline_capture_checkpoints'] = snapshots
        r['baseline_capture_output'] = metric(diagnostic, actual)
        # Hooks and downloads are removed before any measured call.
        for handle in handles:
            handle.remove()
        attn._perform_attention = original_attn
        baseline, row = measured(decoder, x, actual=actual)
        r['cases']['baseline_c128'] = row
        OUT.write_text(json.dumps(r, indent=2) + '\n')
        print(json.dumps({'case': 'baseline_c128', **row}), flush=True)
        for chunk in (512, 1797):
            os.environ['VLLM_OMNI_910A_ATTN_CHUNK'] = str(chunk)
            out, row = measured(decoder, x, baseline=baseline, actual=actual)
            r['cases']['candidate_c%d' % chunk] = row
            OUT.write_text(json.dumps(r, indent=2) + '\n')
            print(json.dumps({'case': 'candidate_c%d' % chunk, **row}), flush=True)
        r['capture_baseline_exact'] = r['baseline_capture_output']['raw_bits_equal']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print('Evidence:', OUT, flush=True)
