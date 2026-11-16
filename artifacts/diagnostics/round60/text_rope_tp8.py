#!/usr/bin/env python3
"""Controlled text-only TP8 vector-fusion ablations on one resident model.

All cases share experimental FP32 linears/causal attention and HF-math option.
Only norm and/or SwiGLU implementation changes. This is not an encoder stage,
independent vision pass, production baseline or video quality measurement.
"""
import importlib.util
import json
import os
from pathlib import Path
import time
import types
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT = Path('/tmp/h3_round60_rope01')
CAP = Path('/tmp/h3_round54_capture01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
OUT = ROOT / f'text_rope_rank{rank}.json'
if OUT.exists():
    raise RuntimeError('Refuse complete-stack vector evidence overwrite')
torch.npu.set_device(rank)
torch.set_num_threads(2)
os.environ['VLLM_OMNI_910A_NO_FUSED_NORM'] = '1'
os.environ['VLLM_OMNI_910A_NO_FUSED_ACT'] = '1'
os.environ['VLLM_OMNI_910A_TEXT_HF_MATH'] = '1'
os.environ['VLLM_OMNI_910A_TEXT_FP32_LINEAR'] = '1'
from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel
from vllm_omni.platforms.npu.models.minimax_h3 import _apply_rotary_pos_emb_npu
from h3_text_causal57 import causal_fp32
import sys
sys.path.insert(0, '/tmp/h3_round58_vec01')
from native_bf16_vector58 import native_launch
from padded_fp32_projection59 import cache_stats

spec = importlib.util.spec_from_file_location('vector_candidate58', ROOT / 'candidate_padded_text_encoder.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate._apply_rotary_pos_emb = None
candidate._scaled_dot_product_attention = causal_fp32
NORM, ACT = False, True
VERIFY_ACT, ACT_CHECKS = False, []
ROPE_MODE, ROPE_CALL = 'native', 0
VERIFY_ROPE, ROPE_CHECKS = False, []


def hf_apply(x, cos, sin):
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        rotated = torch.cat((-xf[..., 64:], xf[..., :64]), dim=-1)
        first = (xf * cos.float().unsqueeze(1)).to(x.dtype)
        second = (rotated * sin.float().unsqueeze(1)).to(x.dtype)
        return (first.float() + second.float()).to(x.dtype)


def apply_rope(q, k, cos, sin):
    global ROPE_CALL
    layer = ROPE_CALL
    ROPE_CALL += 1
    captured = None
    if ROPE_MODE == 'native' and rank == 0 and layer in (0, 24, 49) and not (ROOT / 'capture' / f'rope_layer{layer:02d}.pt').exists():
        captured = {name: value.detach().cpu().contiguous().clone() for name, value in zip(('q', 'k', 'cos', 'sin'), (q, k, cos, sin))}
    if ROPE_MODE == 'native':
        output = _apply_rotary_pos_emb_npu(q, k, cos, sin)
    elif ROPE_MODE == 'hf_eager':
        output = (hf_apply(q, cos, sin), hf_apply(k, cos, sin))
    else:
        from rope_vector60 import rope
        output = (rope(q, cos, sin, hf_round=ROPE_MODE == 'tbe_hf'), rope(k, cos, sin, hf_round=ROPE_MODE == 'tbe_hf'))
    if VERIFY_ROPE and ROPE_MODE in ('tbe_native', 'tbe_hf'):
        reference = _apply_rotary_pos_emb_npu(q, k, cos, sin) if ROPE_MODE == 'tbe_native' else (hf_apply(q, cos, sin), hf_apply(k, cos, sin))
        for side, got, ref in zip(('q', 'k'), output, reference):
            gcpu, rcpu = got.detach().cpu().contiguous(), ref.detach().cpu().contiguous()
            ROPE_CHECKS.append({'layer': layer, 'side': side, 'elements': got.numel(),
                                'raw_bits_equal_reference': torch.equal(gcpu.view(torch.uint8), rcpu.view(torch.uint8)),
                                'finite': bool(torch.isfinite(gcpu.float()).all())})
    if captured is not None:
        target = ROOT / 'capture' / f'rope_layer{layer:02d}.pt'
        if not target.exists():
            captured.update({name: value.detach().cpu().contiguous().clone() for name, value in zip(('q_output', 'k_output'), output)})
            metadata = {'layer': layer, 'shapes': {name: list(value.shape) for name, value in captured.items()},
                        'input_strides': {'q': list(q.stride()), 'k': list(k.stride())},
                        'inputs_unchanged': torch.equal(captured['q'].view(torch.uint8), q.detach().cpu().contiguous().view(torch.uint8)) and torch.equal(captured['k'].view(torch.uint8), k.detach().cpu().contiguous().view(torch.uint8)),
                        'scope': 'Actual stable padded experimental text pre-RoPE QKV and tables; not whole generation reference'}
            assert metadata['inputs_unchanged'] and sum(value.numel()*value.element_size() for value in captured.values()) < 512*1024*1024
            with target.open('xb') as f:
                torch.save(captured, f)
            target.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return output


candidate._apply_rotary_pos_emb = apply_rope


def mlp_forward(self, x):
    packed = self.gate_up_proj(x)
    if ACT:
        y = native_launch('swiglu', packed.contiguous())[0]
        if VERIFY_ACT:
            half = packed.shape[-1] // 2
            reference = F.silu(packed[..., :half]) * packed[..., half:]
            packed_cpu = packed.detach().cpu()
            got_cpu, ref_cpu = y.detach().cpu().contiguous(), reference.detach().cpu().contiguous()
            ACT_CHECKS.append({'layer': len(ACT_CHECKS), 'elements': y.numel(),
                               'input_finite': bool(torch.isfinite(packed_cpu.float()).all()),
                               'raw_bits_equal_same_input_eager': torch.equal(got_cpu.view(torch.uint8), ref_cpu.view(torch.uint8))})
    else:
        half = packed.shape[-1] // 2
        y = F.silu(packed[..., :half]) * packed[..., half:]
    return self.down_proj(y)


candidate.MiniMaxH3Qwen3VLTextMLP.forward = mlp_forward
r = {'rank': rank, 'cases': {}, 'scope': 'Stable padded experimental text-only TP8 with FP32 linears/causal and native SwiGLU fixed; only RoPE changes, not full encoder/video or global lossless pass'}


def metric(a, b):
    a, b = a.contiguous(), b.contiguous()
    e = a.float().double() - b.float().double()
    return {'raw_bits_equal': torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
            'finite': bool(torch.isfinite(a.float()).all()), 'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / b.float().double().square().mean().sqrt().clamp_min(1e-30))}


try:
    init_distributed_environment(world_size=size, rank=rank, local_rank=rank, backend='hccl')
    initialize_model_parallel(sequence_parallel_size=size, ulysses_degree=size, tensor_parallel_size=1, backend='hccl')
    group = GroupCoordinator([list(range(size))], rank, 'hccl')
    config = Qwen3VLConfig.from_pretrained(str(MODEL)).text_config
    model = candidate.MiniMaxH3Qwen3VLTextModel(group, config, 50, torch.bfloat16).to(dtype=torch.bfloat16).eval()
    owner = candidate.MiniMaxH3Qwen3VLEncoder.__new__(candidate.MiniMaxH3Qwen3VLEncoder)
    nn.Module.__init__(owner)
    owner.text_model, owner.vision, owner.quant_config = model, nn.Module(), None
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    def weights():
        for shard in sorted(set(index.values())):
            keys = [k for k, s in index.items() if s == shard and k.startswith('model.language_model.layers.') and int(k.split('.')[3]) < 50]
            if not keys:
                continue
            with safe_open(str(MODEL / shard), framework='pt', device='cpu') as f:
                for key in keys:
                    yield key, f.get_tensor(key)
        # The same real captured embedding bypasses this unused vocabulary.
        yield 'model.language_model.embed_tokens.weight', torch.zeros(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    t = time.perf_counter()
    loaded = owner.load_weights(weights())
    r['load_s'], r['loaded_parameter_tensors'] = time.perf_counter() - t, len(loaded)
    model.embed_tokens = nn.Identity()
    model.to(torch.device('npu', rank))
    count = 0
    for module in model.modules():
        if isinstance(module, candidate.MiniMaxH3Qwen3VLRMSNorm):
            original = module.forward
            def forward(self, x, residual=None, original=original):
                if not NORM:
                    return original(x, residual)
                y, updated = native_launch('addrms' if residual is not None else 'rms', x.contiguous(),
                                             self.weight.contiguous(), residual.contiguous() if residual is not None else None,
                                             eps=self.variance_epsilon, qwen=True)
                return (y, updated) if residual is not None else y
            module.forward = types.MethodType(forward, module)
            count += 1
    r['norm_modules_wrapped'] = count
    assert count == 200
    d = torch.load(CAP / 'text_inputs.pt', weights_only=True)
    x, positions, mask = d['inputs_embeds'].to('npu'), d['positions'].to('npu'), d['visual_pos_masks'].to('npu')
    deep = [z.to('npu') for z in d['deepstack_visual_embeds']]
    outputs = {}
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--capture-only', action='store_true'); args = parser.parse_args()
    if args.capture_only:
        OUT = ROOT / f'text_rope_capture_rank{rank}.json'
        if OUT.exists():
            raise RuntimeError('Refuse completed RoPE capture receipt overwrite')
    modes = ('native',) if args.capture_only else ('native', 'hf_eager', 'tbe_native', 'tbe_hf')
    for label in modes:
        ROPE_MODE, ROPE_CALL = label, 0
        os.environ['VLLM_OMNI_910A_TEXT_PAD_KV'] = '1'
        NORM, ACT = False, True
        VERIFY_ACT, ACT_CHECKS = False, []
        VERIFY_ROPE, ROPE_CHECKS = label in ('tbe_native', 'tbe_hf'), []
        ROPE_CALL = 0
        torch.distributed.barrier(); torch.npu.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            out = model(x, positions, visual_pos_masks=mask, deepstack_visual_embeds=deep)
        torch.npu.synchronize()
        first = time.perf_counter() - t
        cpu = out.detach().cpu().contiguous()
        VERIFY_ACT = False
        VERIFY_ROPE = False
        ROPE_CALL = 0
        torch.distributed.barrier(); torch.npu.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            repeat = model(x, positions, visual_pos_masks=mask, deepstack_visual_embeds=deep)
        torch.npu.synchronize()
        second = time.perf_counter() - t
        repeat_cpu = repeat.detach().cpu().contiguous()
        ROPE_CALL = 0
        torch.distributed.barrier(); torch.npu.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            third = model(x, positions, visual_pos_masks=mask, deepstack_visual_embeds=deep)
        torch.npu.synchronize(); third_s = time.perf_counter() - t
        third_cpu = third.detach().cpu().contiguous()
        row = {'first_forward_s': first, 'repeat_forward_s': second, 'third_forward_s': third_s, 'shape': list(out.shape),
               'finite': bool(torch.isfinite(cpu.float()).all()), 'repeat_vs_first': metric(repeat_cpu, cpu),
               'third_vs_repeat': metric(third_cpu, repeat_cpu), 'padding_cache': cache_stats(),
               'first_forward_includes_rope_capture': label == 'native' and args.capture_only,
               'first_forward_includes_rope_reference_checks': label in ('tbe_native', 'tbe_hf'), 'rope_reference_checks': ROPE_CHECKS,
               'act_reference_checks': ACT_CHECKS}
        if label in ('tbe_native', 'tbe_hf'):
            assert len(ROPE_CHECKS) == 100
        if rank == 0:
            outputs[label] = cpu.clone()
            if label != 'native':
                row['vs_native_rope_first'] = metric(cpu, outputs['native'])
            teacher = torch.load(Path('/tmp/h3_round56_audit01/cpu_last_hf.pt'), weights_only=True)['output']
            row['vs_cpu_fp32_oracle'] = metric(cpu, teacher)
        r['cases'][label] = row
        OUT.write_text(json.dumps(r, indent=2) + '\n')
        print(json.dumps({'rank': rank, 'case': label, **row}), flush=True)
    if rank == 0:
        target = ROOT / ('text_rope_capture_outputs.pt' if args.capture_only else 'text_rope_outputs.pt')
        if target.exists():
            raise RuntimeError('Refuse final vector tensor evidence overwrite')
        with target.open('xb') as f:
            torch.save(outputs, f)
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print('Evidence:', OUT, flush=True)
    if torch.distributed.is_initialized():
        destroy_model_parallel(); torch.distributed.destroy_process_group()
