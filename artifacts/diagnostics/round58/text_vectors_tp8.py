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

ROOT = Path('/tmp/h3_round58_vec01')
CAP = Path('/tmp/h3_round54_capture01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
OUT = ROOT / f'text_vectors_rank{rank}.json'
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
from native_bf16_vector58 import native_launch

spec = importlib.util.spec_from_file_location('vector_candidate58', ROOT / 'candidate_text_linear_encoder.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate._apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
candidate._scaled_dot_product_attention = causal_fp32
NORM, ACT = False, False
VERIFY_ACT, ACT_CHECKS = False, []


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
r = {'rank': rank, 'cases': {}, 'scope': 'Resident experimental text-only TP8 with FP32 linears/causal and HF math fixed; only vector fusion changes, not full encoding/lossless video'}


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
    for label, use_norm, use_act in [('baseline', False, False), ('swiglu_only', False, True), ('norm_only', True, False), ('norm_swiglu', True, True)]:
        NORM, ACT = use_norm, use_act
        VERIFY_ACT, ACT_CHECKS = use_act, []
        torch.distributed.barrier(); torch.npu.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            out = model(x, positions, visual_pos_masks=mask, deepstack_visual_embeds=deep)
        torch.npu.synchronize()
        first = time.perf_counter() - t
        cpu = out.detach().cpu().contiguous()
        VERIFY_ACT = False
        torch.distributed.barrier(); torch.npu.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            repeat = model(x, positions, visual_pos_masks=mask, deepstack_visual_embeds=deep)
        torch.npu.synchronize()
        second = time.perf_counter() - t
        repeat_cpu = repeat.detach().cpu().contiguous()
        row = {'first_forward_s': first, 'repeat_forward_s': second, 'shape': list(out.shape),
               'finite': bool(torch.isfinite(cpu.float()).all()), 'repeat_vs_first': metric(repeat_cpu, cpu),
               'first_forward_includes_act_reference_checks': use_act, 'act_reference_checks': ACT_CHECKS}
        if use_act:
            assert len(ACT_CHECKS) == 50
        if rank == 0:
            outputs[label] = cpu.clone()
            if label != 'baseline':
                row['vs_same_run_baseline'] = metric(cpu, outputs['baseline'])
            teacher = torch.load(Path('/tmp/h3_round56_audit01/cpu_last_hf.pt'), weights_only=True)['output']
            row['vs_cpu_fp32_oracle'] = metric(cpu, teacher)
        r['cases'][label] = row
        OUT.write_text(json.dumps(r, indent=2) + '\n')
        print(json.dumps({'rank': rank, 'case': label, **row}), flush=True)
    if rank == 0:
        target = ROOT / 'text_vectors_outputs.pt'
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
