#!/usr/bin/env python3
"""Bounded repeatability audit: unchanged experimental model, inputs, parameters.

First test clean forwards, then synchronize selected/all-layer observations.
Full hashes are reported for inputs and every parameter. Layer hashes cover
all elements on all ranks, while numerical comparison uses full rank0 and
bounded samples elsewhere. Hooks always return None and do not change math.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT = Path('/tmp/h3_round59_repeat01')
CAP = Path('/tmp/h3_round54_capture01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
OUT = ROOT / f'repeat_audit_rank{rank}.json'
if OUT.exists():
    raise RuntimeError('Refuse repeat audit overwrite')
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

spec = importlib.util.spec_from_file_location('repeat_candidate59', ROOT / 'candidate_text_linear_encoder.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate._apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
candidate._scaled_dot_product_attention = causal_fp32


def mlp_forward(self, x):
    packed = self.gate_up_proj(x)
    half = packed.shape[-1] // 2
    return self.down_proj(F.silu(packed[..., :half]) * packed[..., half:])


candidate.MiniMaxH3Qwen3VLTextMLP.forward = mlp_forward
r = {'rank': rank, 'forwards': {}, 'layer_rows': [], 'stage_rows': [],
     'scope': 'Unchanged experimental FP32-linear/causal/HF-math text stack repeatability; not production/encoder/video benchmark'}
PHASE = ''
OBSERVE = False
LOCAL_CALL = 0
BASE = {}
FIRST_DIFF_SAVED = set()


def cpu_tensor(x):
    return x.detach().cpu().contiguous()


def digest(x):
    x = cpu_tensor(x)
    return hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()


def metric(a, b):
    x, y = a.float().double(), b.float().double()
    e = x - y
    return {'raw_bits_equal': a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
            'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'max_abs_error': float(e.abs().max()), 'relative_rms': float(e.square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-30))}


def payload(x):
    if rank == 0:
        return x.clone()
    y = x.reshape(-1, x.shape[-1])
    idx = sorted({0, min(5, y.shape[0]-1), min(512, y.shape[0]-1), y.shape[0]-1})
    return y[idx].clone()


def record(key, value, kind, inputs=None):
    if not OBSERVE:
        return
    value = cpu_tensor(value)
    h = digest(value)
    sample = payload(value)
    row = {'phase': PHASE, 'key': key, 'dtype': str(value.dtype), 'shape': list(value.shape), 'full_sha256': h,
           'metric_scope': 'full' if rank == 0 else 'four_selected_rows'}
    if inputs is not None:
        row['input_sha256'] = digest(inputs)
    if key not in BASE:
        BASE[key] = (h, sample)
        row['comparison_base_phase'] = PHASE
        if rank == 0 and key in ('layer0_output', 'layer49_output'):
            target = ROOT / 'evidence' / (key + '_base.pt')
            if target.exists():
                raise RuntimeError('Refuse base evidence overwrite')
            assert sample.numel() * sample.element_size() < 512 * 1024 * 1024
            with target.open('xb') as f:
                torch.save(sample, f)
    else:
        old_hash, old_value = BASE[key]
        row['full_bits_equal_baseline'] = old_hash == h
        row['numerical_vs_baseline'] = metric(sample, old_value)
        if rank == 0 and old_hash != h and key.startswith('layer') and key.endswith('_output') and PHASE not in FIRST_DIFF_SAVED:
            target = ROOT / 'evidence' / (PHASE + '_first_different_layer.pt')
            if target.exists():
                raise RuntimeError('Refuse divergence evidence overwrite')
            assert (sample.numel()+old_value.numel())*sample.element_size() < 512*1024*1024
            with target.open('xb') as f:
                torch.save({'key': key, 'baseline': old_value, 'actual': sample}, f)
            FIRST_DIFF_SAVED.add(PHASE)
    r['layer_rows' if kind == 'layer' else 'stage_rows'].append(row)


original_local = candidate._text_fp32_local_linear


def local_linear(x, weight):
    global LOCAL_CALL
    i = LOCAL_CALL
    LOCAL_CALL += 1
    value = original_local(x, weight)
    if OBSERVE and i < 4:
        record('layer0_local_' + ('q', 'k', 'v', 'gate_up')[i], value, 'stage', x)
    return value


candidate._text_fp32_local_linear = local_linear


def inputs_digest(values):
    return {name: digest(value) for name, value in values.items()}


def parameters_digest(model):
    result = {}
    for name, p in model.named_parameters():
        result[name] = {'shape': list(p.shape), 'dtype': str(p.dtype), 'sha256': digest(p)}
    return result


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
            keys = [k for k, v in index.items() if v == shard and k.startswith('model.language_model.layers.') and int(k.split('.')[3]) < 50]
            if not keys:
                continue
            with safe_open(str(MODEL / shard), framework='pt', device='cpu') as f:
                for key in keys:
                    yield key, f.get_tensor(key)
        yield 'model.language_model.embed_tokens.weight', torch.zeros(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    t = time.perf_counter()
    loaded = owner.load_weights(weights())
    r['load_s'], r['loaded_tensors'] = time.perf_counter()-t, len(loaded)
    model.embed_tokens = nn.Identity()
    model.to(torch.device('npu', rank))
    r['parameters_before'] = parameters_digest(model)
    source_inputs = torch.load(CAP / 'text_inputs.pt', weights_only=True)
    source = {'input': source_inputs['inputs_embeds'], 'positions': source_inputs['positions'], 'mask': source_inputs['visual_pos_masks']}
    source.update({f'deep{i}': x for i, x in enumerate(source_inputs['deepstack_visual_embeds'])})
    source_hash = inputs_digest(source)
    shared = {k: v.to('npu') for k, v in source.items()}
    assert inputs_digest(shared) == source_hash
    r['source_input_sha256'] = source_hash
    r['context'] = {'autocast_enabled': torch.is_autocast_enabled('npu'), 'matmul_allow_hf32': torch.npu.matmul.allow_hf32,
                    'conv_allow_hf32': torch.npu.conv.allow_hf32, 'default_device': str(torch.get_default_device())}
    first_clean = None
    handles = []
    for phase in ('clean0', 'clean1', 'clean2', 'hook0', 'hook1', 'hook2', 'fresh_hook0', 'fresh_hook1'):
        PHASE, LOCAL_CALL = phase, 0
        if phase == 'hook0':
            for i, module in enumerate(model.layers):
                def output_hook(mod, args, out, i=i):
                    record(f'layer{i}_output', out, 'layer')
                    return None
                handles.append(module.register_forward_hook(output_hook))
            for name, module in [('norm1', model.layers[0].input_layernorm), ('qnorm', model.layers[0].self_attn.q_norm),
                                 ('knorm', model.layers[0].self_attn.k_norm), ('attention', model.layers[0].self_attn),
                                 ('norm2', model.layers[0].post_attention_layernorm), ('gate_up', model.layers[0].mlp.gate_up_proj),
                                 ('down', model.layers[0].mlp.down_proj)]:
                def stage_hook(mod, args, out, name=name):
                    value = out[0] if isinstance(out, tuple) else out
                    record('layer0_' + name, value, 'stage', args[0])
                    return None
                handles.append(module.register_forward_hook(stage_hook))
            def rotary_hook(mod, args, out):
                record('rope_cos', out[0], 'stage'); record('rope_sin', out[1], 'stage')
                return None
            handles.append(model.rotary_emb.register_forward_hook(rotary_hook))
        OBSERVE = not phase.startswith('clean')
        values = {k: v.clone().to('npu') for k, v in source.items()} if phase.startswith('fresh') else shared
        before = inputs_digest(values)
        torch.distributed.barrier(); torch.npu.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            out = model(values['input'], values['positions'], visual_pos_masks=values['mask'],
                        deepstack_visual_embeds=[values['deep0'], values['deep1'], values['deep2']])
        torch.npu.synchronize()
        dt = time.perf_counter()-t
        cpu = cpu_tensor(out)
        if first_clean is None:
            first_clean = cpu.clone()
        after = inputs_digest(values)
        row = {'forward_s': dt, 'output_sha256': digest(cpu), 'vs_first_clean': metric(cpu, first_clean),
               'inputs_bits_equal_source_before': before == source_hash, 'inputs_unchanged_after': after == before,
               'inputs_sha256_after': after, 'local_linear_calls': LOCAL_CALL, 'observations_synchronize': OBSERVE}
        r['forwards'][phase] = row
        OUT.write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps({'rank': rank, 'phase': phase, **row}), flush=True)
    r['parameters_after'] = parameters_digest(model)
    r['all_parameter_bits_unchanged'] = r['parameters_before'] == r['parameters_after']
    r['all_input_bits_unchanged'] = all(x['inputs_unchanged_after'] and x['inputs_bits_equal_source_before'] for x in r['forwards'].values())
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2)+'\n')
    print('Evidence:', OUT, flush=True)
    if torch.distributed.is_initialized():
        destroy_model_parallel(); torch.distributed.destroy_process_group()
