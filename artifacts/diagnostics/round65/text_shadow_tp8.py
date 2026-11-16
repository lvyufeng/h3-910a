#!/usr/bin/env python3
"""Exact conversion-cache ablation, actual stable text inputs and TP8 weights.

Resident experimental text-only stack, not full service encode, vision, offload,
CPU-oracle equivalence or official-lossless video quality. No production edits.
"""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
os.environ.update(VLLM_OMNI_910A_NO_FUSED_NORM='1', VLLM_OMNI_910A_NO_FUSED_ACT='1',
                  VLLM_OMNI_910A_TEXT_HF_MATH='1', VLLM_OMNI_910A_TEXT_FP32_LINEAR='1',
                  VLLM_OMNI_910A_TEXT_PAD_KV='1')
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig
from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel
from shadow_cache65 import ShadowWeights, SharedActivation
from evidence_io65 import EvidenceWriter
sys.path.insert(0, '/tmp/h3_round60_rope01')
sys.path.insert(0, '/tmp/h3_round58_vec01')
from h3_text_causal57 import causal_fp32
from native_bf16_vector58 import native_launch
from rope_vector60 import rope
import padded_fp32_projection59 as padding

ROOT = Path('/tmp/h3_round65_shadow01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
writer = EvidenceWriter(ROOT / ('shadow_rank%d.json' % rank))
torch.npu.set_device(rank)
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('text_candidate65', ROOT / 'candidate_padded_text_encoder.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
candidate._scaled_dot_product_attention = causal_fp32
candidate._apply_rotary_pos_emb = lambda q, k, cos, sin: (rope(q, cos, sin), rope(k, cos, sin))
weights, activations = ShadowWeights(), SharedActivation()
MODE, VERIFY = 'baseline', False
SLOTS, CHECKS = {}, []
r = {'rank': rank, 'cases': {}, 'scope': __doc__,
     'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in ('text_shadow_tp8.py', 'shadow_cache65.py', 'candidate_padded_text_encoder.py', 'evidence_io65.py')}}


def raw_hash(value):
    cpu = value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()


def metric(x, y):
    a, b = x.detach().cpu().contiguous(), y.detach().cpu().contiguous()
    assert a.shape == b.shape and a.dtype == b.dtype
    e = a.double() - b.double()
    return {'raw_bits_equal': torch.equal(a.view(torch.uint8), b.view(torch.uint8)), 'elements': a.numel(),
            'finite': bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()),
            'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / b.double().square().mean().sqrt().clamp_min(1e-30))}


def lookup(weight):
    return (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()))


def linear_fp32(x, weight):
    x32 = activations.get(x) if MODE in ('shared_qkv', 'activation_only') else x.float()
    w32 = weight.float() if MODE == 'activation_only' else weights.get(SLOTS[lookup(weight)], weight)
    return F.linear(x32, w32)


original_local = candidate._text_fp32_local_linear


def local(x, weight):
    if MODE in ('baseline', 'restored'):
        return original_local(x, weight)
    with torch.autocast(device_type=x.device.type, enabled=False):
        if padding.supported(x, weight):
            x32 = activations.get(x) if MODE in ('shared_qkv', 'activation_only') else x.float()
            value = F.linear(x32.contiguous(), padding.padded_weight(weight))[..., :128].contiguous().to(x.dtype)
        else:
            value = linear_fp32(x, weight).to(x.dtype)
    if VERIFY:
        proof = metric(value, original_local(x, weight))
        CHECKS.append({'kind': 'local', 'slot': SLOTS.get(lookup(weight), 'padded_kv'), **proof})
        if not proof['raw_bits_equal']:
            raise AssertionError('Same-input local projection changed under exact cast reuse')
    return value


candidate._text_fp32_local_linear = local
original_attention = candidate.MiniMaxH3Qwen3VLTextAttention.forward


def attention_forward(self, hidden_states, position_embeddings):
    context = activations.scope(hidden_states) if MODE in ('shared_qkv', 'activation_only') else nullcontext()
    with context:
        return original_attention(self, hidden_states, position_embeddings)


candidate.MiniMaxH3Qwen3VLTextAttention.forward = attention_forward
original_row = candidate.MiniMaxH3Qwen3VLRowParallelLinear.forward


def row_forward(self, input_):
    if MODE in ('baseline', 'restored', 'activation_only'):
        return original_row(self, input_)
    # The fixed model uses input_is_parallel=True, but retain original splitting
    # semantics for completeness and never change the actual GEMM shape.
    parallel = input_ if self.input_is_parallel else input_.split(self.input_size_per_partition, dim=-1)[self._tp_rank].contiguous()
    assert parallel.dtype == self.weight.dtype == torch.bfloat16
    with torch.autocast(device_type=parallel.device.type, enabled=False):
        value = linear_fp32(parallel, self.weight)
        if VERIFY:
            reference = F.linear(parallel.float(), self.weight.float())
            proof = metric(value, reference)
            CHECKS.append({'kind': 'row_partial_fp32', 'slot': SLOTS[lookup(self.weight)], **proof})
            if not proof['raw_bits_equal']:
                raise AssertionError('Same-input FP32 partial changed before all-reduce')
        if self._tp_size > 1:
            value = self.group.all_reduce(value)
        return value.to(self.output_dtype)


candidate.MiniMaxH3Qwen3VLRowParallelLinear.forward = row_forward


def mlp_forward(self, x):
    # Exactly the round60 geometry: one merged gate/up GEMM, not two split GEMMs.
    packed = self.gate_up_proj(x)
    return self.down_proj(native_launch('swiglu', packed.contiguous())[0])


candidate.MiniMaxH3Qwen3VLTextMLP.forward = mlp_forward


def parameters_sha(model):
    total = hashlib.sha256()
    for name, parameter in model.named_parameters():
        total.update(name.encode())
        total.update(memoryview(parameter.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return total.hexdigest()


def once(model, x, positions, mask, deep):
    torch.distributed.barrier()
    torch.npu.synchronize()
    t = time.perf_counter()
    with torch.inference_mode():
        out = model(x, positions, visual_pos_masks=mask, deepstack_visual_embeds=deep)
    torch.npu.synchronize()
    seconds = time.perf_counter() - t
    return out.detach().cpu().contiguous(), seconds


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
    def tensors():
        for shard in sorted(set(index.values())):
            keys = [key for key, name in index.items() if name == shard and key.startswith('model.language_model.layers.') and int(key.split('.')[3]) < 50]
            if not keys:
                continue
            with safe_open(str(MODEL / shard), framework='pt', device='cpu') as f:
                for key in keys:
                    yield key, f.get_tensor(key)
        yield 'model.language_model.embed_tokens.weight', torch.zeros(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    t = time.perf_counter()
    loaded = owner.load_weights(tensors())
    r['loaded_parameter_tensors'], r['cpu_load_s'] = len(loaded), time.perf_counter() - t
    model.embed_tokens = nn.Identity()
    model.to(torch.device('npu', rank))
    r['bf16_model_parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
    for layer, block in enumerate(model.layers):
        qkv = block.self_attn.qkv_proj
        width = qkv.local_num_heads * qkv.head_dim
        SLOTS[lookup(qkv.weight[:width])] = ('q', layer)
        for name, module in (('gate_up', block.mlp.gate_up_proj), ('o', block.self_attn.o_proj), ('down', block.mlp.down_proj)):
            SLOTS[lookup(module.weight)] = (name, layer)
    assert len(SLOTS) == 200
    r['parameters_before'] = parameters_sha(model)
    d = torch.load('/tmp/h3_round54_capture01/text_inputs.pt', weights_only=True)
    x, positions, mask = [d[key].to(torch.device('npu', rank)) for key in ('inputs_embeds', 'positions', 'visual_pos_masks')]
    deep = [value.to(torch.device('npu', rank)) for value in d['deepstack_visual_embeds']]
    r['inputs_before'] = [raw_hash(value) for value in (x, positions, mask, *deep)]
    baseline = None
    for MODE in ('baseline', 'activation_only', 'weight_shadow', 'shared_qkv', 'restored'):
        CHECKS.clear()
        VERIFY = MODE in ('activation_only', 'weight_shadow', 'shared_qkv')
        first, first_s = once(model, x, positions, mask, deep)
        VERIFY = False
        if baseline is None:
            baseline = first
            old = torch.load('/tmp/h3_round60_rope01/text_rope_outputs.pt', weights_only=True)['tbe_native']
            r['baseline_vs_round60_full_output'] = metric(first, old)
            assert r['baseline_vs_round60_full_output']['raw_bits_equal']
        row = {'first_s': first_s, 'first_includes_same_input_projection_checks': MODE in ('activation_only', 'weight_shadow', 'shared_qkv'),
               'first_vs_baseline': metric(first, baseline), 'projection_checks': CHECKS.copy(), 'warm_calls': []}
        r['cases'][MODE] = row
        assert row['first_vs_baseline']['raw_bits_equal']
        for _ in range(3):
            out, seconds = once(model, x, positions, mask, deep)
            proof = metric(out, baseline)
            row['warm_calls'].append({'seconds': seconds, 'vs_baseline': proof, 'output_sha256': raw_hash(out)})
            assert proof['raw_bits_equal']
        row['shadow_cache'] = weights.stats()
        row['existing_padding_cache'] = padding.cache_stats()
        row['shared_activation'] = {'conversions': activations.conversions, 'hits': activations.hits, 'scope_closed': activations.active is None}
        row['hbm_allocated_bytes'] = torch.npu.memory_allocated()
        row['hbm_reserved_bytes'] = torch.npu.memory_reserved()
        writer.write(r)
        print(json.dumps({'rank': rank, 'mode': MODE, 'warm_s': [call['seconds'] for call in row['warm_calls']],
                          'exact': True, 'cache': row['shadow_cache']}), flush=True)
    r['parameters_after'] = parameters_sha(model)
    r['inputs_after'] = [raw_hash(value) for value in (x, positions, mask, *deep)]
    assert r['parameters_before'] == r['parameters_after'] and r['inputs_before'] == r['inputs_after']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    weights.clear()
    if torch.distributed.is_initialized():
        destroy_model_parallel()
        torch.distributed.destroy_process_group()
    writer.write(r)
    print('Evidence:', ROOT / ('shadow_rank%d.json' % rank), flush=True)
