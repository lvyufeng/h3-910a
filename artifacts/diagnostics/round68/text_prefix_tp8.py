#!/usr/bin/env python3
"""Unchanged native-reduction Norm-prefix A/B and current-stack event profile.

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
sys.path.insert(0, '/tmp/h3_round65_shadow01')
from shadow_cache65 import ShadowWeights, SharedActivation
sys.path.insert(0, '/tmp/h3_round67_norm01')
from norm_tail_guard67 import norm_tail
from norm_prefix68 import norm_prefix
sys.path.insert(0, '/tmp/h3_round66_profile01')
from event_profile66 import EventProfile
from contextlib import contextmanager
from evidence_io68 import EvidenceWriter
sys.path.insert(0, '/tmp/h3_round60_rope01')
sys.path.insert(0, '/tmp/h3_round58_vec01')
from h3_text_causal57 import causal_fp32
from native_bf16_vector58 import native_launch
from rope_vector60 import rope
import padded_fp32_projection59 as padding

ROOT = Path('/tmp/h3_round68_prefix01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
writer = EvidenceWriter(ROOT / ('prefix_rank%d.json' % rank))
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
                 for name in ('text_prefix_tp8.py', 'norm_prefix68.py', 'candidate_padded_text_encoder.py', 'evidence_io68.py')}}


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


NORM_TAIL, VERIFY_NORM = False, False
NORM_CHECKS = []
old_norm = candidate.MiniMaxH3Qwen3VLRMSNorm.forward_npu


def native_statistics_norm(self, x, residual=None):
    if not NORM_TAIL:
        return old_norm(self, x, residual)
    original_x, original_residual = x, residual
    input_dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        # Identical native operations and rounding as the current reference.
        if residual is not None:
            residual = (residual.float() + x.float()).to(input_dtype)
            x = residual
        xf = x.float()
        variance = xf.square().mean(-1, keepdim=True)
        inverse = torch.rsqrt(variance + self.variance_epsilon)
        output = norm_tail(x, self.weight, inverse)
    if VERIFY_NORM:
        reference = old_norm(self, original_x, original_residual)
        expected = reference[0] if isinstance(reference, tuple) else reference
        proof = metric(output, expected)
        row = {'name': self._norm67_name, 'shape': list(x.shape),
               'aligned_input_gamma_inverse': all(value.data_ptr() % 32 == 0 for value in (x, self.weight, inverse)),
               'output': proof}
        if residual is not None:
            row['residual'] = metric(residual, reference[1])
        NORM_CHECKS.append(row)
        # Small selected actual pre-Norm fixtures, NOT post-RoPE lookalikes.
        layer = int(self._norm67_name.split('.')[0])
        if rank == 0 and layer in (0, 24, 49):
            target = ROOT / 'capture' / (self._norm67_name + '.pt')
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                raise RuntimeError('Refuse actual Norm fixture overwrite')
            values = {'x': x.detach().cpu().contiguous(), 'gamma': self.weight.detach().cpu().contiguous(),
                      'inverse_rms': inverse.detach().cpu().contiguous(), 'native_output': expected.detach().cpu().contiguous(),
                      'candidate_output': output.detach().cpu().contiguous()}
            with target.open('xb') as f:
                torch.save(values, f)
        if not proof['raw_bits_equal'] or ('residual' in row and not row['residual']['raw_bits_equal']):
            raise AssertionError('Same-input actual Norm or residual output differs')
    return (output, residual) if residual is not None else output


candidate.MiniMaxH3Qwen3VLRMSNorm.forward_npu = native_statistics_norm

STATS_MODE, PREFIX_VERIFY = 'native', False
PREFIX_CHECKS = []
old_current_norm = native_statistics_norm


def prefix_statistics_norm(self, x, residual=None):
    if STATS_MODE in ('native', 'restored'):
        return old_current_norm(self, x, residual)
    original_x, original_residual = x, residual
    with torch.autocast(device_type=x.device.type, enabled=False):
        if STATS_MODE == 'square':
            if residual is not None:
                residual = (residual.float() + x.float()).to(x.dtype)
                x = residual
            squares, updated = norm_prefix(x)
        elif STATS_MODE == 'add_square':
            squares, updated = norm_prefix(x, residual)
            if residual is not None:
                residual = updated
        else:
            raise ValueError('Unknown prefix mode')
        variance = squares.mean(-1, keepdim=True)
        inverse = torch.rsqrt(variance + self.variance_epsilon)
        output = norm_tail(updated, self.weight, inverse)
    if PREFIX_VERIFY:
        with torch.autocast(device_type=original_x.device.type, enabled=False):
            native_updated = ((original_residual.float() + original_x.float()).to(original_x.dtype)
                              if original_residual is not None else original_x)
            native_square = native_updated.float().square()
            native_variance = native_square.mean(-1, keepdim=True)
            native_inverse = torch.rsqrt(native_variance + self.variance_epsilon)
            native_output = norm_tail(native_updated, self.weight, native_inverse)
        row = {'name': self._norm68_name, 'shape': list(original_x.shape),
               'square': metric(squares, native_square), 'variance': metric(variance, native_variance),
               'inverse': metric(inverse, native_inverse), 'output': metric(output, native_output),
               'square_layout': [list(squares.shape), list(squares.stride()), str(squares.dtype), torch_npu.get_npu_format(squares)],
               'native_square_layout': [list(native_square.shape), list(native_square.stride()), str(native_square.dtype), torch_npu.get_npu_format(native_square)]}
        if original_residual is not None:
            row['residual'] = metric(residual, native_updated)
        PREFIX_CHECKS.append(row)
        valid = all(row[key]['raw_bits_equal'] and row[key]['finite'] for key in ('square', 'variance', 'inverse', 'output'))
        valid = valid and row['square_layout'] == row['native_square_layout']
        valid = valid and ('residual' not in row or row['residual']['raw_bits_equal'])
        layer, family = self._norm68_name.split('.')
        if rank == 0 and STATS_MODE == 'square' and family == 'hidden_residual' and int(layer) in (0, 24, 49):
            target = ROOT / 'capture' / (self._norm68_name + '.pt')
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                raise RuntimeError('Refuse actual pre-add fixture overwrite')
            values = {'original_x': original_x.detach().cpu().contiguous(),
                      'original_residual': original_residual.detach().cpu().contiguous(),
                      'native_updated': native_updated.detach().cpu().contiguous(),
                      'native_variance': native_variance.detach().cpu().contiguous(),
                      'native_inverse': native_inverse.detach().cpu().contiguous()}
            with target.open('xb') as f:
                torch.save(values, f)
        if not valid:
            raise AssertionError('Same-input FP32 square/variance/inverse/layout/Norm/residual changed')
    return (output, residual) if residual is not None else output


candidate.MiniMaxH3Qwen3VLRMSNorm.forward_npu = prefix_statistics_norm

# Only diagnostic wrappers are installed; no original arithmetic body changes.
profiler = None

def measured(name, function, *args, **kwargs):
    return profiler.call(name, function, *args, **kwargs) if profiler is not None else function(*args, **kwargs)

old_linear = F.linear

def profile_linear(x, weight, *args, **kwargs):
    return measured('child.linear_%dx%d' % tuple(weight.shape), old_linear, x, weight, *args, **kwargs)

F.linear = profile_linear

def profile_local(x, weight):
    tag = 'kv_padded' if tuple(weight.shape) == (128, 5120) else ('q' if tuple(weight.shape) == (1024, 5120) else 'gate_up')
    return measured('top.local_' + tag, local, x, weight)

candidate._text_fp32_local_linear = profile_local

def profile_row(self, x):
    tag = 'o' if self.weight.shape[-1] == 1024 else 'down'
    return measured('top.row_' + tag, row_forward, self, x)

candidate.MiniMaxH3Qwen3VLRowParallelLinear.forward = profile_row
profile_norm_impl = candidate.MiniMaxH3Qwen3VLRMSNorm.forward_npu

def profile_norm(self, x, residual=None):
    tag = ('q_head' if x.shape[-2] == 8 else 'k_head') if x.shape[-1] == 128 else ('hidden_residual' if residual is not None else 'hidden_input')
    return measured('top.norm_' + tag, profile_norm_impl, self, x, residual)

candidate.MiniMaxH3Qwen3VLRMSNorm.forward_npu = profile_norm
old_rope = candidate._apply_rotary_pos_emb
candidate._apply_rotary_pos_emb = lambda *args: measured('top.rope', old_rope, *args)
candidate._scaled_dot_product_attention = lambda *args: measured('top.causal_fp32', causal_fp32, *args)
old_swiglu = native_launch
native_launch = lambda *args, **kwargs: measured('top.swiglu', old_swiglu, *args, **kwargs)
old_deepstack = candidate.MiniMaxH3Qwen3VLTextModel._deepstack_process

def profile_deepstack(self, *args):
    return measured('top.deepstack', old_deepstack, self, *args)

candidate.MiniMaxH3Qwen3VLTextModel._deepstack_process = profile_deepstack
old_activation_scope = activations.scope

@contextmanager
def profile_activation_scope(value):
    manager = old_activation_scope(value)
    measured('top.shared_activation_cast', manager.__enter__)
    try:
        yield
    finally:
        manager.__exit__(None, None, None)

activations.scope = profile_activation_scope

try:
    init_distributed_environment(world_size=size, rank=rank, local_rank=rank, backend='hccl')
    initialize_model_parallel(sequence_parallel_size=size, ulysses_degree=size, tensor_parallel_size=1, backend='hccl')
    group = GroupCoordinator([list(range(size))], rank, 'hccl')
    old_reduce = group.all_reduce
    group.all_reduce = lambda *args, **kwargs: measured('child.all_reduce', old_reduce, *args, **kwargs)
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
    for layer, block in enumerate(model.layers):
        for name, module in (('hidden_input', block.input_layernorm), ('hidden_residual', block.post_attention_layernorm),
                             ('q_head', block.self_attn.q_norm), ('k_head', block.self_attn.k_norm)):
            module._norm68_name = '%02d.%s' % (layer, name)
    r['parameters_before'] = parameters_sha(model)
    d = torch.load('/tmp/h3_round54_capture01/text_inputs.pt', weights_only=True)
    x, positions, mask = [d[key].to(torch.device('npu', rank)) for key in ('inputs_embeds', 'positions', 'visual_pos_masks')]
    deep = [value.to(torch.device('npu', rank)) for value in d['deepstack_visual_embeds']]
    r['inputs_before'] = [raw_hash(value) for value in (x, positions, mask, *deep)]
    profiler = EventProfile(torch)
    baseline = None
    MODE, VERIFY, NORM_TAIL, VERIFY_NORM = 'shared_qkv', False, True, False
    for STATS_MODE in ('native', 'square', 'add_square', 'restored'):
        PREFIX_VERIFY = STATS_MODE in ('square', 'add_square')
        PREFIX_CHECKS.clear()
        first, first_s = once(model, x, positions, mask, deep)
        PREFIX_VERIFY = False
        if baseline is None:
            baseline = first
            old = torch.load('/tmp/h3_round60_rope01/text_rope_outputs.pt', weights_only=True)['tbe_native']
            r['baseline_vs_round60_full_output'] = metric(first, old)
            assert r['baseline_vs_round60_full_output']['raw_bits_equal']
        row = {'first_s_excluded_checks_compilation': first_s, 'stats_mode': STATS_MODE,
               'first_vs_baseline': metric(first, baseline), 'prefix_checks': PREFIX_CHECKS.copy(),
               'clean_before': [], 'profiles': [], 'clean_after': []}
        r['cases'][STATS_MODE] = row
        assert row['first_vs_baseline']['raw_bits_equal']
        for _ in range(3):
            out, seconds = once(model, x, positions, mask, deep)
            proof = metric(out, baseline)
            row['clean_before'].append({'seconds': seconds, 'vs_baseline': proof, 'output_sha256': raw_hash(out)})
            assert proof['raw_bits_equal']
        if STATS_MODE != 'restored':
            for _ in range(3):
                profiler.begin()
                out, seconds = once(model, x, positions, mask, deep)
                events = profiler.finish()
                proof = metric(out, baseline)
                row['profiles'].append({'seconds': seconds, 'events': events,
                                        'stream_begin': profiler.stream_begin, 'stream_end': profiler.stream_end,
                                        'vs_baseline': proof, 'output_sha256': raw_hash(out)})
                assert proof['raw_bits_equal']
            for _ in range(3):
                out, seconds = once(model, x, positions, mask, deep)
                proof = metric(out, baseline)
                row['clean_after'].append({'seconds': seconds, 'vs_baseline': proof, 'output_sha256': raw_hash(out)})
                assert proof['raw_bits_equal']
        row['shadow_cache'], row['existing_padding_cache'] = weights.stats(), padding.cache_stats()
        row['hbm_allocated_bytes'], row['hbm_reserved_bytes'] = torch.npu.memory_allocated(), torch.npu.memory_reserved()
        writer.write(r)
        print(json.dumps({'rank': rank, 'mode': STATS_MODE, 'clean_s': [call['seconds'] for call in row['clean_before']],
                          'profile_s': [call['seconds'] for call in row['profiles']], 'actual_prefix_checks': len(row['prefix_checks']),
                          'full_output_exact': True}), flush=True)
    r['parameters_after'] = parameters_sha(model)
    r['inputs_after'] = [raw_hash(value) for value in (x, positions, mask, *deep)]
    assert r['parameters_before'] == r['parameters_after'] and r['inputs_before'] == r['inputs_after']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    r['prefix_checks_at_failure'] = PREFIX_CHECKS.copy()
    raise
finally:
    weights.clear()
    if torch.distributed.is_initialized():
        destroy_model_parallel()
        torch.distributed.destroy_process_group()
    writer.write(r)
    print('Evidence:', ROOT / ('prefix_rank%d.json' % rank), flush=True)
