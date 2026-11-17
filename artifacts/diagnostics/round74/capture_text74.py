#!/usr/bin/env python3
"""Excluded current-experimental text-layer captures for independent HF attribution.

Complete resident experimental encoder with fresh vision and real vocabulary
embedding. Excludes upstream HTTP preprocessing/broadcast and model load/offload;
not original BF16/HF-oracle or official-lossless video quality. No production edits.
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
                  VLLM_OMNI_910A_TEXT_PAD_KV='1', VLLM_OMNI_910A_VISION_FP32='1')
os.environ.pop('VLLM_OMNI_910A_TE_DTYPE', None)
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
sys.path.insert(0, '/tmp/h3_round68_prefix01')
from norm_prefix68 import norm_prefix
sys.path.insert(0, '/tmp/h3_round69_cast01')
from activation_cast69 import expand_activation
sys.path.insert(0, '/tmp/h3_round66_profile01')
from event_profile66 import EventProfile
from contextlib import contextmanager
from collections import Counter
from evidence_io74 import EvidenceWriter
sys.path.insert(0,'/tmp/h3_round73_cast_ab02')
sys.path.insert(0,'/tmp/h3_round73_cast02')
from cast_attention73 import CastAttention
sys.path.insert(0, '/tmp/h3_round71_vision01')
sys.path.insert(0, '/tmp/h3_round72_tail01')
from vision_rope_tail72 import rope_pair
from vision_linear71 import VisionLinearModes
sys.path.insert(0, '/tmp/h3_round60_rope01')
sys.path.insert(0, '/tmp/h3_round58_vec01')
from h3_text_causal57 import causal_fp32
from native_bf16_vector58 import native_launch
from rope_vector60 import rope
import padded_fp32_projection59 as padding

ROOT = Path('/tmp/h3_round74_capture01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
writer = EvidenceWriter(ROOT / ('capture_rank%d.json' % rank))
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
                 for name in ('capture_text74.py', 'candidate_padded_text_encoder.py', 'evidence_io74.py')}}


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

CAST_MODE, VERIFY_CAST = 'native', False
CAST_CHECKS = []


def activation_layout(value):
    return [list(value.shape), list(value.stride()), str(value.dtype), torch_npu.get_npu_format(value)]


class MeasuredActivation(SharedActivation):
    def _convert(self, value, kind):
        enabled = (CAST_MODE in ('shared_only', 'both') if kind == 'shared' else CAST_MODE in ('fallback_only', 'both'))
        before = value.detach().cpu().contiguous().clone() if VERIFY_CAST else None
        converted = expand_activation(value) if enabled else value.float()
        if VERIFY_CAST:
            reference = value.float()
            proof = metric(converted, reference)
            row = {'kind': kind, 'candidate_enabled': enabled, 'input_shape': list(value.shape),
                   'input_dtype': str(value.dtype), 'proof': proof,
                   'candidate_layout': activation_layout(converted), 'native_layout': activation_layout(reference),
                   'input_readonly': torch.equal(value.detach().cpu().contiguous().view(torch.uint8), before.view(torch.uint8))}
            CAST_CHECKS.append(row)
            if not proof['raw_bits_equal'] or not proof['finite'] or row['candidate_layout'] != row['native_layout'] or not row['input_readonly']:
                raise AssertionError('Actual activation cast bits/layout/input changed')
        return converted

    @contextmanager
    def scope(self, value):
        if self.active is not None:
            raise RuntimeError('Nested shared-activation scopes are unsupported')
        version = self.signature(value)
        converted = self._convert(value, 'shared')
        self.conversions += 1
        self.active = (value, version, converted)
        try:
            yield
        finally:
            self.active = None

    def get(self, value):
        if self.active is not None:
            original, descriptor, converted = self.active
            if value is original:
                if self.signature(value) != descriptor:
                    raise RuntimeError('Attention input changed during exact activation reuse')
                self.hits += 1
                return converted
        self.conversions += 1
        return self._convert(value, 'fallback')


activations = MeasuredActivation()

VERIFY_NORM70, PARTS_ACTIVE = False, False
NORM_CHECKS70, PARTS = [], []
CALL_OBJECTS = {}
vision_modes = None


def norm_dispatch70(self, x, residual=None):
    value = prefix_statistics_norm(self, x, residual)
    if VERIFY_NORM70:
        reference = old_norm(self, x, residual)
        proof = metric(value[0] if isinstance(value, tuple) else value, reference[0] if isinstance(reference, tuple) else reference)
        row = {'name': self._norm68_name, 'output': proof}
        if residual is not None:
            row['residual'] = metric(value[1], reference[1])
        NORM_CHECKS70.append(row)
        if not proof['raw_bits_equal'] or ('residual' in row and not row['residual']['raw_bits_equal']):
            raise AssertionError('Actual full-encoder Norm differs from its same-input native reference')
    return value


candidate.MiniMaxH3Qwen3VLRMSNorm.forward_npu = norm_dispatch70


def tensor_identity(value):
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(),
                'sha256': raw_hash(cpu), 'finite': bool(torch.isfinite(cpu.float()).all())}
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [tensor_identity(item) for item in value]
    if isinstance(value, dict):
        return {name: tensor_identity(item) for name, item in value.items()}
    raise ValueError('Unexpected encoder sub-output')


def measure_part(name, function, *args, **kwargs):
    if name == 'text':
        CALL_OBJECTS['text_inputs'] = {'inputs_embeds': args[0], 'positions': args[1], **kwargs}
    elif name == 'embedding':
        CALL_OBJECTS['embedding_ids'] = args[0]
    elif name == 'vision':
        CALL_OBJECTS['vision_inputs'] = {'pixels': args[0], 'grid': args[1]}
    # Retain live tensors only. CPU transfer/hash is after the COMPLETE timer;
    # these extra references may raise HBM and are reported as diagnostic overhead.
    if not PARTS_ACTIVE:
        output = function(*args, **kwargs)
        CALL_OBJECTS[name + '_output'] = output
        return output
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    start.record()
    t = time.perf_counter()
    output = function(*args, **kwargs)
    end.record()
    torch.npu.synchronize()
    seconds = time.perf_counter() - t
    CALL_OBJECTS[name + '_output'] = output
    PARTS.append({'name': name, 'synchronized_wall_s': seconds, 'current_stream_elapsed_ms': start.elapsed_time(end)})
    return output


try:
    init_distributed_environment(world_size=size, rank=rank, local_rank=rank, backend='hccl')
    initialize_model_parallel(sequence_parallel_size=size, ulysses_degree=size, tensor_parallel_size=1, backend='hccl')
    group = GroupCoordinator([list(range(size))], rank, 'hccl')
    encoder = candidate.MiniMaxH3Qwen3VLEncoder(str(MODEL), device=torch.device('npu', rank), load_model=True, encoder_group=group)
    encoder.eval()
    model = encoder.text_model
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    wanted = set(dict(encoder.named_parameters()))
    def tensors():
        for shard in sorted(set(index.values())):
            keys = [key for key, name in index.items() if name == shard and encoder._map_weight_name(key) is not None]
            if not keys:
                continue
            with safe_open(str(MODEL / shard), framework='pt', device='cpu') as f:
                for key in keys:
                    yield key, f.get_tensor(key)
    t = time.perf_counter()
    loaded = encoder.load_weights(tensors())
    r['loaded_parameter_tensors'], r['cpu_load_s'] = len(loaded), time.perf_counter() - t
    assert loaded == wanted
    assert all(parameter.dtype == torch.bfloat16 for parameter in encoder.parameters())
    encoder.to(torch.device('npu', rank))
    r['bf16_complete_encoder_parameter_bytes'] = sum(p.numel() * p.element_size() for p in encoder.parameters())
    r['parameter_dtypes'] = dict(Counter(str(p.dtype) for p in encoder.parameters()))
    r['component_parameter_bytes'] = {'vision': sum(p.numel() * p.element_size() for p in encoder.vision.parameters()),
                                      'embedding': encoder.text_model.embed_tokens.weight.numel() * encoder.text_model.embed_tokens.weight.element_size(),
                                      'text_all_including_embedding': sum(p.numel() * p.element_size() for p in model.parameters())}
    r['checkpoint_root'] = str(MODEL)
    r['checkpoint_index_sha256'] = hashlib.sha256((MODEL / 'model.safetensors.index.json').read_bytes()).hexdigest()
    for layer, block in enumerate(model.layers):
        qkv = block.self_attn.qkv_proj
        width = qkv.local_num_heads * qkv.head_dim
        SLOTS[lookup(qkv.weight[:width])] = ('q', layer)
        for name, module in (('gate_up', block.mlp.gate_up_proj), ('o', block.self_attn.o_proj), ('down', block.mlp.down_proj)):
            SLOTS[lookup(module.weight)] = (name, layer)
        for name, module in (('hidden_input', block.input_layernorm), ('hidden_residual', block.post_attention_layernorm),
                             ('q_head', block.self_attn.q_norm), ('k_head', block.self_attn.k_norm)):
            module._norm68_name = '%02d.%s' % (layer, name)
    assert len(SLOTS) == 200
    vision_modes = VisionLinearModes(metric)
    r['actual_vision_linear_modules'] = vision_modes.install(encoder.vision, candidate._NpuBF16CompatibleLinear)
    r['math_flags_before'] = {'matmul_allow_hf32': torch.npu.matmul.allow_hf32, 'conv_allow_hf32': torch.npu.conv.allow_hf32}
    r['parameters_before'] = parameters_sha(encoder)
    capture = Path('/tmp/h3_round54_capture01/encoder_input.pt')
    original = torch.load(capture, weights_only=True)
    assert set(original) == {'input_ids', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'}
    assert original['pixel_values_videos'] is None and original['video_grid_thw'] is None
    assert original['input_ids'].shape == (2043,) and original['image_grid_thw'].tolist() == [[1, 48, 84], [1, 48, 84]]
    assert original['pixel_values'].shape == (8064, 1536) and original['pixel_values'].dtype == torch.float16
    inputs = {name: value.cpu().contiguous() for name, value in original.items() if value is not None}
    r['capture_path'], r['capture_file_sha256'] = str(capture), hashlib.sha256(capture.read_bytes()).hexdigest()
    r['actual_inputs_before'] = tensor_identity(inputs)
    r['input_transport_note'] = 'Original captured pixels are already FP16 after production input distribution; _encode casts them to BF16 as before. No fresh FP32 processor input is substituted.'
    originals = {'embedding': encoder.text_model.embed_tokens.forward, 'vision': encoder.vision.forward,
                 'text': encoder.text_model.forward}
    encoder.text_model.embed_tokens.forward = lambda *args, **kwargs: measure_part('embedding', originals['embedding'], *args, **kwargs)
    encoder.vision.forward = lambda *args, **kwargs: measure_part('vision', originals['vision'], *args, **kwargs)
    encoder.text_model.forward = lambda *args, **kwargs: measure_part('text', originals['text'], *args, **kwargs)
    def once_complete():
        CALL_OBJECTS.clear()
        torch.distributed.barrier()
        torch.npu.synchronize()
        try:
            t = time.perf_counter()
            output = encoder.encode_ids(**inputs)
            torch.npu.synchronize()
            seconds = time.perf_counter() - t
            # encode_ids' own hidden.cpu() is INCLUDED. Only proof hashing and
            # extra sub-output copies are outside the complete timed interval.
            identities = tensor_identity(CALL_OBJECTS)
            identities['complete_output'] = tensor_identity(output)
            return output.detach().cpu().contiguous(), seconds, identities
        finally:
            CALL_OBJECTS.clear()
    baseline = None
    MODE, VERIFY_NORM, PREFIX_VERIFY = 'shared_qkv', False, False
    r['scope'] = 'Complete resident encode_ids on actual2043-token and two48x84-image processor tensors: real vocab embedding, vision, positions, deepstack,50-layer text, finalCPUoutput. Excludes upstreamHTTPprocessor/tokenization/groupbroadcast/load/offload and fullpipeline; BF16 storage, fixedexperimentalFP32vision/text math, not officiallossless.'
    r['environment'] = {name: value for name, value in os.environ.items() if name.startswith('VLLM_OMNI_910A_')}
    # All five vision variants keep the proven fast text kernel/reference stack
    # fixed. These are vision factors, not another text-math mode switch.
    NORM_TAIL, STATS_MODE, CAST_MODE = True, 'add_square', 'both'
    VERIFY_NORM70 = VERIFY_CAST = VERIFY = False
    original_vision_rope=candidate._apply_rotary_pos_emb_vision
    original_vision_attention73=candidate._vision_sdpa_bf16_compatible
    cast_attention=CastAttention(original_vision_attention73,metric,tensor_identity)
    cast_attention.enabled=True;cast_attention.verify=False
    candidate._apply_rotary_pos_emb_vision=rope_pair
    candidate._vision_sdpa_bf16_compatible=cast_attention.forward
    captured=[];handles=[]
    def capture_layer(index):
        def hook(module,args,output):
            if rank!=0:return
            path=ROOT/f'text_layer{index:02d}.pt'
            payload=output.detach().cpu().contiguous().clone()
            with path.open('xb')as f:torch.save(payload,f)
            row={'layer':index,'shape':list(payload.shape),'dtype':str(payload.dtype),'bytes':payload.numel()*payload.element_size(),
                 'sha256':raw_hash(payload),'file_bytes':path.stat().st_size,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'finite':bool(torch.isfinite(payload.float()).all()),'boundary':'Raw text block output BEFORE model-loop DeepStack injection at layers0/1/2; later layers unaffected.'}
            captured.append(row)
            assert row['file_bytes']<32*1024*1024 and sum(item['file_bytes']for item in captured)<256*1024*1024
            with path.with_suffix('.json').open('x')as f:json.dump(row,f,indent=2)
        return hook
    for index in(0,1,2,9,24,49):handles.append(encoder.text_model.layers[index].register_forward_hook(capture_layer(index)))
    try:
        vision_modes.begin_mode('both',verify=False);PARTS_ACTIVE=False
        output,excluded_s,identities=once_complete()
        reference=torch.load('/tmp/h3_round70_components01/fresh_encoder_reference.pt',map_location='cpu',weights_only=True)['complete_output']
        r['full_output_vs_round70']=metric(output,reference);assert r['full_output_vs_round70']['raw_bits_equal']
        r['actual_identities']=identities;r['captured_layers']=captured;r['capture_wall_s_excluded_IO']=excluded_s
        if rank==0:assert len(captured)==6
        r['vision_shadow_cache']=vision_modes.shadow.stats()
    finally:
        for handle in handles:handle.remove()
        candidate._apply_rotary_pos_emb_vision=original_vision_rope
        candidate._vision_sdpa_bf16_compatible=original_vision_attention73
        r['vision_rope_restored']=candidate._apply_rotary_pos_emb_vision is original_vision_rope
        r['attention_restored']=candidate._vision_sdpa_bf16_compatible is original_vision_attention73
    r['math_flags_after'] = {'matmul_allow_hf32': torch.npu.matmul.allow_hf32, 'conv_allow_hf32': torch.npu.conv.allow_hf32}
    assert r['math_flags_before'] == r['math_flags_after']
    r['parameters_after'] = parameters_sha(encoder)
    r['actual_inputs_after'] = tensor_identity(inputs)
    assert r['parameters_before'] == r['parameters_after'] and r['actual_inputs_before'] == r['actual_inputs_after']
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    r['checks_at_failure']={'attention':cast_attention.checks.copy() if 'cast_attention'in globals()else [],'parts':PARTS.copy()}
    raise
finally:
    if vision_modes is not None:
        vision_modes.restore()
        r['vision_wrappers_restored'] = not vision_modes.bound
        r['vision_shadow_bytes_after_clear'] = vision_modes.shadow.stats()['cached_fp32_bytes']
    weights.clear()
    if torch.distributed.is_initialized():
        destroy_model_parallel()
        torch.distributed.destroy_process_group()
    writer.write(r)
    print('Evidence:', ROOT / ('capture_rank%d.json' % rank), flush=True)
