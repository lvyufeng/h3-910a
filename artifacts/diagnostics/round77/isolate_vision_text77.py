#!/usr/bin/env python3
"""Paired independent-HF vision/DeepStack versus fixed experimental text isolation.
This reuses cached upstream tensors ONLY to identify hidden-state discrepancy.
It is not full-encoder performance,live pipeline latency,video-quality scoring,
or nativeBF16 compute. Original checkpoint weights/text math remain fixed.
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
from evidence_io76 import EvidenceWriter
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

ROOT = Path('/tmp/h3_round77_isolate01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
writer = EvidenceWriter(ROOT / ('isolation_rank%d.json' % rank))
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
                 for name in ('isolate_vision_text77.py', 'candidate_padded_text_encoder.py', 'evidence_io76.py', 'fixture_hashes77.json')}}


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


FIXTURE_HASHES77=json.loads((ROOT/'fixture_hashes77.json').read_text())
r['fixture_hashes_checked']={}

def checked_load77(path):
    path=Path(path);expected=FIXTURE_HASHES77[str(path)]
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==expected,'Independent actual fixture changed:'+str(path)
    r['fixture_hashes_checked'][str(path)]=actual
    return torch.load(path,map_location='cpu',weights_only=True)


def modal77(a,b,visual):
    a=a[None]if a.ndim==2 else a;b=b[None]if b.ndim==2 else b
    assert a.shape==b.shape==(1,2043,5120)and visual.shape==(1,2043)
    return {'full':metric(a,b),'text_positions':metric(a[~visual],b[~visual]),'visual_positions':metric(a[visual],b[visual])}


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
    original = checked_load77(capture)
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
    try:
        vision_modes.begin_mode('both',verify=False);PARTS_ACTIVE=False
        expref=checked_load77('/tmp/h3_round70_components01/fresh_encoder_reference.pt')
        hf=checked_load77('/tmp/h3_round74_hf01/hf_multimodal_inputs.pt')
        hf_final=checked_load77('/tmp/h3_round74_hf01/hf_selected50.pt')['pre_final_norm']
        exp=expref['text_inputs'];reference=expref['complete_output']
        assert set(hf)=={'inputs_embeds','position_ids','visual_pos_masks','deepstack_visual_embeds'}
        assert set(exp)=={'inputs_embeds','positions','visual_pos_masks','deepstack_visual_embeds'}
        assert hf['inputs_embeds'].shape==exp['inputs_embeds'].shape==(1,2043,5120)
        assert hf['inputs_embeds'].dtype==exp['inputs_embeds'].dtype==torch.bfloat16
        assert hf['position_ids'].shape==exp['positions'].shape==(3,1,2043)
        assert hf['position_ids'].dtype==exp['positions'].dtype==torch.int64
        assert hf['visual_pos_masks'].shape==exp['visual_pos_masks'].shape==(1,2043)
        assert hf['visual_pos_masks'].dtype==exp['visual_pos_masks'].dtype==torch.bool
        assert torch.equal(hf['position_ids'],exp['positions'])and torch.equal(hf['visual_pos_masks'],exp['visual_pos_masks'])
        assert all(isinstance(p['deepstack_visual_embeds'],list)and len(p['deepstack_visual_embeds'])==3 for p in(hf,exp))
        assert all(t.shape==(2016,5120)and t.dtype==torch.bfloat16 for p in(hf,exp)for t in p['deepstack_visual_embeds'])
        visual=hf['visual_pos_masks'];assert int(visual.sum())==2016
        assert torch.equal(exp['inputs_embeds'][~visual],hf['inputs_embeds'][~visual])
        r['HF_upstream_input_identity']=tensor_identity(hf);r['experimental_upstream_input_identity']=tensor_identity(exp)
        r['HF_reference_pre_final_norm_identity']=tensor_identity(hf_final)
        first,excluded_s,first_ids=once_complete()
        r['fresh_complete_control_vs_fixed_experiment']=metric(first,reference)
        assert r['fresh_complete_control_vs_fixed_experiment']['raw_bits_equal']and raw_hash(first)=='67ef196e33bcaa495420da1ebc07d1ff6dcaf89ddbaf75a350940ed8ae210eee'
        assert first_ids['text_inputs']==tensor_identity(exp)
        r['fresh_complete_control_identities']=first_ids
        r['fresh_control_excluded_compilation_s']=excluded_s
        r['baseline_complete_vs_HF']=modal77(first,hf_final,visual)
        selected=(0,1,2,9,24,49)
        hf_layers={i:checked_load77('/tmp/h3_round74_repeat01/hf_layer%02d.pt'%i)for i in selected}if rank==0 else{}
        cases=[('experimental_inputs',False,False),('HF_merged_only',True,False),('HF_deepstack_only',False,True),('HF_all_vision',True,True)]
        r['case_order']=[name for name,_,_ in cases]
        r['scope']='Factorial upstream replacement on fixed experimental50-layer text graph. Cachedinput diagnostic ONLY;not freshfullencoder latency,not nativeBF16 compute,not videoquality,not optimization-degradation attribution.'
        for name,use_merged,use_deep in cases:
            cpuargs={'inputs_embeds':hf['inputs_embeds']if use_merged else exp['inputs_embeds'],
                     'positions':hf['position_ids'],'visual_pos_masks':hf['visual_pos_masks'],
                     'deepstack_visual_embeds':hf['deepstack_visual_embeds']if use_deep else exp['deepstack_visual_embeds']}
            cpu_before=tensor_identity(cpuargs)
            device=torch.device('npu',rank)
            args={'inputs_embeds':cpuargs['inputs_embeds'].to(device),'positions':cpuargs['positions'].to(device),
                  'visual_pos_masks':cpuargs['visual_pos_masks'].to(device),
                  'deepstack_visual_embeds':[value.to(device)for value in cpuargs['deepstack_visual_embeds']]}
            assert tensor_identity(args)==cpu_before
            def direct_once77():
                CALL_OBJECTS.clear();torch.distributed.barrier();torch.npu.synchronize()
                with torch.inference_mode():out=model(args['inputs_embeds'],args['positions'],visual_pos_masks=args['visual_pos_masks'],deepstack_visual_embeds=args['deepstack_visual_embeds'])[0].to(torch.bfloat16)
                torch.npu.synchronize();result=out.detach().cpu().contiguous();CALL_OBJECTS.clear();return result
            captured={};handles=[]
            def capture77(index):
                def hook(module,arguments,out):
                    assert out.shape==(1,2043,5120)and out.dtype==torch.bfloat16
                    captured[index]=out.detach().cpu().contiguous()
                return hook
            if rank==0:
                for i in selected:handles.append(model.layers[i].register_forward_hook(capture77(i)))
            try:out=direct_once77()
            finally:
                for handle in handles:handle.remove()
            assert tensor_identity(args)==cpu_before and tensor_identity(cpuargs)==cpu_before
            row={'merged_source':'independent_HF'if use_merged else'fixed_experiment',
                 'deepstack_source':'independent_HF'if use_deep else'fixed_experiment',
                 'input_identity':cpu_before,'output_identity':tensor_identity(out),
                 'compared_to_independent_HF':modal77(out,hf_final,visual),
                 'compared_to_fixed_experiment':modal77(out,reference,visual),'repeat_checks':[]}
            if name=='experimental_inputs':assert metric(out,reference)['raw_bits_equal']
            for repeat in range(2):
                again=direct_once77();proof=metric(again,out)
                assert proof['raw_bits_equal']and proof['finite']
                row['repeat_checks'].append(proof)
            if rank==0:
                assert set(captured)==set(selected)
                row['raw_decoder_PRE_DeepStack_layers']={str(i):modal77(captured[i],hf_layers[i],visual)for i in selected}
                target=ROOT/(name+'.pt');payload={'complete_output':out,'raw_decoder_PRE_DeepStack_layers':captured}
                tensor_bytes=out.numel()*out.element_size()+sum(v.numel()*v.element_size()for v in captured.values())
                assert tensor_bytes<256*1024*1024
                with target.open('xb')as f:torch.save(payload,f)
                row['fixture']={'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            r['cases'][name]=row
            assert tensor_identity(args)==cpu_before and tensor_identity(cpuargs)==cpu_before
            del args,captured
            print(json.dumps({'rank':rank,'case':name,'vsHF':row['compared_to_independent_HF'],'not_performance':True}),flush=True)
        restored,_,restored_ids=once_complete()
        r['fresh_complete_restore_vs_fixed_experiment']=metric(restored,reference)
        assert r['fresh_complete_restore_vs_fixed_experiment']['raw_bits_equal']and restored_ids==first_ids
        r['fresh_inputs_readonly']=True;r['same_text_math_and_original_weights_for_all_cases']=True
        r['cached_features_not_claimed_full_encoder_performance']=True
        r['vision_shadow_cache']=vision_modes.shadow.stats()
    finally:
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
    r['text_shadow_bytes_after_clear']=weights.stats()['cached_fp32_bytes']
    assert r['text_shadow_bytes_after_clear']==0
    if torch.distributed.is_initialized():
        destroy_model_parallel()
        torch.distributed.destroy_process_group()
    writer.write(r)
    print('Evidence:', ROOT / ('isolation_rank%d.json' % rank), flush=True)
