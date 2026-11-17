#!/usr/bin/env python3
"""Round81 zero-padded KV TP8 FP16 diagnostic driver for the Qwen3-VL layer-50 encoder.

STATUS: PREPARED SOURCE ONLY.  This file has never been executed on the NPU by
its author.  Nothing here is a precision result, a speed result, or an
official-lossless claim.  The root agent deploys it and runs it later under its
own supervisor.

What this driver does
---------------------
* Imports the Round79 generated candidate ``candidate_encoder79.py``
  (SHA256 2141b29a...), which removes the three hardcoded BF16 I/O conversions
  and restores the original-HF FP16 rounding order for RMSNorm / text M-RoPE /
  vision position interpolation.  The candidate is byte-verified by path and
  registered in ``sys.modules``; production code is never modified.
* Runs the genuine 8-rank tensor-parallel graph: real BF16 checkpoint
  parameters are loaded through the candidate's own TP-aware ``load_weights``
  into uniform FP16 parameters, then the submodules move to ``npu:<rank>``.
* Replaces exactly one module-level symbol,
  ``candidate._scaled_dot_product_attention``, with an explicit causal
  FP32-math helper (FP16 Q/K/V in -> FP32 scores, FP32 softmax, FP32 P@V ->
  one FP16 out).  That mirrors the declared CPU reference baseline (PyTorch
  ``SDPBackend.MATH``) and deliberately carries no ``+/-50000`` clamp, which is
  a 910A-only construct with no CPU analogue.  It is a scoped assignment on an
  isolated diagnostic module, NOT a global ``torch.nn.functional`` monkeypatch,
  and nothing is installed into production.
* Compares every measurable boundary against the independently recomputed CPU
  FP16 reference (Round78): fresh vision + 3 DeepStack, vocabulary embedding,
  text-model inputs (inputs_embeds / positions / visual_pos_masks /
  deepstack_visual_embeds), all 50 raw pre-DeepStack decoder-layer boundary
  hashes, and the selected-50 pre-final-norm hidden state.
* Records 3 warmed full-forward timings per rank, each including the final
  device->host copy of the hidden state.  The root takes the rank maximum
  later; this driver never collapses ranks into a single number.
* Repeats one more full forward and requires bit-level equality with the first
  output, and requires parameters/inputs to hash unchanged.

Honesty constraints baked into the receipt
------------------------------------------
* ``full_encoder_precision_accepted`` is always False.  No tolerance is chosen
  here, before or after seeing the numbers; the numeric call belongs to the
  root with the CPU gold in hand.
* There is no ``numerical_pass`` field.  Only ``finite`` and
  ``repeat_verified`` are asserted, because those are structural rather than
  tolerance-based.
* The known tensor-parallel deviation is recorded as a warning, never hidden:
  a row-parallel projection rounds its FP16 partial GEMM output *before* the
  FP32 all-reduce, and that collective cannot restore precision already
  rounded away.  This driver measures the consequence; it does not fix it and
  must not be described as fixed.
* Timings are wall clock for this diagnostic graph.  They are NOT the
  production ``encode_ids`` path and NOT an official benchmark.

Entry-point shape contract
--------------------------
The vocabulary embedding preserves the leading batch dimension: IDs shaped
(1,seq) produce (1,seq,hidden). Text attention,MLP and residuals preserve that
shape,so the bare tensor returned by text_model is still batched. The [0] in
_encode removes the singleton batch correctly;it is NOT evidence of a defect.
The explicit diagnostic path retains the batch for boundary hooks. The actual
stock encode_ids is probed and used for warm timing if its runtime checks pass;
an observed exception is recorded without attributing it to an assumed shape bug.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/tmp/h3_round81_full_fp16_pad01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
CPU_ROOT = Path('/tmp/h3_round78_hf_fp1601')
FIXTURE = Path('/tmp/h3_round54_capture01/encoder_input.pt')

CANDIDATE_SHA256 = '2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78'
CPU_SOURCE_SHA256 = '50165e77adb393b4573fbe0fdb29c62a6ddb14b8f39082d0ea67e904848010ea'
CPU_PARAMETER_SHA256 = 'a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e'
FIXTURE_SHA256 = 'ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
CHECKPOINT_INDEX_SHA256 = '06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a'
CPU_CONFIG_SHA256 = 'd2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427'

# Round78 stage payload hashes: fresh CPU FP16 tensors from the independent
# recomputation, not old BF16 outputs that were merely cast down.
EXPECTED_CPU_FILES = {
    'FP16_vision.pt': 'c01bbe180f454fe44b182bd9307491ef0f4c342b6c2ff6be08d1e3453d8c312a',
    'FP16_multimodal_inputs.pt': '6ab37f7202731843aebce9f3f8453232c72d45b2239caba679bcb2611e626e70',
    'FP16_selected50.pt': '0137a434148366c3dcbf825970274ad0faa036666561d7aecc0170cf136b3fe4',
}
EXPECTED_CPU_LAYERS = {
    0: '6bbe6e4a02d196a40dc3aa975d0cc359d6238899122816f0079ef6c740d03d9b',
    1: '619d40ec8d3bc7b2dc093b57c49169bbd977d72b802a6dc4c4559a56d9c18047',
    2: '4b4e1d7378145789b54aadd63fc3131fd513d88868aba489253b0fd8a438fe9e',
    9: '3b999e53ed28d1735762a09d460f2485c3ebb7b98642d3699e913fffc285b217',
    24: '6d081a27aa99eeaf0bd73084a72d09c969f180788041d246cf0b1983e182b0e9',
    49: '98302d42b7ecbaa455d156a1bcbb14006ca3995b3267cb1a4368126f0822475e',
}

NUM_RANKS = 8
WARM_CALLS = 3
# Bounded-memory causal attention chunk along the query axis. Row-wise softmax
# is exactly decomposable so the math is unchanged; only GEMM tiling differs,
# which is documented rather than claimed bit-identical to a single-shot matmul.
SDPA_QUERY_CHUNK = 512


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_info(value) -> dict:
    import torch

    cpu = value.detach().cpu().contiguous()
    entry = {
        'shape': list(cpu.shape),
        'dtype': str(cpu.dtype),
        'bytes': cpu.numel() * cpu.element_size(),
        'sha256': hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest(),
        'finite': bool(torch.isfinite(cpu.float()).all()) if cpu.is_floating_point() else True,
    }
    if cpu.is_floating_point() and cpu.numel():
        entry['max_abs'] = float(cpu.float().abs().max())
    return entry


def load_candidate(receipt: dict):
    """Import the Round79 candidate by path, byte-verified, isolated module name."""
    path = ROOT / 'candidate_encoder79.py'
    actual = file_sha(path)
    if actual != CANDIDATE_SHA256:
        raise RuntimeError(f'R79 candidate SHA mismatch: {actual} != {CANDIDATE_SHA256}')
    if 'candidate_encoder79' in sys.modules:
        raise RuntimeError('candidate_encoder79 already imported; refusing an ambiguous graph')
    spec = importlib.util.spec_from_file_location('candidate_encoder79', path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so __all__/pickle/dataclass resolution sees the same
    # module object the rest of the process references.
    sys.modules['candidate_encoder79'] = module
    spec.loader.exec_module(module)
    if not getattr(module, 'R79_CANDIDATE', False):
        raise RuntimeError('loaded module is not marked as the R79 diagnostic candidate')
    receipt['candidate'] = {'path': str(path), 'sha256': actual,
                            'module_name': 'candidate_encoder79',
                            'origin_production_sha256': getattr(module, 'R79_ORIGIN_SHA256', None)}
    return module


def install_fp16_f32_math_attention(candidate, receipt: dict):
    """Scope-replace only the module-level text SDPA symbol (no global patch)."""
    import torch

    original = candidate._scaled_dot_product_attention

    def fp16_f32_math_causal_sdpa(query, key, value):
        groups = query.shape[1] // key.shape[1]
        if groups != 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        if not (query.dtype == key.dtype == value.dtype == torch.float16):
            raise RuntimeError('FP16 diagnostic graph reached attention with non-FP16 operands: '
                               f'{query.dtype}/{key.dtype}/{value.dtype}')
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        scale = query.shape[-1] ** -0.5
        qf, kf, vf = query.float(), key.float(), value.float()
        out = torch.empty(query.shape[:-1] + (value.shape[-1],), dtype=torch.float32,
                          device=query.device)
        rows = torch.arange(seq_q, device=query.device).unsqueeze(-1)
        cols = torch.arange(seq_k, device=query.device).unsqueeze(0)
        causal_mask = cols > rows
        for begin in range(0, seq_q, SDPA_QUERY_CHUNK):
            end = min(begin + SDPA_QUERY_CHUNK, seq_q)
            scores = torch.matmul(qf[..., begin:end, :], kf.transpose(-2, -1)) * scale
            scores = scores.masked_fill(causal_mask[begin:end], float('-inf'))
            out[..., begin:end, :] = torch.matmul(torch.softmax(scores, dim=-1), vf)
        return out.to(query.dtype)

    candidate._scaled_dot_product_attention = fp16_f32_math_causal_sdpa
    receipt['text_attention'] = {
        'scope': 'module-level candidate._scaled_dot_product_attention replacement',
        'global_functional_monkeypatch': False,
        'torch_nn_functional_replaced': False,
        'operands': 'FP16 Q/K/V in, FP32 scores, FP32 softmax, FP32 P@V, single FP16 out',
        'causal': True,
        'clamp_applied': False,
        'clamp_note': 'the 910A-only +/-50000 clamp of the production shim is deliberately absent; '
                      'the CPU reference baseline has no such clamp',
        'query_chunk': SDPA_QUERY_CHUNK,
        'query_chunk_note': 'row-wise softmax is exactly decomposable; GEMM tiling differs from a '
                            'single-shot FP32 matmul, so bit-equality with CPU is not asserted',
        'original_reference': {'module': getattr(original, '__module__', None),
                               'qualname': getattr(original, '__qualname__', None)},
    }
    return original


def load_cpu_gate(receipt: dict) -> dict:
    """Fail closed unless the independent CPU FP16 reference is complete and verified."""
    reference_path = CPU_ROOT / 'reference.json'
    if not reference_path.exists():
        raise RuntimeError(f'CPU FP16 reference receipt missing: {reference_path}')
    reference = json.loads(reference_path.read_text())
    gate = {
        'completed': reference.get('completed'),
        'full_multimodal_reference_completed': reference.get('full_multimodal_reference_completed'),
        'repeat_pre_final_norm_bits_equal': reference.get('repeat_pre_final_norm_bits_equal'),
        'repeat_post_final_norm_bits_equal': reference.get('repeat_post_final_norm_bits_equal'),
        'pre_final_norm_matches_layer49_identity': reference.get('pre_final_norm_matches_layer49_identity'),
    }
    receipt['cpu_reference_gate'] = gate
    if not all(value is True for value in gate.values()):
        raise RuntimeError('CPU FP16 reference is not complete and repeat-verified yet; refusing to '
                           'build an NPU precision comparison against an unfinished gold: '
                           + json.dumps(gate))
    if reference['source_sha256'] != CPU_SOURCE_SHA256:
        raise RuntimeError(f'CPU reference source SHA mismatch: {reference["source_sha256"]}')
    if reference['original_selected_parameter_sha256'] != CPU_PARAMETER_SHA256:
        raise RuntimeError('CPU reference original BF16 parameter SHA mismatch; the gold was not '
                           'built from the pinned official weights')
    for stage, expected in EXPECTED_CPU_FILES.items():
        staged = reference['stages'].get(Path(stage).stem)
        if staged is None or staged['file_sha256'] != expected:
            raise RuntimeError(f'CPU reference stage {stage} hash changed: {staged}')
    for layer, expected in EXPECTED_CPU_LAYERS.items():
        staged = reference['stages'].get(f'FP16_layer{layer:02d}')
        if staged is None or staged['file_sha256'] != expected:
            raise RuntimeError(f'CPU reference layer {layer} payload hash changed: {staged}')
    if len(reference['language_layers']) != 50 or len(reference['vision_layers']) != 27:
        raise RuntimeError('CPU reference boundary inventory is not 27 vision + 50 text layers')
    for name, expected in EXPECTED_CPU_FILES.items():
        if file_sha(CPU_ROOT / name) != expected:
            raise RuntimeError(f'on-disk CPU reference payload changed: {name}')
    receipt['cpu_reference'] = {
        'root': str(CPU_ROOT),
        'source_sha256': CPU_SOURCE_SHA256,
        'original_selected_parameter_sha256': CPU_PARAMETER_SHA256,
        'boundary': 'original HF selected-50 decoder layers, PRE final norm',
        'vision_layers': len(reference['vision_layers']),
        'language_layers': len(reference['language_layers']),
        'isolation': 'recomputed original HF FP16 graph; old BF16 outputs were never cast to FP16',
    }
    return reference


def compare(value, reference) -> dict:
    """Compare two tensors: report bits and error magnitudes, choose no threshold."""
    import torch

    a = value.detach().cpu().contiguous()
    b = reference.detach().cpu().contiguous()
    if a.shape != b.shape and a.numel() == b.numel():
        b = b.reshape(a.shape)
    result = {'npu': tensor_info(a), 'cpu': tensor_info(b)}
    if a.shape != b.shape:
        result.update({'shape_mismatch': True, 'comparable': False, 'raw_bits_equal': False})
        return result
    finite = bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all())
    result.update({'shape_mismatch': False, 'comparable': True, 'finite': finite,
                   'dtype_matches': a.dtype == b.dtype})
    if not finite or a.dtype != b.dtype:
        result['raw_bits_equal'] = False
        return result
    delta = a.double() - b.double()
    denominator = b.double().square().mean().sqrt().clamp_min(1e-30)
    result.update({'raw_bits_equal': bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8))),
                   'elements': a.numel(),
                   'max_abs_error': float(delta.abs().max()),
                   'RMSE': float(delta.square().mean().sqrt()),
                   'relative_rms': float(delta.square().mean().sqrt() / denominator)})
    return result


def parameters_sha(encoder) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in encoder.named_parameters():
        digest.update(name.encode())
        digest.update(memoryview(parameter.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return digest.hexdigest()


def inputs_sha(payload) -> str:
    return tensor_info(payload['input_ids'])['sha256'] + tensor_info(payload['pixel_values'])['sha256']


def main() -> None:
    import torch
    import torch_npu  # Explicit backend registration before torch.npu APIs.

    rank = int(os.environ['RANK'])
    size = int(os.environ['WORLD_SIZE'])
    if size != NUM_RANKS:
        raise RuntimeError(f'this diagnostic requires exactly {NUM_RANKS} ranks, got {size}')
    plan_only = os.environ.get('H3_R81_PLAN_ONLY', '0') == '1'

    from evidence_io81 import EvidenceWriter  # staged next to this file by the root

    writer = EvidenceWriter(ROOT / (f'plan_rank{rank}.json' if plan_only else f'tp8_rank{rank}.json'))
    r = {
        'scope': __doc__,
        'source_sha256': file_sha(Path(__file__)),
        'rank': rank,
        'world_size': size,
        'text_encoder_tp_size': size,
        'plan_only_no_model_run': plan_only,
        'prepared_source_never_executed_by_author': True,
        'executed_by_root': True,
        'numerical_threshold_chosen': None,
        'numerical_pass_field_absent': True,
        'full_encoder_precision_accepted': False,
        'whole_goal_complete': False,
        'timing_scope': 'explicit _encode-equivalent full forward including final device->host copy',
        'stock_encode_ids_timed': False,
        'warnings': [],
        'cases': {},
        'language_layers': [],
    }
    handles = []
    cleanup = {}
    dist_ready = False
    kv_adapter = None

    def warn(text):
        r['warnings'].append(text)

    try:
        torch.set_num_threads(2)
        torch.npu.set_device(rank)
        r['device'] = f'npu:{rank}'
        r['environment'] = {name: os.environ.get(name) for name in (
            'VLLM_OMNI_910A_TE_DTYPE', 'VLLM_OMNI_910A_TEXT_FP32_LINEAR',
            'VLLM_OMNI_910A_TEXT_PAD_KV', 'VLLM_OMNI_910A_VISION_FP32')}
        if (os.environ.get('VLLM_OMNI_910A_TE_DTYPE', '') or '').lower() != 'half':
            raise RuntimeError('VLLM_OMNI_910A_TE_DTYPE must be half; refusing an unlabelled dtype run')
        for forbidden in ('VLLM_OMNI_910A_TEXT_FP32_LINEAR', 'VLLM_OMNI_910A_TEXT_PAD_KV'):
            if os.environ.get(forbidden, '0') == '1':
                raise RuntimeError(f'{forbidden}=1 would reintroduce the removed R65 shadow/padded '
                                   'helpers into a FP16 diagnostic graph; unused here by contract')

        candidate = load_candidate(r)
        install_fp16_f32_math_attention(candidate, r)
        from kv_padding81 import install
        kv_adapter = install(candidate)
        r['kv_padding_before'] = kv_adapter.snapshot()
        reference = load_cpu_gate(r)

        if file_sha(FIXTURE) != FIXTURE_SHA256:
            raise RuntimeError('encoder fixture changed')
        if file_sha(MODEL / 'model.safetensors.index.json') != CHECKPOINT_INDEX_SHA256:
            raise RuntimeError('checkpoint index changed')
        if file_sha(MODEL / 'config.json') != CPU_CONFIG_SHA256:
            raise RuntimeError('checkpoint config changed')
        payload = torch.load(FIXTURE, map_location='cpu', weights_only=True)
        if tuple(payload['input_ids'].shape) != (2043,):
            raise RuntimeError(f'unexpected token count {tuple(payload["input_ids"].shape)}')
        if (tuple(payload['pixel_values'].shape) != (8064, 1536)
                or payload['pixel_values'].dtype != torch.float16):
            raise RuntimeError('unexpected pixel fixture')
        if payload['image_grid_thw'].tolist() != [[1, 48, 84], [1, 48, 84]]:
            raise RuntimeError('unexpected image grid')
        if payload['pixel_values_videos'] is not None or payload['video_grid_thw'] is not None:
            raise RuntimeError('fixture unexpectedly carries video inputs')
        r['fixture'] = {'path': str(FIXTURE), 'sha256': FIXTURE_SHA256,
                        'input_ids': tensor_info(payload['input_ids']),
                        'pixel_values': tensor_info(payload['pixel_values']),
                        'image_grid_thw': payload['image_grid_thw'].tolist()}

        if plan_only:
            r['plan_only_checks_passed'] = True
            return

        from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
        from vllm_omni.diffusion.distributed.parallel_state import (
            destroy_model_parallel, init_distributed_environment, initialize_model_parallel)
        from safetensors import safe_open

        init_distributed_environment(world_size=size, rank=rank, local_rank=rank, backend='hccl')
        dist_ready = True
        initialize_model_parallel(sequence_parallel_size=size, ulysses_degree=size,
                                  tensor_parallel_size=1, backend='hccl')
        group = GroupCoordinator([list(range(size))], rank, 'hccl')
        r['group'] = {'world_size': int(group.world_size), 'backend': 'hccl'}

        encoder = candidate.MiniMaxH3Qwen3VLEncoder(
            str(MODEL), device=torch.device('npu', rank), load_model=True, encoder_group=group)
        encoder.eval()
        index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
        wanted = set(dict(encoder.named_parameters()))

        def tensors():
            for shard in sorted(set(index.values())):
                keys = [key for key, name in index.items() if name == shard
                        and encoder._map_weight_name(key) is not None]
                if not keys:
                    continue
                with safe_open(str(MODEL / shard), framework='pt', device='cpu') as handle:
                    for key in keys:
                        yield key, handle.get_tensor(key)

        started = time.perf_counter()
        loaded = encoder.load_weights(tensors())
        r['weight_load'] = {
            'loaded_parameter_tensors': len(loaded),
            'cpu_load_s': time.perf_counter() - started,
            'source_checkpoint_dtype': 'torch.bfloat16',
            'conversion': 'deterministic BF16 storage -> FP16 parameter via copy_ in load_weights',
        }
        if loaded != wanted:
            raise RuntimeError('weight loading did not cover every parameter')
        if len(encoder.text_model.layers) != 50:
            raise RuntimeError('text encoder is not the selected 50 layers')
        if any(name == 'norm' for name, _ in encoder.text_model.named_children()):
            raise RuntimeError('text model unexpectedly has a final norm; the boundary would not be '
                               'the H3 pre-final-norm selected-50 output')
        if candidate.MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER != 50:
            raise RuntimeError('selected-layer constant changed')
        r['boundary'] = {
            'selected_lm_layer_constant': candidate.MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
            'retained_decoder_layers': len(encoder.text_model.layers),
            'text_model_has_final_norm': False,
            'cpu_reference_boundary': 'original HF decoder layers 0..49, PRE final norm',
            'matches_cpu_reference_boundary': True,
        }

        r['self_check'] = encoder.r79_self_check()
        if r['self_check']['parameter_dtype_counts'] != {'torch.float16': len(wanted)}:
            raise RuntimeError('encoder parameters are not uniformly FP16')

        # Replicated vision tower: per-rank converted parameter hashes must match
        # the CPU reference conversion exactly. TP shards of text parameters are
        # not comparable to full-tensor CPU hashes and are covered by the frozen
        # whole-model parameter hash instead.
        conversion = reference['parameter_conversion']
        vision_params = dict(encoder.vision.named_parameters())
        matched, mismatched, missing = [], [], []
        for name, parameter in vision_params.items():
            expected = conversion.get('visual.' + name)
            if expected is None:
                missing.append(name)
                continue
            actual = hashlib.sha256(memoryview(
                parameter.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
            (matched if actual == expected['reference_sha256'] else mismatched).append(name)
        r['vision_parameter_provenance'] = {
            'checked': len(vision_params), 'matched': len(matched),
            'mismatched': mismatched, 'missing_cpu_reference': missing,
            'fully_verified': not mismatched and not missing,
        }
        if mismatched:
            raise RuntimeError(f'vision parameter conversion mismatch on {len(mismatched)} tensors: '
                               f'{mismatched[:5]}')
        if missing:
            warn(f'{len(missing)} vision parameters had no CPU reference conversion entry')

        encoder.to(torch.device('npu', rank))
        if next(encoder.parameters()).device.type != 'npu':
            raise RuntimeError('encoder is not resident on the NPU')
        r['self_check_after_move'] = encoder.r79_self_check()
        kv_adapter.prime(encoder.text_model)
        r['kv_padding_primed'] = kv_adapter.snapshot()
        parameter_sha_before = parameters_sha(encoder)
        input_sha_before = inputs_sha(payload)
        r['frozen_inputs_before'] = {'parameters_sha256': parameter_sha_before,
                                     'input_pair_sha256': input_sha_before}

        vision_reference = torch.load(CPU_ROOT / 'FP16_vision.pt', map_location='cpu', weights_only=True)
        inputs_reference = torch.load(CPU_ROOT / 'FP16_multimodal_inputs.pt', map_location='cpu',
                                      weights_only=True)
        selected_reference = torch.load(CPU_ROOT / 'FP16_selected50.pt', map_location='cpu',
                                        weights_only=True)
        layer_references = {layer: torch.load(CPU_ROOT / f'FP16_layer{layer:02d}.pt',
                                              map_location='cpu', weights_only=True)
                            for layer in EXPECTED_CPU_LAYERS}

        captured: dict = {}

        def run_equivalent(collect: bool):
            """Documented _encode computation retaining its singleton batch for hooks."""
            device = torch.device('npu', rank)
            ids = payload['input_ids'].to(device, torch.long)[None]
            attention_mask = torch.ones_like(ids)
            mm_types = torch.zeros_like(ids, dtype=torch.int32)
            mm_types.masked_fill_(ids == encoder.image_token_id, 1)
            mm_types.masked_fill_(ids == encoder.video_token_id, 2)
            grid = payload['image_grid_thw'].to(device, torch.long)
            inputs_embeds = encoder.text_model.embed_tokens(ids)
            if collect:
                captured['embedding'] = inputs_embeds.detach().clone()
            image_embeds, deepstack = encoder.vision(
                payload['pixel_values'].to(device, encoder._r79_io_dtype()), grid)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            if collect:
                captured['vision'] = {'merged': image_embeds.detach().clone(),
                                      'deepstack': [item.detach().clone() for item in deepstack]}
            image_mask, _ = encoder._get_placeholder_mask(
                ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks, deepstack_visual_embeds = encoder._aggregate_deepstack(
                image_mask, None, deepstack, None)
            positions, _ = candidate._get_rope_index(
                ids, mm_token_type_ids=mm_types, image_grid_thw=grid, video_grid_thw=None,
                attention_mask=attention_mask,
                spatial_merge_size=int(encoder.vision.config.spatial_merge_size))
            if collect:
                captured['text_inputs'] = {
                    'inputs_embeds': inputs_embeds.detach().clone(),
                    'position_ids': positions.detach().clone(),
                    'visual_pos_masks': visual_pos_masks.detach().clone(),
                    'deepstack_visual_embeds': [i.detach().clone() for i in deepstack_visual_embeds],
                }
            # Keep the batch for hooks;stock _encode subsequently removes it with [0].
            hidden = encoder.text_model(
                inputs_embeds, positions,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds)
            return hidden.cpu()

        def layer_post(index):
            def hook(module, args, output):
                captured.setdefault('layers', {})[index] = output.detach().clone()
            return hook

        # ---- 1. excluded diagnostic call with proof hooks ----
        for index, layer in enumerate(encoder.text_model.layers):
            handles.append(layer.register_forward_hook(layer_post(index)))
        torch.npu.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            first_hidden = run_equivalent(collect=True)
        torch.npu.synchronize()
        r['cases']['excluded_diagnostic_first_s'] = time.perf_counter() - started
        for handle in handles:
            handle.remove()
        handles.clear()
        if len(captured.get('layers', {})) != 50:
            raise RuntimeError(f'captured {len(captured.get("layers", {}))} of 50 decoder layers')

        # ---- 2. compare every boundary against the CPU FP16 gold ----
        cmp = {
            'vision_merged': compare(captured['vision']['merged'], vision_reference['merged']),
            'vision_deepstack': [compare(a, b) for a, b in
                                 zip(captured['vision']['deepstack'], vision_reference['deepstack'])],
            'text_inputs_embeds': compare(captured['text_inputs']['inputs_embeds'],
                                          inputs_reference['inputs_embeds']),
            'position_ids': compare(captured['text_inputs']['position_ids'],
                                    inputs_reference['position_ids']),
            'visual_pos_masks': compare(captured['text_inputs']['visual_pos_masks'].reshape(-1).to(torch.int8),
                                        inputs_reference['visual_pos_masks'].reshape(-1).to(torch.int8)),
            'deepstack_visual_embeds': [
                compare(a, b) for a, b in zip(captured['text_inputs']['deepstack_visual_embeds'],
                                              inputs_reference['deepstack_visual_embeds'])],
        }
        embedding_gold = reference.get('embedding_output')
        embedding_info = tensor_info(captured['embedding'])
        cmp['embedding'] = {
            'npu': embedding_info,
            'cpu_reference_sha256': None if embedding_gold is None else embedding_gold['sha256'],
            'cpu_reference_info': embedding_gold,
            'raw_bits_equal': (None if embedding_gold is None
                               else embedding_info['sha256'] == embedding_gold['sha256']),
        }
        if embedding_gold is None:
            raise RuntimeError('CPU reference has no embedding_output hash')
        # Gather the original checkpoint rows on CPU before converting them.
        # Elementwise BF16->FP16 cast commutes with row lookup;prove its hash
        # against the independently computed CPU FP16 embedding,not an old output.
        embedding_key = 'model.language_model.embed_tokens.weight'
        with safe_open(str(MODEL / index[embedding_key]), framework='pt', device='cpu') as f:
            original_embedding = f.get_tensor(embedding_key)
            cpu_embedding = torch.nn.functional.embedding(payload['input_ids'][None], original_embedding).to(torch.float16)
        if tensor_info(cpu_embedding)['sha256'] != embedding_gold['sha256']:
            raise RuntimeError('Reconstructed CPU FP16 embedding does not match the original reference')
        actual_embedding = captured['embedding'].detach().cpu().contiguous()
        cmp['embedding_full_numeric'] = compare(actual_embedding, cpu_embedding)
        cmp['embedding_sign_zero_differences'] = int(((actual_embedding == 0) & (cpu_embedding == 0) &
                                                      (actual_embedding.view(torch.int16) != cpu_embedding.view(torch.int16))).sum())
        del original_embedding, cpu_embedding

        cpu_layer_hashes = {row['layer']: row['output']['sha256'] for row in reference['language_layers']}
        for index in range(50):
            npu_info = tensor_info(captured['layers'][index])
            row = {'layer': index, 'npu': npu_info,
                   'cpu_reference_sha256': cpu_layer_hashes.get(index),
                   'raw_bits_equal': npu_info['sha256'] == cpu_layer_hashes.get(index)}
            if index in layer_references:
                row['full_compare'] = compare(captured['layers'][index], layer_references[index])
            r['language_layers'].append(row)

        gold_pre_norm = selected_reference['pre_final_norm']
        cmp['selected50_pre_final_norm'] = compare(first_hidden, gold_pre_norm)
        flat_hidden = first_hidden.reshape(-1, 5120)
        flat_gold = gold_pre_norm.reshape(-1, 5120)
        visual_rows = inputs_reference['visual_pos_masks'].reshape(-1)
        cmp['selected50_token_subsets'] = {}
        diagnostic_high_range = flat_gold.abs().amax(dim=-1) >= 512
        for subset, select in (('text_nonvisual', ~visual_rows), ('visual', visual_rows),
                               ('reference_abs_ge_512', diagnostic_high_range),
                               ('reference_abs_lt_512', ~diagnostic_high_range)):
            if bool(select.any()):
                cmp['selected50_token_subsets'][subset] = compare(flat_hidden[select], flat_gold[select])
        cmp['range_subset_is_diagnostic_not_an_acceptance_threshold'] = True
        cmp['selected50_boundary_declaration'] = {
            'cpu_key_compared': 'pre_final_norm',
            'cpu_post_final_norm_present_but_not_our_boundary': 'HF_post_final_norm' in selected_reference,
            'note': 'the H3 encoder consumes the unnormalized output of decoder layer 49; the CPU '
                    'gold stores that boundary as pre_final_norm',
        }
        cmp['language_layers_bit_equal_count'] = sum(1 for row in r['language_layers']
                                                     if row['raw_bits_equal'])
        cmp['npu_hidden_finite'] = bool(tensor_info(first_hidden)['finite'])
        if not cmp['npu_hidden_finite']:
            raise RuntimeError('NPU FP16 encoder produced a non-finite hidden state')
        r['cases']['cpu_fp16_comparison'] = cmp

        # ---- 3. repeat equality with the first output ----
        with torch.inference_mode():
            repeat_hidden = run_equivalent(collect=False)
        repeat = compare(repeat_hidden, first_hidden)
        repeat.pop('npu', None)
        repeat.pop('cpu', None)
        r['cases']['repeat_vs_first'] = repeat
        if not repeat.get('raw_bits_equal'):
            raise RuntimeError('repeat forward is not bit-identical to the first output on this rank')

        # ---- 4. actual stock entry-point probe;do not infer failure from [0] ----
        def stock_forward():
            return encoder.encode_ids(payload['input_ids'],pixel_values=payload['pixel_values'],
                                      image_grid_thw=payload['image_grid_thw'])
        probe: dict = {'attempted': True}
        try:
            with torch.inference_mode():
                stock = stock_forward()
            probe['raised'] = False
            probe['returned'] = tensor_info(stock)
            probe['matches_equivalent_path'] = compare(stock, first_hidden).get('raw_bits_equal')
            if not probe['matches_equivalent_path']:
                raise RuntimeError('stock encode_ids differs from the equivalent diagnostic path')
        except BaseException as error:
            probe['raised'] = True
            probe['exception_type'] = type(error).__name__
            probe['exception_message'] = str(error)[:2000]
            warn('stock encode_ids raised the recorded exception;no assumed shape-defect attribution. '
                 'Warm timings fall back to the explicit diagnostic path.')
        r['cases']['stock_encode_ids_probe'] = probe
        use_stock = not probe['raised']
        r['stock_encode_ids_timed'] = use_stock

        # ---- 5. warmed timings, proof hooks off, includes final D2H ----
        timings, output_hashes = [], []
        for call in range(WARM_CALLS):
            torch.npu.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                hidden = stock_forward() if use_stock else run_equivalent(collect=False)
            torch.npu.synchronize()
            seconds = time.perf_counter() - started
            timings.append(seconds)
            output_hashes.append(tensor_info(hidden)['sha256'])
            r['cases'].setdefault('warm_calls', []).append(
                {'call': call, 'synchronized_wall_s_including_d2h': seconds,
                 'hidden_sha256': output_hashes[-1]})
        r['timing'] = {
            'samples_synchronized_wall_s_including_d2h': timings,
            'rank_note': 'the root takes the rank maximum across all 8 receipts; this driver never '
                         'collapses ranks',
            'path': ('private FP16 candidate encode_ids including final .cpu();proof hooks removed'
                     if use_stock else 'explicit _encode-equivalent fallback including final .cpu()'),
            'not_stock_encode_ids': not use_stock,
            'not_official_benchmark': True,
            'warm_outputs_bit_equal': len(set(output_hashes)) == 1,
        }
        if len(set(output_hashes)) != 1:
            raise RuntimeError('warm full forwards on this rank produced different outputs')

        # ---- 6. frozen parameter/input re-hash ----
        parameter_sha_after = parameters_sha(encoder)
        input_sha_after = inputs_sha(payload)
        r['frozen_inputs_after'] = {'parameters_sha256': parameter_sha_after,
                                    'input_pair_sha256': input_sha_after}
        r['frozen_inputs_unchanged'] = (parameter_sha_before == parameter_sha_after
                                        and input_sha_before == input_sha_after)
        if not r['frozen_inputs_unchanged']:
            raise RuntimeError('parameters or inputs changed during the diagnostic run')

        r['cases']['final_checks'] = {
            'finite': True,
            'repeat_verified': True,
            'frozen_inputs_unchanged': True,
            'uniform_fp16_parameters': True,
            'numerical_threshold_chosen': None,
            'full_encoder_precision_accepted': False,
        }
        r['tp8_deviation'] = {
            'row_parallel_partial_rounding':
                'row-parallel projections round the FP16 partial GEMM output to FP16 before the '
                'FP32 all-reduce; the FP32 collective cannot restore precision already rounded '
                'away. This driver measures the consequence and does not fix it.',
            'status': 'known row-GEMM rounding deviation, not fixed, not waived',
            'embedding_collective': 'the candidate promotes the FP16 vocabulary-embedding '
                                    'all-reduce to FP32',
        }
        # Context reported by the root's separate Round80 leaf-operator controls. It is
        # recorded for interpretation only: this driver did not measure it, and it is not
        # a tolerance or a waiver.
        r['context_from_root_leaf_controls'] = {
            'attribution': 'reported by root Round80 leaf operator controls; NOT measured by this driver',
            'half_linear_with_bias_is_bitexact_equal_to': 'native no-bias Half GEMM plus a Half '
                                                          'bias add, i.e. one extra output rounding',
            'unbiased_linear_cpu_delta_relative_rms_approx': 9.7e-06,
            'biased_linear_cpu_delta_relative_rms_approx': 3.6e-04,
            'direct_fp32_proxy_cpu_delta_relative_rms_approx': 9.2e-06,
            'grouped_fp32_output_half_matmul': 'not available: npu_grouped_matmul with FP16 inputs '
                                               'and output_dtype=FP32 failed with 161002 (y must be FP16)',
            'implication_for_this_run': 'biased Half linears are expected to contribute more '
                                        'divergence than unbiased ones; the receipt reports measured '
                                        'error magnitudes and does not attribute them to one cause',
        }
        r['completed'] = True
    except BaseException as error:
        r['failure'] = {'type': type(error).__name__, 'message': str(error)[:4000]}
        raise
    finally:
        for handle in handles:
            try:
                handle.remove()
            except BaseException:  # noqa: BLE001
                pass
        if dist_ready:
            try:
                destroy_model_parallel()
                cleanup['destroy_model_parallel'] = 'ok'
            except BaseException as error:  # noqa: BLE001
                cleanup['destroy_model_parallel_error'] = repr(error)
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                    cleanup['destroy_process_group'] = 'ok'
            except BaseException as error:  # noqa: BLE001
                cleanup['destroy_process_group_error'] = repr(error)
        if kv_adapter is not None:
            r['kv_padding_final'] = kv_adapter.snapshot()
            kv_adapter.uninstall()
            cleanup['kv_adapter_uninstalled'] = True
        r['cleanup'] = cleanup
        writer.write(r)


if __name__ == '__main__':
    main()