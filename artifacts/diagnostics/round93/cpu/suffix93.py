#!/usr/bin/env python3
"""Round93 original-HF CPU FP16 selected-suffix (raw layers 25..49) propagation control.

STATUS: SOURCE ONLY.  Never executed here against torch/transformers/a CPU model/an
NPU/CANN: no SSH, no install, no model run, no production edit, no commit.  Every
number it prints belongs to the root agent that stages and runs it with the frozen
Round78 CPU gold and the actual Round91/Round92 eight-rank raw-layer-24 capture.

What it computes
----------------
Round92 propagated the actual NPU raw layer-24 state (the Round78
``language_layers[24]`` boundary: PRE DeepStack, PRE final norm) through one
original CPU layer.  This module propagates it through the whole remaining
selected suffix, **raw layers 25..49**, rebuilt from the pinned official
checkpoint as 25 original HF ``Qwen3VLTextDecoderLayer`` modules with the original
HF ``Qwen3VLTextRotaryEmbedding`` (Float32 CPU inverse frequencies, M-RoPE
layout, cos/sin cast to FP16), the original ``create_causal_mask`` call of
``Qwen3VLTextModel.forward`` (same five kwargs, ``past_key_values=None``,
all-ones mask) and ``SDPBackend.MATH`` for all 64 heads.  No DeepStack injection
after raw24, no final norm, so the last boundary is the pre-final-norm
selected-50 output.

Reference gate (fail closed, before any NPU input is read)
---------------------------------------------------------
``CPU_raw24`` propagated through layers 25..49 must reproduce **all 25** frozen
Round78 raw-boundary SHA256 values (``language_layers[25..49]``) bit-exactly, and
raw49 must equal the frozen ``FP16_selected50.pt`` tensor SHA.  That gate
validates the reconstruction end to end: pinned HF source, original FP16
conversion for 275 tensors, M-RoPE rows, causal mask, SDPA MATH attention and the
layer sequence.  Any mismatch raises instead of producing numbers.

What it is and is not
--------------------
``CPU_suffix(NPU raw24)`` versus ``CPU_suffix(CPU raw24)`` is an *input-only*
sensitivity control with **no NPU arithmetic factor**: it cannot by itself prove a
hardware operator bug, a per-layer NPU error, or a lossless pass.  It is not a
Float-weight teacher, casts no BF16 gold output, adds no final norm, and selects
no numerical tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

ROOT_DEFAULT = Path('/tmp/h3_round93_suffix01')
INPUTS_DEFAULT = Path('/tmp/h3_round92_case02/NPU_raw24_inputs92.pt')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
R78_ROOT = Path('/tmp/h3_round78_hf_fp1601')

# --- pinned Round92 helper: reused verbatim, never edited --------------------
HELPER_NAME = 'original_cpu_block25_92.py'
HELPER_SHA256 = '838027c883bed6d1e5a0b839e0c2accde2457bef9cc81d58129fdcad38d65f4b'
HELPER_MODULE_NAME = 'original_cpu_block25_92'

FIRST_LAYER, LAST_LAYER = 25, 49
LAYER_INDICES = tuple(range(FIRST_LAYER, LAST_LAYER + 1))
SELECTED_LANGUAGE_LAYERS = 50
EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN = 1, 2043, 5120
PARAMETER_SUFFIXES = (
    'input_layernorm.weight', 'mlp.down_proj.weight', 'mlp.gate_proj.weight',
    'mlp.up_proj.weight', 'post_attention_layernorm.weight', 'self_attn.k_norm.weight',
    'self_attn.k_proj.weight', 'self_attn.o_proj.weight', 'self_attn.q_norm.weight',
    'self_attn.q_proj.weight', 'self_attn.v_proj.weight')

R78_RECEIPT_SHA256 = '44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
R78_MM_INPUTS_FILE_SHA256 = '6ab37f7202731843aebce9f3f8453232c72d45b2239caba679bcb2611e626e70'
R78_SELECTED50_NAME = 'FP16_selected50.pt'
R78_SELECTED50_FILE_SHA256 = '0137a434148366c3dcbf825970274ad0faa036666561d7aecc0170cf136b3fe4'
R78_RAW24_TENSOR_SHA256 = '7a2178195b3ec65e09ba4c7587acb052e04996a4023f6b1a5f1b2088e2eef084'
R78_RAW49_TENSOR_SHA256 = '82944620090359e7fc7eccb86471117e010b7838da83de5a43ceb46bf3453bd5'

CORRECTED_CASE = 'corrected_vision_native_text'
CORRECTED_RAW24_SHA256 = 'e232f61b4260b81b08bb0cf3eb96a7df6d0a56ed3c0c82ab9d0ec0f15004a528'

REFERENCE_NAME = 'CPU_suffix_reference93.json'
PROGRESS_NAME = 'CPU_suffix_progress93.json'
OUTPUTS_NAME = 'CPU_suffix_outputs93.pt'
FAILURE_NAME = 'CPU_suffix_failure93.json'

STAGE = ['start']


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_original_helper(path=None):
    candidates = [Path(path)] if path is not None else [
        Path(__file__).resolve().parent / HELPER_NAME, ROOT_DEFAULT / HELPER_NAME]
    chosen = next((item for item in candidates if item.is_file()), None)
    if chosen is None:
        raise RuntimeError('staged Round92 helper not found in %r'
                           % [str(item) for item in candidates])
    actual = file_sha(chosen)
    if actual != HELPER_SHA256:
        raise RuntimeError('Round92 helper SHA mismatch: %s != %s' % (actual, HELPER_SHA256))
    if HELPER_MODULE_NAME in sys.modules:
        raise RuntimeError('%s already imported; refusing an ambiguous helper'
                           % HELPER_MODULE_NAME)
    spec = importlib.util.spec_from_file_location(HELPER_MODULE_NAME, chosen)
    module = importlib.util.module_from_spec(spec)
    sys.modules[HELPER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    if file_sha(module.__file__) != HELPER_SHA256:
        raise RuntimeError('imported Round92 helper changed underneath the SHA gate')
    return module, chosen, actual


def atomic_replace(root, temporary, destination):
    """Replace only inside the run root, with matching basenames and a .tmp suffix."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    temporary = Path(temporary).resolve()
    if temporary.parent != root or destination.parent != root:
        raise RuntimeError('atomic replace escaped the run root: %s -> %s'
                           % (temporary, destination))
    if temporary.name != destination.name + '.tmp':
        raise RuntimeError('unexpected temporary name %r for %r'
                           % (temporary.name, destination.name))
    os.replace(temporary, destination)


def write_progress(root, receipt):
    path = Path(root) / PROGRESS_NAME
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    atomic_replace(root, temporary, path)


def layers_parameters_sha(_original, layers):
    import torch

    digest = hashlib.sha256()
    for index, layer in zip(LAYER_INDICES, layers):
        digest.update(('layer.%d' % index).encode())
        for name, parameter in layer.named_parameters():
            digest.update(name.encode())
            digest.update(memoryview(parameter.detach().cpu().contiguous()
                                     .view(torch.uint8).numpy()))
    return digest.hexdigest()


def boundary_truths(_original, reference, receipt):
    layers = reference['language_layers']
    if len(layers) != SELECTED_LANGUAGE_LAYERS:
        raise RuntimeError('Round78 gold language-layer inventory is not 50')
    truths = {}
    for index in LAYER_INDICES:
        sha = layers[index].get('output', {}).get('sha256')
        if not isinstance(sha, str) or len(sha) != 64:
            raise RuntimeError('Round78 raw layer %d has no boundary SHA' % index)
        truths[index] = sha
    if truths[LAST_LAYER] != R78_RAW49_TENSOR_SHA256:
        raise RuntimeError('Round78 raw layer-49 SHA changed')
    if layers[FIRST_LAYER - 1]['output']['sha256'] != R78_RAW24_TENSOR_SHA256:
        raise RuntimeError('Round78 raw layer-24 SHA changed')
    receipt['r78_boundary_truths'] = {str(i): truths[i] for i in LAYER_INDICES}
    return truths


def verify_conversion_inventory(reference, receipt):
    """Fail fast, before the expensive 25-layer build, if a suffix layer's Round78
    FP16 conversion entries are missing."""
    conversion = reference.get('parameter_conversion')
    if not isinstance(conversion, dict) or not conversion:
        raise RuntimeError('Round78 parameter_conversion inventory is missing')
    missing = []
    for index in LAYER_INDICES:
        prefix = 'language_model.layers.%d.' % index
        for suffix in PARAMETER_SUFFIXES:
            if prefix + suffix not in conversion:
                missing.append(prefix + suffix)
    receipt['conversion_inventory'] = {
        'layer_indices': list(LAYER_INDICES),
        'parameters_per_layer': len(PARAMETER_SUFFIXES),
        'expected_entries': len(LAYER_INDICES) * len(PARAMETER_SUFFIXES),
        'missing_entries': missing[:16], 'all_present': not missing}
    if missing:
        raise RuntimeError('Round78 conversion inventory lacks %d suffix entries for layers '
                           '25..49 (first: %r); refusing a partial reconstruction'
                           % (len(missing), missing[0]))
    return conversion


def build_suffix_layers(original, reference, receipt):
    import torch

    layers, infos = [], []
    for index in LAYER_INDICES:
        STAGE[0] = 'build_layer:%d' % index
        layer, hf, text_config, info = original.build_layer(MODEL, reference, index)
        info = dict(info)
        stale = info.pop('only_layer_25_weights_loaded', None)
        if set(info.get('parameter_names', [])) != set(PARAMETER_SUFFIXES):
            raise RuntimeError('layer %d parameter inventory changed' % index)
        if info.get('parameter_dtypes') != [str(torch.float16)]:
            raise RuntimeError('layer %d is not uniformly FP16' % index)
        if info.get('layer_index') != index:
            raise RuntimeError('built layer index %r != %d' % (info.get('layer_index'), index))
        if len(info.get('conversion_reference_sha256_verified', {})) != len(PARAMETER_SUFFIXES):
            raise RuntimeError('layer %d conversion verification is incomplete' % index)
        if any(p.dtype != torch.float16 or p.device.type != 'cpu' for p in layer.parameters()):
            raise RuntimeError('layer %d parameters are not real CPU FP16 tensors' % index)
        info.update({'layer_index': index, 'only_suffix_layers_25_to_49_loaded': True,
                     'stale_inherited_layer25_flag_removed': stale is not None,
                     'stale_inherited_layer25_flag_value_ignored': stale})
        layers.append(layer)
        infos.append(info)
    receipt['original_suffix'] = {
        'layer_indices': list(LAYER_INDICES), 'layer_count': len(LAYER_INDICES),
        'layers': infos, 'loaded_only_layers_25_to_49': True,
        'retained_parameter_tensors': len(LAYER_INDICES) * len(PARAMETER_SUFFIXES),
        'all_parameters_original_Half': True,
        'stale_helper_layer25_metadata_ignored_not_published': True}
    return layers, hf, text_config


def propagate(original, layers, hidden_states, mask, text_positions, position_embeddings,
              boundary_sink, label):
    outputs, shas = {}, {}
    hidden = hidden_states
    for index, layer in zip(LAYER_INDICES, layers):
        STAGE[0] = '%s:layer%d' % (label, index)
        hidden = original.run_block25(layer, hidden, mask, text_positions,
                                      position_embeddings)
        info = original.tensor_info(hidden)
        if info['finite'] is not True:
            raise RuntimeError('%s layer %d output is not finite' % (label, index))
        shas[index], outputs[index] = info['sha256'], hidden
        boundary_sink[index] = {'output': info}
    return outputs, shas


def run_reference(root=ROOT_DEFAULT, inputs_path=None, helper_path=None,
                  baseline_repeats=1, case=CORRECTED_CASE):
    """Write CPU_suffix_reference93.json and CPU_suffix_outputs93.pt; never overwrite."""
    import torch

    root = Path(root)
    inputs_path = Path(inputs_path) if inputs_path is not None else INPUTS_DEFAULT
    reference_path, outputs_path = root / REFERENCE_NAME, root / OUTPUTS_NAME
    progress_path, failure_path = root / PROGRESS_NAME, root / FAILURE_NAME
    receipt = {
        'scope': __doc__,
        'kind': 'original-HF CPU FP16 raw layers 25..49 suffix propagation control',
        'source_sha256': file_sha(Path(__file__)), 'root': str(root),
        'layer_indices': list(LAYER_INDICES),
        'selected_language_layers': SELECTED_LANGUAGE_LAYERS,
        'boundary': 'original HF raw decoder layers 25..49, PRE DeepStack injection, PRE '
                    'final norm; last boundary is the selected-50 pre-final-norm output',
        'no_final_norm_after_layer49': True, 'no_DeepStack_in_suffix': True,
        'no_Float_linear_weights_or_teacher': True, 'no_BF16_gold_output_cast': True,
        'numerical_tolerance': None, 'full_encoder_precision_accepted': False,
        'whole_goal_complete': False, 'not_a_speed_measurement': True,
        'does_not_modify_R78_gold': True, 'contains_no_NPU_arithmetic_factor': True,
        'gate_order': 'R78 receipt gates -> CPU raw24 -> original text inputs -> build 25 '
                      'layers -> reproduce ALL 25 R78 raw boundaries -> then read the actual '
                      'NPU raw24 bundle and propagate the corrected case twice',
        'baseline_repeats': int(baseline_repeats), 'completed': False}
    try:
        if not root.is_dir():
            raise RuntimeError('missing root directory: %s' % root)
        for path in (reference_path, progress_path, outputs_path, failure_path):
            if path.exists():
                raise RuntimeError('refusing to reuse a root that already holds %s' % path.name)
        if baseline_repeats not in (1, 2):
            raise RuntimeError('baseline_repeats must be 1 or 2')
        STAGE[0] = 'helper'
        original, helper_file, helper_sha = load_original_helper(helper_path)
        receipt['original_helper'] = {
            'module': HELPER_MODULE_NAME, 'path': str(helper_file), 'sha256': helper_sha,
            'matches_pin': helper_sha == HELPER_SHA256, 'reused_verbatim_not_edited': True,
            'imported_by_verified_path_not_copied': True}
        receipt['helper_metrics'] = original.install_helpers()
        torch.set_num_threads(original.CPU_THREADS)
        receipt['cpu_execution'] = {
            'torch': torch.__version__, 'threads': torch.get_num_threads(),
            'mkldnn_enabled': bool(torch.backends.mkldnn.enabled), 'works_on_cpu_only': True}
        if torch.get_num_threads() != original.CPU_THREADS or not torch.backends.mkldnn.enabled:
            raise RuntimeError('could not reproduce the Round78 CPU execution settings')

        STAGE[0] = 'r78_receipt'
        reference = original.load_r78_receipt(receipt)
        STAGE[0] = 'conversion_inventory'
        verify_conversion_inventory(reference, receipt)
        STAGE[0] = 'cpu_raw24'
        cpu_raw24 = original.load_cpu_raw24(receipt)
        STAGE[0] = 'original_text_inputs'
        original_embeds, original_positions = original.load_original_text_inputs(receipt)
        STAGE[0] = 'boundary_truths'
        truths = boundary_truths(original, reference, receipt)
        STAGE[0] = 'build_suffix'
        layers, hf, text_config = build_suffix_layers(original, reference, receipt)
        if len(layers) != len(LAYER_INDICES):
            raise RuntimeError('suffix layer construction is incomplete')
        STAGE[0] = 'positions_rotary_mask'
        text_positions, rotary_positions = original.split_original_positions(
            original_positions, receipt)
        position_embeddings = original.build_rotary(hf, text_config, rotary_positions,
                                                    original_embeds, receipt)
        mask = original.build_causal_mask(hf, text_config, original_embeds, text_positions,
                                          receipt)
        parameters_before = layers_parameters_sha(original, layers)

        STAGE[0] = 'baseline_pass'
        baseline_receipt = {}
        baseline_outputs, baseline_shas = propagate(
            original, layers, cpu_raw24, mask, text_positions, position_embeddings,
            baseline_receipt, 'baseline')
        mismatched = [i for i in LAYER_INDICES if baseline_shas[i] != truths[i]]
        receipt['baseline_pass'] = {
            'input': 'CPU_raw24: frozen Round78 raw layer-24 tensor',
            'boundaries': {str(i): baseline_receipt[i] for i in LAYER_INDICES},
            'matches_R78_all_25_boundaries': not mismatched,
            'mismatched_layers': mismatched,
            'R78_boundary_SHA256': {str(i): truths[i] for i in LAYER_INDICES},
            'validated_before_any_NPU_input_was_read': True,
            'repeat_mode': ('single_pass_gated_by_25_independent_R78_boundary_SHA256'
                            if baseline_repeats == 1 else 'two_full_passes_bit_equal')}
        if mismatched:
            raise RuntimeError('CPU suffix failed to reproduce frozen Round78 raw boundaries '
                               '%r (first: layer %d %s != %s); refusing to read NPU inputs'
                               % (mismatched, mismatched[0], baseline_shas[mismatched[0]],
                                  truths[mismatched[0]]))
        if baseline_repeats == 2:
            repeat_outputs, repeat_shas = propagate(
                original, layers, cpu_raw24, mask, text_positions, position_embeddings,
                {}, 'baseline_repeat')
            if repeat_shas != baseline_shas:
                raise RuntimeError('baseline suffix repeat is not bit-identical: %r'
                                   % [i for i in LAYER_INDICES
                                      if repeat_shas[i] != baseline_shas[i]])
            del repeat_outputs
        write_progress(root, receipt)

        STAGE[0] = 'npu_inputs'
        inputs, bundle_positions, declared = original.load_inputs_bundle(inputs_path, receipt)
        if case not in inputs:
            raise RuntimeError('capture bundle has no %r case; it has %r'
                               % (case, sorted(inputs)))
        corrected_input = inputs[case]
        corrected_input_sha = original.bytes_sha(corrected_input)
        if corrected_input_sha != CORRECTED_RAW24_SHA256:
            raise RuntimeError('corrected raw24 input is %s, not the declared %s'
                               % (corrected_input_sha, CORRECTED_RAW24_SHA256))
        if tuple(corrected_input.shape) != (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN):
            raise RuntimeError('corrected raw24 shape %r is not the pinned H3 text geometry'
                               % (tuple(corrected_input.shape),))
        if original.bytes_sha(bundle_positions) != original.bytes_sha(original_positions):
            raise RuntimeError('capture bundle positions differ from the frozen original rows')
        receipt['corrected_case'] = {
            'case': case, 'input_SHA256': corrected_input_sha,
            'declared_source_raw24_SHA256': declared[case],
            'all_declared_cases': {name: declared[name] for name in sorted(declared)},
            'input_vs_CPU_raw24': original.compare_pair(corrected_input, cpu_raw24)}

        pass_shas, pass_receipts = {}, []
        corrected_outputs = corrected_shas = None
        for pass_index in (1, 2):
            STAGE[0] = 'corrected_pass:%d' % pass_index
            boundary_receipt = {}
            outputs, shas = propagate(
                original, layers, corrected_input, mask, text_positions,
                position_embeddings, boundary_receipt, 'corrected_pass%d' % pass_index)
            boundaries = {}
            for index in LAYER_INDICES:
                boundaries[str(index)] = {
                    'output': boundary_receipt[index]['output'],
                    'matches_R78_boundary_SHA256': shas[index] == truths[index],
                    'R78_boundary_SHA256': truths[index],
                    'vs_CPU_baseline_boundary': original.compare_pair(
                        outputs[index], baseline_outputs[index])}
            pass_shas[pass_index] = shas
            corrected_outputs, corrected_shas = outputs, shas
            pass_receipts.append({
                'pass_index': pass_index, 'boundaries': boundaries,
                'all_25_boundaries_recorded': len(shas) == len(LAYER_INDICES),
                'matches_R78_all_25_boundaries':
                    all(shas[i] == truths[i] for i in LAYER_INDICES)})
            receipt['corrected_passes'] = pass_receipts
            write_progress(root, receipt)
            del outputs
        if pass_shas[1] != pass_shas[2]:
            raise RuntimeError('corrected suffix propagation is not bit-repeatable: %r'
                               % [i for i in LAYER_INDICES if pass_shas[1][i] != pass_shas[2][i]])
        receipt['corrected_repeat_bits_equal_all_25_boundaries'] = True

        STAGE[0] = 'r78_selected50_truth'
        selected_path = R78_ROOT / R78_SELECTED50_NAME
        selected_file_sha = file_sha(selected_path)
        if selected_file_sha != R78_SELECTED50_FILE_SHA256:
            raise RuntimeError('Round78 selected-50 payload changed: %s' % selected_file_sha)
        selected_payload = torch.load(selected_path, map_location='cpu', weights_only=True)
        truth49 = original.materialise(selected_payload['pre_final_norm'])
        if original.tensor_info(truth49)['sha256'] != R78_RAW49_TENSOR_SHA256:
            raise RuntimeError('Round78 selected-50 tensor SHA changed')
        if tuple(truth49.shape) != (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN):
            raise RuntimeError('Round78 selected-50 shape %r is not the pinned geometry'
                               % (tuple(truth49.shape),))

        baseline49, corrected49 = baseline_outputs[LAST_LAYER], corrected_outputs[LAST_LAYER]
        baseline_vs_truth = original.compare_pair(baseline49, truth49)
        corrected_vs_truth = original.compare_pair(corrected49, truth49)
        corrected_vs_baseline = original.compare_pair(corrected49, baseline49)
        receipt['selected50'] = {
            'R78_selected50_file_sha256': selected_file_sha,
            'R78_selected50_tensor_sha256': R78_RAW49_TENSOR_SHA256,
            'baseline_vs_R78_truth': baseline_vs_truth,
            'baseline_matches_R78_truth_bits': baseline_vs_truth['raw_bits_equal'],
            'corrected_vs_R78_truth': corrected_vs_truth,
            'corrected_vs_CPU_baseline': corrected_vs_baseline,
            'is_input_only_control_not_whole_encoder_accuracy': True}
        if not baseline_vs_truth['raw_bits_equal']:
            raise RuntimeError('CPU baseline raw49 does not match the frozen Round78 truth')

        input_rms = receipt['corrected_case']['input_vs_CPU_raw24']['relative_rms']
        growth, previous = [], input_rms
        for index in LAYER_INDICES:
            rms = pass_receipts[0]['boundaries'][str(index)][
                'vs_CPU_baseline_boundary']['relative_rms']
            growth.append({'layer': index, 'relative_rms_vs_baseline': rms,
                           'growth_over_previous_boundary': (None if not previous
                                                             else rms / previous)})
            previous = rms
        receipt['output_error_growth'] = {
            'per_layer': growth,
            'largest_boundary_relative_rms': max(
                growth, key=lambda row: row['relative_rms_vs_baseline']),
            'input_relative_rms': input_rms,
            'selected50_relative_rms': corrected_vs_baseline['relative_rms'],
            'selected50_over_input_amplification': (
                None if not input_rms else corrected_vs_baseline['relative_rms'] / input_rms),
            'descriptive_only_no_tolerance': True}
        receipt['isolation_semantics'] = (
            'Original CPU FP16 raw layers 25..49 evaluated on an actual NPU raw layer-24 '
            'state is an input-only propagation control: it measures CPU_suffix(NPU input) '
            'versus CPU_suffix(CPU input) and contains no NPU arithmetic factor. It is not a '
            'Float-weight teacher, casts no BF16 gold output, adds no final norm and no '
            'DeepStack, and selects no tolerance. It alone cannot show a unique hardware '
            'operator bug, a per-layer NPU error, or a lossless pass.')

        STAGE[0] = 'invariants'
        frozen_after = {
            'parameters_sha256': layers_parameters_sha(original, layers),
            'cpu_raw24_tensor_sha256': original.raw_tensor_sha(cpu_raw24),
            'r78_receipt_sha256': file_sha(R78_ROOT / original.R78_RECEIPT_NAME),
            'r78_multimodal_inputs_sha256': file_sha(R78_ROOT / original.R78_MM_INPUTS_NAME),
            'r78_selected50_sha256': file_sha(R78_ROOT / R78_SELECTED50_NAME),
            'inputs_file_sha256': file_sha(inputs_path),
            'helper_sha256': file_sha(helper_file),
            'source_sha256': file_sha(Path(__file__))}

        receipt['frozen_after'] = frozen_after
        expectations = {
            'parameters_unchanged': frozen_after['parameters_sha256'] == parameters_before,
            'cpu_raw24_unchanged':
                frozen_after['cpu_raw24_tensor_sha256'] == R78_RAW24_TENSOR_SHA256,
            'r78_receipt_unchanged': frozen_after['r78_receipt_sha256'] == R78_RECEIPT_SHA256,
            'r78_multimodal_inputs_unchanged': frozen_after['r78_multimodal_inputs_sha256']
            == R78_MM_INPUTS_FILE_SHA256,
            'r78_selected50_unchanged':
                frozen_after['r78_selected50_sha256'] == R78_SELECTED50_FILE_SHA256,
            'inputs_file_unchanged': frozen_after['inputs_file_sha256']
            == receipt['inputs_source']['file_sha256'],
            'helper_unchanged': frozen_after['helper_sha256'] == HELPER_SHA256,
            'source_unchanged': frozen_after['source_sha256'] == receipt['source_sha256']}
        receipt['frozen_after'].update(expectations)
        if not all(expectations.values()):
            raise RuntimeError('a frozen input, weight, helper or reference changed during the '
                               'run: %r' % [key for key, value in expectations.items()
                                            if not value])

        STAGE[0] = 'write_outputs'
        temporary_outputs = outputs_path.with_name(outputs_path.name + '.tmp')
        temporary_reference = reference_path.with_name(reference_path.name + '.tmp')
        torch.save({
            'baseline_selected50': baseline49, 'corrected_selected50': corrected49,
            'source_input_SHA256': corrected_input_sha, 'source_input_case': case,
            'baseline_selected50_SHA256': original.bytes_sha(baseline49),
            'corrected_selected50_SHA256': original.bytes_sha(corrected49),
            'R78_selected50_tensor_SHA256': R78_RAW49_TENSOR_SHA256,
            'layer_boundaries': list(LAYER_INDICES),
            'boundary': 'raw layer 49 pre-final-norm selected-50 output',
            'is_input_only_control': True}, temporary_outputs)
        outputs_sha = file_sha(temporary_outputs)
        receipt['outputs_file'] = {
            'path': str(outputs_path), 'bytes': temporary_outputs.stat().st_size,
            'sha256': outputs_sha,
            'tensor_keys': ['baseline_selected50', 'corrected_selected50'],
            'source_input_SHA256': corrected_input_sha,
            'baseline_selected50_SHA256': original.bytes_sha(baseline49),
            'corrected_selected50_SHA256': original.bytes_sha(corrected49)}
        receipt['outputs_file_SHA256'] = outputs_sha
        receipt['inputs_source_path'] = str(inputs_path)
        receipt['inputs_file_SHA256'] = receipt['inputs_source']['file_sha256']
        receipt['completed'] = True
        temporary_reference.write_text(json.dumps(receipt, indent=2) + '\n')
        if reference_path.exists() or outputs_path.exists():
            raise RuntimeError('refusing to overwrite outputs created concurrently')
        atomic_replace(root, temporary_outputs, outputs_path)
        atomic_replace(root, temporary_reference, reference_path)
        return receipt
    except BaseException as error:
        failure = {
            'scope': 'Round93 original CPU suffix 25..49 propagation-control failure record',
            'stage': STAGE[0], 'completed': False,
            'error': {'type': type(error).__name__, 'message': str(error)},
            'traceback': traceback.format_exc(),
            'reference_written': reference_path.exists(),
            'outputs_written': outputs_path.exists(),
            'progress_written': progress_path.exists(),
            'source_sha256': file_sha(Path(__file__)), 'helper_sha256': HELPER_SHA256}
        try:
            with failure_path.open('x') as handle:
                handle.write(json.dumps(failure, indent=2) + '\n')
        except FileExistsError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Original HF CPU FP16 raw layers 25..49 suffix propagation control '
                    '(Round93)')
    parser.add_argument('--root', default=str(ROOT_DEFAULT))
    parser.add_argument('--inputs', default=str(INPUTS_DEFAULT))
    parser.add_argument('--helper', default=None)
    parser.add_argument('--baseline-repeats', type=int, default=1, choices=(1, 2))
    parser.add_argument('--case', default=CORRECTED_CASE)
    arguments = parser.parse_args(argv)
    root = Path(arguments.root)
    receipt = run_reference(root=root, inputs_path=Path(arguments.inputs),
                            helper_path=Path(arguments.helper) if arguments.helper else None,
                            baseline_repeats=arguments.baseline_repeats, case=arguments.case)
    print(json.dumps({
        'reference': str(root / REFERENCE_NAME), 'outputs': str(root / OUTPUTS_NAME),
        'progress': str(root / PROGRESS_NAME), 'completed': receipt['completed'],
        'baseline_matches_all_25_R78_boundaries':
            receipt['baseline_pass']['matches_R78_all_25_boundaries'],
        'corrected_repeat_bits_equal_all_25_boundaries':
            receipt['corrected_repeat_bits_equal_all_25_boundaries'],
        'corrected_selected50_relative_rms':
            receipt['selected50']['corrected_vs_R78_truth']['relative_rms'],
        'outputs_file_SHA256': receipt['outputs_file_SHA256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())