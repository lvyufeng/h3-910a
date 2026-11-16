#!/usr/bin/env python3
"""Actual fresh vision Linear factors; full encode_ids and exact-reference gates."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'round70'))
from summarize import exact, tensor_contract, validate_identities

ROOT = Path(__file__).resolve().parent
MODES = ('native', 'activation_only', 'shadow_only', 'both', 'restored')
MAX_BYTES = 2362750400


def expected_modules():
    result = {}
    for layer in range(27):
        for family, width, out in (('attn.qkv', 1152, 3456), ('attn.proj', 1152, 1152),
                                   ('mlp.linear_fc1', 1152, 4304), ('mlp.linear_fc2', 4304, 1152)):
            result['blocks.%d.%s' % (layer, family)] = ([8064, width], [out, width], [out], [8064, out])
    for prefix in ('merger', *(f'deepstack_merger_list.{i}' for i in range(3))):
        result[prefix + '.linear_fc1'] = ([2016, 4608], [4608, 4608], [4608], [2016, 4608])
        result[prefix + '.linear_fc2'] = ([2016, 4608], [5120, 4608], [5120], [2016, 5120])
    assert len(result) == 116
    assert sum((w[0] * w[1] + b[0]) * 4 for _, w, b, _ in result.values()) == MAX_BYTES
    return result


def count(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def validate_checks(case, modules):
    checks = case['vision_linear_checks']
    mode = case['vision_mode']
    if mode in ('native', 'restored'):
        if checks:
            raise ValueError('Native/restored unexpectedly contain double-compute checks')
        return
    if len(checks) != 116 or Counter(check['name'] for check in checks) != Counter({name: 1 for name in modules}):
        raise ValueError('Missing or duplicate actual vision Linear check')
    for check in checks:
        x, w, b, y = modules[check['name']]
        if (check['input_shape'] != x or check['weight_shape'] != w or check['bias_shape'] != b or check['output_shape'] != y
                or check['activation_enabled'] != (mode in ('activation_only', 'both')) or check['shadow_enabled'] != (mode in ('shadow_only', 'both'))):
            raise ValueError('Vision Linear actual GEMM shape or factor selection changed')
        for name, shape in (('input_cast', x), ('weight_shadow', w), ('bias_shadow', b), ('output', y)):
            if not exact(check[name], count(shape)):
                raise ValueError('Actual vision input/weight/bias/output identity proof failed')
        for actual, native in (('input_cast_layout', 'native_input_cast_layout'), ('weight_shadow_layout', 'native_weight_cast_layout'),
                               ('bias_shadow_layout', 'native_bias_cast_layout'), ('output_layout', 'native_output_layout')):
            if check[actual] != check[native]:
                raise ValueError('Actual vision cast/projection native shape/stride/format changed')
        if not check['input_readonly']:
            raise ValueError('Actual original BF16 activation changed')
        for name, shape, dtype in (('input_cast_layout', x, 'torch.float32'), ('weight_shadow_layout', w, 'torch.float32'),
                                   ('bias_shadow_layout', b, 'torch.float32'), ('output_layout', y, 'torch.bfloat16')):
            if check[name]['shape'] != shape or check[name]['dtype'] != dtype:
                raise ValueError('Actual projection operand/result layout dtype changed')


def aggregate(rows, previous):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique genuine NPU rank receipts')
    if any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Incomplete/failed vision factor experiment cannot be accepted')
    rows = sorted(rows, key=lambda row: row['rank'])
    prior = {row['rank']: row for row in previous}
    modules = expected_modules()
    first_reference = prior[0]['cases']['baseline']['first_identities']
    identities, output_proofs = [], []
    result = {'scope': 'Five independent vision Linear activation/shadow factors on full fresh resident encode_ids: original FP16 transport pixels and complete original BF16 checkpoint, fixedRound70fasttext, unchangedFP32visionGEMM/bias/math. Includes nativeCPUoutput; excludes outerHTTPprocessor/distribute/broadcast/load/offload. Exact stable experimental reference, not independent HF/original production/officiallossless quality.',
              'cases': {}, 'all_input_and_parameter_bytes_unchanged': True,
              'complete_output_identity_accepted': False, 'complete_encode_ids_under2s_accepted': False,
              'whole_service_text_under2s_proven': False, 'official_goal_complete': False, 'production_deployed': False,
              'actual_vision_linear_checks': 0, 'actual_input_cast_checks': 0, 'actual_weight_and_bias_checks': 0,
              'enabled_activation_cast_checks': 0, 'native_activation_control_comparisons': 0,
              'enabled_weight_and_bias_shadow_checks': 0, 'native_weight_and_bias_control_comparisons': 0}
    expected_flags = {'VLLM_OMNI_910A_NO_FUSED_NORM': '1', 'VLLM_OMNI_910A_NO_FUSED_ACT': '1',
                      'VLLM_OMNI_910A_TEXT_HF_MATH': '1', 'VLLM_OMNI_910A_TEXT_FP32_LINEAR': '1',
                      'VLLM_OMNI_910A_TEXT_PAD_KV': '1', 'VLLM_OMNI_910A_VISION_FP32': '1'}
    for row in rows:
        old = prior[row['rank']]
        if (row['parameters_before'] != row['parameters_after'] or row['parameters_before'] != old['parameters_before']
                or row['actual_inputs_before'] != row['actual_inputs_after'] or row['actual_inputs_before'] != old['actual_inputs_before']):
            raise ValueError('Vision factors changed actual input or complete parameter bytes')
        for field in ('checkpoint_root', 'checkpoint_index_sha256', 'capture_file_sha256', 'bf16_complete_encoder_parameter_bytes',
                      'component_parameter_bytes', 'parameter_dtypes', 'loaded_parameter_tensors'):
            if row[field] != old[field]:
                raise ValueError('Original full BF16 checkpoint or transported processor input changed')
        if any(row['environment'].get(name) != value for name, value in expected_flags.items()) or 'VLLM_OMNI_910A_TE_DTYPE' in row['environment']:
            raise ValueError('Fixed vision/text precision flags changed')
        if (row['math_flags_before'] != row['math_flags_after'] or row['math_flags_before'] != rows[0]['math_flags_before']
                or set(row['math_flags_before']) != {'matmul_allow_hf32', 'conv_allow_hf32'}):
            raise ValueError('Actual HF32 runtime flags changed across rank or time')
        if not row['vision_wrappers_restored'] or row['vision_shadow_bytes_after_clear'] != 0:
            raise ValueError('Vision instance wrappers/shadow did not restore on exit')
        if not exact(row['first_native_vs_round70_fresh_reference'], 10460160):
            raise ValueError('Native full fresh encoder reference differs from Round70')
        actual_modules = row['actual_vision_linear_modules']
        if len(actual_modules) != 116 or Counter(module['name'] for module in actual_modules) != Counter({name: 1 for name in modules}):
            raise ValueError('Actual vision Linear parameter set missing or duplicated')
        for module in actual_modules:
            _, w, b, _ = modules[module['name']]
            if module['weight_shape'] != w or module['bias_shape'] != b or module['weight_dtype'] != 'torch.bfloat16' or module['bias_dtype'] != 'torch.bfloat16':
                raise ValueError('Actual vision weight/bias geometry or dtype changed')
    for mode in MODES:
        cases = [row['cases'][mode] for row in rows]
        for case in cases:
            if case['vision_mode'] != mode or case['fixed_text_mode'] != 'shared_qkv_add_square_norm_tail_both_casts' or len(case['warm_calls']) != 3:
                raise ValueError('Vision factor selection or fixed text math/repeat count changed')
            validate_checks(case, modules)
            result['actual_vision_linear_checks'] += len(case['vision_linear_checks'])
            result['actual_input_cast_checks'] += len(case['vision_linear_checks'])
            result['actual_weight_and_bias_checks'] += 2 * len(case['vision_linear_checks'])
            result['enabled_activation_cast_checks'] += sum(check['activation_enabled'] for check in case['vision_linear_checks'])
            result['native_activation_control_comparisons'] += sum(not check['activation_enabled'] for check in case['vision_linear_checks'])
            result['enabled_weight_and_bias_shadow_checks'] += 2 * sum(check['shadow_enabled'] for check in case['vision_linear_checks'])
            result['native_weight_and_bias_control_comparisons'] += 2 * sum(not check['shadow_enabled'] for check in case['vision_linear_checks'])
            validate_identities(case['first_identities'])
            identities.append(case['first_identities']); output_proofs.append(case['first_vs_baseline'])
            for call in case['warm_calls']:
                validate_identities(call['identities'])
                if call['output'] != call['identities']['complete_output'] or not math.isfinite(call['seconds']) or call['seconds'] <= 0:
                    raise ValueError('Wrong complete CPU output receipt or wall timer')
                identities.append(call['identities']); output_proofs.append(call['vs_baseline'])
            cache = case['vision_shadow_cache']
            enabled = mode in ('shadow_only', 'both')
            expected = {'entries': 232 if enabled else 0, 'cached_fp32_bytes': MAX_BYTES if enabled else 0,
                        'misses': 232 if enabled else 0, 'hits': 696 if enabled else 0,
                        'invalidations': 0, 'uncacheable_bypasses': 0, 'max_bytes': MAX_BYTES}
            if cache != expected:
                raise ValueError('Vision weight/bias shadow factor did not run its exact bounded policy')
            if case['text_shadow_cache']['entries'] != 200 or case['text_shadow_cache']['cached_fp32_bytes'] != 11927552000:
                raise ValueError('Fixed text shadow stack changed')
            if case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000}:
                raise ValueError('Fixed text K/V padding changed')
        maxima = [max(case['warm_calls'][i]['seconds'] for case in cases) for i in range(3)]
        result['cases'][mode] = {'warm_complete_encode_ids_rank_max_s': maxima, 'median_complete_encode_ids_s': statistics.median(maxima),
                                 'all_warm_complete_encode_ids_under2s': all(value < 2 for value in maxima),
                                 'first_rank_max_s_excluded_checks_compile_cache': max(case['first_s_excluded_checks_compile_cache_and_component_probes'] for case in cases),
                                 'vision_shadow_bytes_per_rank': cases[0]['vision_shadow_cache']['cached_fp32_bytes'],
                                 'allocated_hbm_bytes_per_rank': [case['hbm_allocated_bytes'] for case in cases],
                                 'reserved_hbm_bytes_per_rank': [case['hbm_reserved_bytes'] for case in cases]}
    result['intermediate_identity_against_round70'] = {name: all(item[name] == first_reference[name] for item in identities) for name in first_reference}
    result['complete_outputs_compared'] = len(identities)
    result['all_complete_outputs_exact'] = all(exact(proof, 10460160) for proof in output_proofs)
    result['complete_output_identity_accepted'] = all(result['intermediate_identity_against_round70'].values()) and result['all_complete_outputs_exact']
    result['complete_encode_ids_under2s_accepted'] = result['complete_output_identity_accepted'] and result['cases']['both']['all_warm_complete_encode_ids_under2s']
    native = result['cases']['native']['median_complete_encode_ids_s']
    result['speedups_vs_same_run_native'] = {mode: native / result['cases'][mode]['median_complete_encode_ids_s'] for mode in MODES}
    result['restored_timing_drift'] = result['cases']['restored']['median_complete_encode_ids_s'] / native - 1
    result['runtime_math_flags'] = rows[0]['math_flags_before']
    result['full_encode_ids_output_sha256'] = first_reference['complete_output']['sha256']
    return result


def summarize(root=ROOT):
    rows = [json.loads((root / 'ranks' / ('vision_rank%d.json' % rank)).read_text()) for rank in range(8)]
    previous = [json.loads((root.parent / 'round70' / 'ranks' / ('encoder_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources']) != {'vision_encode_tp8.py', 'vision_linear71.py', 'candidate_padded_text_encoder.py', 'evidence_io71.py'}:
            raise ValueError('Missing exact executed-source fingerprints')
        for name, digest in row['sources'].items():
            path = root / name
            if name == 'candidate_padded_text_encoder.py':
                path = root.parent / 'round65' / 'source' / name
            elif name == 'evidence_io71.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Vision experiment executed-source fingerprint mismatch')
    return aggregate(rows, previous)


if __name__ == '__main__':
    result = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
