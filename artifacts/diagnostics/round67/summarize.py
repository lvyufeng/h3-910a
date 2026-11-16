#!/usr/bin/env python3
"""Strict retained proof aggregation for native-statistics Norm-tail A/B."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SHAPES = {'hidden_input': [1, 2043, 5120], 'hidden_residual': [1, 2043, 5120],
          'q_head': [1, 2043, 8, 128], 'k_head': [1, 2043, 1, 128]}


def exact(proof, elements=None):
    return (proof['raw_bits_equal'] and proof['finite'] and proof['max_abs_error'] == 0.
            and proof['relative_rms'] == 0. and (elements is None or proof['elements'] == elements))


def aggregate(rows):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique real-rank receipts')
    if any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Failed/incomplete text run must remain rejected')
    if any(row['parameters_before'] != row['parameters_after'] or row['inputs_before'] != row['inputs_after']
           or row['inputs_before'] != rows[0]['inputs_before'] for row in rows):
        raise ValueError('Actual text input/weights changed or mismatched')
    if any(not exact(row['baseline_vs_round60_full_output'], 10460160) for row in rows):
        raise ValueError('Stable complete text baseline lost')
    result = {'scope': 'Actual captured2043-token,50-layer resident experimental text-only TP8 with fixed Round65 FP32 shadows/shared activation, paddedKV, FP32 causal and native-round TBE SwiGLU/RoPE. Not production full encode, originalBF16/HF quality, or official49-step generation.',
              'all_input_and_parameter_bytes_immutable': True, 'cases': {}, 'actual_norm_identity_checks': 0,
              'actual_residual_identity_checks': 0,
              'whole_service_text_under2s_proven': False, 'production_deployed': False, 'official_goal_complete': False}
    output_hashes = set()
    for label in ('native_stats_baseline', 'native_stats_tail', 'native_restored'):
        cases = [row['cases'][label] for row in rows]
        active = label == 'native_stats_tail'
        for case in cases:
            if case['tail_enabled'] != active or not exact(case['first_vs_baseline'], 10460160) or len(case['warm_calls']) != 3:
                raise ValueError('Wrong actual candidate mode/first tensor/repeat count')
            for call in case['warm_calls']:
                if not exact(call['vs_baseline'], 10460160):
                    raise ValueError('Complete final text output identity failed')
                output_hashes.add(call['output_sha256'])
            checks = case['norm_checks']
            if not active:
                if checks:
                    raise ValueError('Unexpected reference-double-compute in baseline/restored mode')
            else:
                names = {check['name'] for check in checks}
                expected = {'%02d.%s' % (layer, kind) for layer in range(50) for kind in SHAPES}
                if len(checks) != 200 or names != expected:
                    raise ValueError('Need all200 actual pre-Norm checks per rank')
                for check in checks:
                    kind = check['name'].split('.')[1]
                    shape = SHAPES[kind]
                    elements = 1
                    for dimension in shape:
                        elements *= dimension
                    if check['shape'] != shape or not check['aligned_input_gamma_inverse'] or not exact(check['output'], elements):
                        raise ValueError('Actual Norm input shape/alignment/output proof failed')
                    if kind == 'hidden_residual':
                        if not exact(check['residual'], elements):
                            raise ValueError('Native residual update changed')
                        result['actual_residual_identity_checks'] += 1
                    elif 'residual' in check:
                        raise ValueError('Unexpected residual path')
                result['actual_norm_identity_checks'] += len(checks)
            if case['shadow_cache']['entries'] != 200 or case['shadow_cache']['cached_fp32_bytes'] != 11927552000:
                raise ValueError('Fixed FP32 shadow stack/memory changed')
            if case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000}:
                raise ValueError('Stable padded K/V geometry changed')
        maxima = [max(case['warm_calls'][i]['seconds'] for case in cases) for i in range(3)]
        result['cases'][label] = {'warm_per_call_rank_max_s': maxima, 'median_rank_max_s': statistics.median(maxima),
                                 'first_rank_max_s_excluded_checks_compilation': max(case['first_s_excluded_contains_checks_and_compilation'] for case in cases),
                                 'all_full_outputs_raw_bits_equal': True, 'complete_outputs_checked': 4 * len(rows),
                                 'allocated_hbm_bytes_per_rank': [case['hbm_allocated_bytes'] for case in cases],
                                 'reserved_hbm_bytes_per_rank': [case['hbm_reserved_bytes'] for case in cases],
                                 'all_clean_resident_calls_under2s': all(value < 2 for value in maxima)}
    if len(output_hashes) != 1:
        raise ValueError('Across-rank/repeat/mode full output hash differs')
    result['all_modes_ranks_repeats_complete_output_sha256'] = output_hashes.pop()
    result['speedup_vs_same_run_cached_baseline'] = result['cases']['native_stats_baseline']['median_rank_max_s'] / result['cases']['native_stats_tail']['median_rank_max_s']
    result['restored_timing_drift'] = result['cases']['native_restored']['median_rank_max_s'] / result['cases']['native_stats_baseline']['median_rank_max_s'] - 1
    return result


def summarize(root=ROOT):
    rows = [json.loads((root / 'ranks' / ('text_norm_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources']) != {'text_norm_tp8.py', 'norm_tail_guard67.py', 'candidate_padded_text_encoder.py', 'evidence_io67.py'}:
            raise ValueError('Missing executed-source fingerprints')
        for name, digest in row['sources'].items():
            path = root / name
            if name == 'candidate_padded_text_encoder.py':
                path = root.parent / 'round65' / 'source' / name
            elif name == 'evidence_io67.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Executed source differs: ' + name)
    result = aggregate(rows)
    for name in ('validation01', 'validation02'):
        proof = json.loads((root / (name + '.json')).read_text())
        if not proof.get('completed') or proof.get('failure'):
            raise ValueError('Standalone native numerical/boundary gate incomplete')
        for source, digest in proof['source_sha256'].items():
            if hashlib.sha256((root / source).read_bytes()).hexdigest() != digest:
                raise ValueError('Standalone validation source changed')
    numerical = json.loads((root / 'validation01.json').read_text())
    expected_shapes = {(17, 128), (9, 5120), (2043, 128), (2043, 5120)}
    if len(numerical['cases']) != 4 or {tuple(case['shape']) for case in numerical['cases']} != expected_shapes:
        raise ValueError('Missing standalone native-statistics fixture shapes')
    for case in numerical['cases']:
        proof = case['proof']
        if (not proof['raw_bits_equal'] or not proof['all_finite'] or proof['different_raw_words'] != 0
                or proof['max_abs_error'] != 0 or proof['elements'] != case['shape'][0] * case['shape'][1]
                or len(case['warm_s']) != 3):
            raise ValueError('Standalone actual finite fixture proof failed')
    exhaustive = numerical['all_bf16_words']
    finite = exhaustive['finite_inputs']
    if (not finite['raw_bits_equal'] or finite['elements'] != 65280 or finite['different_raw_words'] != 0
            or not finite['all_finite'] or finite['max_abs_error'] != 0):
        raise ValueError('Exhaustive finite BF16 point probe failed')
    if exhaustive['all_inputs']['elements'] != 65536 or exhaustive['all_inputs']['raw_bits_equal'] or exhaustive['all_inputs']['different_raw_words'] != 252:
        raise ValueError('Retained NaN-payload identity counterexample was lost or changed')
    boundaries = json.loads((root / 'validation02.json').read_text())
    expected_boundaries = {'normalized_round_below_tie_above_both_signs': [3, 128],
                           'gamma_product_all128x128_mantissa_pairs': [128, 128],
                           'gamma_product_negative128x128_mantissa_pairs': [128, 128]}
    if len(boundaries['cases']) != 3 or {case['name'] for case in boundaries['cases']} != set(expected_boundaries):
        raise ValueError('Missing/duplicate native rounding-boundary fixtures')
    for case in boundaries['cases']:
        shape = expected_boundaries[case['name']]
        if not case['raw_bits_equal'] or case['different_raw_words'] != 0 or case['shape'] != shape or case['elements'] != shape[0] * shape[1]:
            raise ValueError('Native RNE-boundary checks failed')
    expected_guards = {'contiguous_unaligned_x_rejected', 'contiguous_unaligned_gamma_rejected', 'wrong_broadcast_shape_same_numel_rejected'}
    if (len(boundaries['guards']) != 3 or {guard['name'] for guard in boundaries['guards']} != expected_guards
            or any(not guard['rejected'] for guard in boundaries['guards'])):
        raise ValueError('Unsafe GM/broadcast guards not demonstrated')
    counts = boundaries['final_product_boundaries']
    if (set(counts) != {'exact_midpoints', 'below_midpoint_nonexact', 'above_midpoint_nonexact'}
            or any(not isinstance(value, int) or value <= 0 for value in counts.values()) or sum(counts.values()) > 128 * 128):
        raise ValueError('Both native product-rounding boundary sides and midpoints must be covered')
    result['standalone_finite_bf16_word_identity_count'] = exhaustive['finite_inputs']['elements']
    result['nonfinite_payload_identity_not_accepted'] = not exhaustive['all_inputs']['raw_bits_equal']
    result['all_bf16_pattern_native_differences'] = exhaustive['all_inputs']['different_raw_words']
    result['gamma_product_boundary_counts'] = boundaries['final_product_boundaries']
    return result


if __name__ == '__main__':
    result = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
