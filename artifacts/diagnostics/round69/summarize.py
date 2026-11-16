#!/usr/bin/env python3
"""Aggregate actual activation cast/layout and full-text identity/performance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
MODES = ('native', 'shared_only', 'fallback_only', 'both', 'restored')


def exact(proof, elements=None):
    return (proof['raw_bits_equal'] and proof['finite'] and proof['max_abs_error'] == 0
            and proof['relative_rms'] == 0 and (elements is None or proof['elements'] == elements))


def aggregate(rows):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique rank receipts')
    if any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Failed/incomplete cast A/B cannot be accepted')
    if any(row['inputs_before'] != row['inputs_after'] or row['parameters_before'] != row['parameters_after']
           or row['inputs_before'] != rows[0]['inputs_before'] for row in rows):
        raise ValueError('Inputs or weights changed/mismatched')
    if any(not exact(row['baseline_vs_round60_full_output'], 10460160) for row in rows):
        raise ValueError('Stable complete text baseline lost')
    result = {'scope': 'Actual2043-token,50-layer cached resident experimental text-only TP8, fixed Round68add_square/native.mean+rsqrt/Round67tail, shadows/KVpadding/causal/SwiGLU/RoPE. Not full production encoder, originalBF16/HF/official generation quality.',
              'all_input_and_parameter_bytes_immutable': True, 'cases': {}, 'same_input_cast_checks': 0,
              'same_input_projection_checks': 0, 'production_deployed': False,
              'whole_service_text_under2s_proven': False, 'official_goal_complete': False}
    hashes = set()
    for mode in MODES:
        cases = [row['cases'][mode] for row in rows]
        for case in cases:
            if case['cast_mode'] != mode or not exact(case['first_vs_baseline'], 10460160) or len(case['warm_calls']) != 3:
                raise ValueError('Wrong candidate mode/first complete output/repeat count')
            for call in case['warm_calls']:
                if not exact(call['vs_baseline'], 10460160):
                    raise ValueError('Complete final text tensor identity failed')
                hashes.add(call['output_sha256'])
            if mode in ('native', 'restored'):
                if case['cast_checks'] or case['projection_checks']:
                    raise ValueError('Unexpected first-call double-compute in baseline/restored')
            else:
                if len(case['cast_checks']) != 200 or Counter(check['kind'] for check in case['cast_checks']) != {'shared': 50, 'fallback': 150}:
                    raise ValueError('Missing actual casts or changing shared-hit behavior')
                for check in case['cast_checks']:
                    enabled = mode in ('shared_only', 'both') if check['kind'] == 'shared' else mode in ('fallback_only', 'both')
                    shape = check['input_shape']
                    elements = 1
                    for dimension in shape:
                        elements *= dimension
                    if check['candidate_enabled'] != enabled or check['input_dtype'] != 'torch.bfloat16' or not check['input_readonly']:
                        raise ValueError('Wrong cast mode/type or input mutation')
                    if not exact(check['proof'], elements) or check['candidate_layout'] != check['native_layout']:
                        raise ValueError('Actual cast raw bytes/layout mismatch')
                    if check['candidate_layout'][0] != shape or check['candidate_layout'][2] != 'torch.float32' or check['candidate_layout'][3] not in (0, 2):
                        raise ValueError('Wrong actual projection operand layout')
                cast_shapes = Counter((check['kind'], tuple(check['input_shape'])) for check in case['cast_checks'])
                expected_cast_shapes = {('shared', (1, 2043, 5120)): 50,
                                        ('fallback', (1, 2043, 5120)): 50,
                                        ('fallback', (1, 2043, 1024)): 50,
                                        ('fallback', (1, 2043, 3200)): 50}
                if cast_shapes != expected_cast_shapes:
                    raise ValueError('Actual shared/gate-up/O/down operand shape coverage changed')
                projections = case['projection_checks']
                if len(projections) != 300 or Counter(check['kind'] for check in projections) != {'local': 200, 'row_partial_fp32': 100}:
                    raise ValueError('Missing actual local/FP32-row projection proof')
                slots = Counter(tuple(check['slot']) if isinstance(check['slot'], list) else check['slot'] for check in projections)
                expected_slots = {('q', layer): 1 for layer in range(50)}
                expected_slots.update({(kind, layer): 1 for kind in ('gate_up', 'o', 'down') for layer in range(50)})
                expected_slots['padded_kv'] = 100
                if slots != expected_slots:
                    raise ValueError('Actual projection layer/slice coverage changed')
                output_elements = {'q': 2092032, 'gate_up': 13075200, 'o': 10460160, 'down': 10460160, 'padded_kv': 261504}
                for check in projections:
                    slot = check['slot'][0] if isinstance(check['slot'], list) else check['slot']
                    expected_kind = 'row_partial_fp32' if slot in ('o', 'down') else 'local'
                    if check['kind'] != expected_kind or not exact(check, output_elements[slot]):
                        raise ValueError('Same-input projection output/element coverage changed')
                result['same_input_cast_checks'] += len(case['cast_checks'])
                result['same_input_projection_checks'] += len(projections)
            if not case['activation_scope_closed']:
                raise ValueError('Activation scope leaked')
            if case['shadow_cache']['entries'] != 200 or case['shadow_cache']['cached_fp32_bytes'] != 11927552000:
                raise ValueError('Fixed shadow stack changed')
            if case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000}:
                raise ValueError('Stable K/V padding changed')
        maxima = [max(case['warm_calls'][i]['seconds'] for case in cases) for i in range(3)]
        result['cases'][mode] = {'warm_per_call_rank_max_s': maxima, 'median_rank_max_s': statistics.median(maxima),
                                 'first_rank_max_s_excluded_checks_compilation': max(case['first_s_excluded_checks_compilation'] for case in cases),
                                 'complete_outputs_checked': 32, 'all_complete_outputs_raw_bits_equal': True,
                                 'allocated_hbm_bytes_per_rank': [case['hbm_allocated_bytes'] for case in cases],
                                 'reserved_hbm_bytes_per_rank': [case['hbm_reserved_bytes'] for case in cases],
                                 'all_clean_resident_calls_under2s': all(value < 2 for value in maxima)}
    if len(hashes) != 1:
        raise ValueError('Across-mode/rank/repeat complete output hash changed')
    result['all_complete_text_output_sha256'] = hashes.pop()
    baseline = result['cases']['native']['median_rank_max_s']
    for case in result['cases'].values():
        case['speedup_vs_same_run_native_cast'] = baseline / case['median_rank_max_s']
    result['restored_timing_drift'] = result['cases']['restored']['median_rank_max_s'] / baseline - 1
    return result


def summarize(root=ROOT):
    rows = [json.loads((root / 'ranks' / ('cast_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources']) != {'text_cast_tp8.py', 'activation_cast69.py', 'candidate_padded_text_encoder.py', 'evidence_io69.py'}:
            raise ValueError('Missing executed source identities')
        for name, digest in row['sources'].items():
            path = root / name
            if name == 'candidate_padded_text_encoder.py':
                path = root.parent / 'round65' / 'source' / name
            elif name == 'evidence_io69.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Executed source mismatch')
    result = aggregate(rows)
    v = json.loads((root / 'validation01.json').read_text())
    if not v.get('completed') or v.get('failure'):
        raise ValueError('Standalone native cast gate incomplete')
    for name, digest in v['sources'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Standalone validation source mismatch')
    proof = v['all_bf16_words']
    if proof['elements'] != 65536 or not proof['matches_bit_expansion_oracle'] or not proof['matches_native_cast'] or proof['native_layout'] != proof['candidate_layout']:
        raise ValueError('Exhaustive BF16 conversion oracle/native/layout gate failed')
    expected = {(1, 2043, width) for width in (5120, 1024, 3200)}
    if len(v['cases']) != 3 or {tuple(case['shape']) for case in v['cases']} != expected:
        raise ValueError('Missing actual projection operand shapes')
    for case in v['cases']:
        if not case['raw_bits_equal'] or not case['input_readonly'] or case['native_layout'] != case['candidate_layout'] or len(case['native_s']) != 4 or len(case['candidate_s']) != 4:
            raise ValueError('Actual standalone operand conversion check failed')
    if not v['contiguous_unaligned_view_rejected']:
        raise ValueError('Unsafe contiguous GM offset was not rejected')
    result['all65536_bf16_words_native_and_integer_oracle_equal'] = True
    result['standalone_cast_cases'] = v['cases']
    return result


if __name__ == '__main__':
    r = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2))
