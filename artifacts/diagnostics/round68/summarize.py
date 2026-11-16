#!/usr/bin/env python3
"""Strict same-input prefix/statistics and clean/event-profile aggregation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('profile_contract66', ROOT.parent / 'round66' / 'summarize.py')
profile_contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_contract)
SHAPES = {'hidden_input': [1, 2043, 5120], 'hidden_residual': [1, 2043, 5120],
          'q_head': [1, 2043, 8, 128], 'k_head': [1, 2043, 1, 128]}


def exact(proof, elements):
    return (proof['raw_bits_equal'] and proof['finite'] and proof['elements'] == elements
            and proof['max_abs_error'] == 0 and proof['relative_rms'] == 0)


def aggregate(rows):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique rank receipts')
    if any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Failed/incomplete prefix experiment cannot be accepted')
    if any(row['inputs_before'] != row['inputs_after'] or row['parameters_before'] != row['parameters_after']
           or row['inputs_before'] != rows[0]['inputs_before'] for row in rows):
        raise ValueError('Inputs/weights changed or mismatched')
    if any(not exact(row['baseline_vs_round60_full_output'], 10460160) for row in rows):
        raise ValueError('Prior stable full text output lost')
    result = {'scope': 'Actual2043-token50-layer cached resident experimental text-only TP8, Round67 exact-tail baseline. Native mean/reduction/eps/rsqrt fixed. Not full service, originalBF16/HF or official generation quality.',
              'all_inputs_and_parameter_bytes_immutable': True, 'cases': {}, 'actual_prefix_checks': 0,
              'actual_residual_checks': 0, 'production_deployed': False, 'whole_service_text_under2s_proven': False,
              'official_goal_complete': False}
    hashes = set()
    for mode in ('native', 'square', 'add_square', 'restored'):
        cases = [row['cases'][mode] for row in rows]
        for case in cases:
            if case['stats_mode'] != mode or not exact(case['first_vs_baseline'], 10460160):
                raise ValueError('Wrong mode or first output')
            if len(case['clean_before']) != 3 or len(case['profiles']) != (0 if mode == 'restored' else 3) or len(case['clean_after']) != (0 if mode == 'restored' else 3):
                raise ValueError('Missing clean/profile/repeat calls')
            for phase in ('clean_before', 'profiles', 'clean_after'):
                for call in case[phase]:
                    if not exact(call['vs_baseline'], 10460160):
                        raise ValueError('Complete final text tensor changed')
                    hashes.add(call['output_sha256'])
            checks = case['prefix_checks']
            if mode in ('native', 'restored'):
                if checks:
                    raise ValueError('Unexpected native first-call prefix double computation')
            else:
                expected = {'%02d.%s' % (layer, kind) for layer in range(50) for kind in SHAPES}
                if len(checks) != 200 or {check['name'] for check in checks} != expected:
                    raise ValueError('Missing actual Norm prefix layers/families')
                for check in checks:
                    family = check['name'].split('.')[1]
                    shape = SHAPES[family]
                    elements = 1
                    for dimension in shape:
                        elements *= dimension
                    stats_elements = elements // shape[-1]
                    if check['shape'] != shape or check['square_layout'] != check['native_square_layout']:
                        raise ValueError('Mean input shape/stride/format changed')
                    if check['square_layout'][0] != shape or check['square_layout'][2] != 'torch.float32' or check['square_layout'][3] not in (0, 2):
                        raise ValueError('Square is not the original logical FP32 reduction input')
                    for key, count in (('square', elements), ('variance', stats_elements), ('inverse', stats_elements), ('output', elements)):
                        if not exact(check[key], count):
                            raise ValueError('Actual prefix/statistics/output proof failed: ' + key)
                    if family == 'hidden_residual':
                        if not exact(check['residual'], elements):
                            raise ValueError('Original BF16 residual update changed')
                        result['actual_residual_checks'] += 1
                    elif 'residual' in check:
                        raise ValueError('Unexpected residual path')
                result['actual_prefix_checks'] += len(checks)
            if case['shadow_cache']['entries'] != 200 or case['shadow_cache']['cached_fp32_bytes'] != 11927552000:
                raise ValueError('Fixed shadow stack changed')
            if case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000}:
                raise ValueError('Stable K/V padding changed')
        before = [max(case['clean_before'][i]['seconds'] for case in cases) for i in range(3)]
        report = {'clean_before_rank_max_s': before, 'median_clean_before_s': statistics.median(before),
                  'complete_outputs_checked': len(cases) * (4 if mode == 'restored' else 10),
                  'all_clean_resident_calls_under2s': all(value < 2 for value in before)}
        if mode != 'restored':
            after = [max(case['clean_after'][i]['seconds'] for case in cases) for i in range(3)]
            profiled = [max(case['profiles'][i]['seconds'] for case in cases) for i in range(3)]
            report.update(clean_after_rank_max_s=after, median_clean_after_s=statistics.median(after),
                          profile_rank_max_s=profiled, median_profile_s=statistics.median(profiled),
                          profile_overhead_vs_before=statistics.median(profiled) / statistics.median(before) - 1,
                          clean_after_drift=statistics.median(after) / statistics.median(before) - 1)
            vectors = []
            for i in range(3):
                all_rank = [profile_contract.group_events(case['profiles'][i], True) for case in cases]
                slow = max(range(8), key=lambda rank: cases[rank]['profiles'][i]['seconds'])
                vectors.append({'profile_call': i, 'slowest_wall_rank': slow, 'representative': all_rank[slow], 'all_rank_vectors': all_rank})
            report['profiles'] = vectors
        result['cases'][mode] = report
    if len(hashes) != 1:
        raise ValueError('Across-rank/repeat/mode final tensor hash differs')
    result['all_complete_text_output_sha256'] = hashes.pop()
    baseline = result['cases']['native']['median_clean_before_s']
    for case in result['cases'].values():
        case['speedup_vs_same_run_native_prefix'] = baseline / case['median_clean_before_s']
    return result


def summarize(root=ROOT):
    rows = [json.loads((root / 'ranks' / ('prefix_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources']) != {'text_prefix_tp8.py', 'norm_prefix68.py', 'candidate_padded_text_encoder.py', 'evidence_io68.py'}:
            raise ValueError('Missing executed source fingerprints')
        for name, digest in row['sources'].items():
            path = root / name
            if name == 'candidate_padded_text_encoder.py':
                path = root.parent / 'round65' / 'source' / name
            elif name == 'evidence_io68.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Executed source fingerprint changed')
    result = aggregate(rows)
    validation = json.loads((root / 'validation01.json').read_text())
    if not validation.get('completed') or validation.get('failure'):
        raise ValueError('Standalone native prefix gate failed/incomplete')
    for name, digest in validation['sources'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Standalone validation source changed')
    expected = {(rows, width, active) for rows, width in ((17, 128), (9, 5120), (2043, 128), (2043, 5120)) for active in (False, True)}
    if len(validation['cases']) != 8 or {(case['shape'][0], case['shape'][1], case['residual']) for case in validation['cases']} != expected:
        raise ValueError('Missing bounded native square/add fixture')
    for case in validation['cases']:
        for key in ('square', 'updated'):
            proof = case[key]
            if not proof['raw_bits_equal'] or not proof['finite'] or proof['different_values'] or proof['elements'] != case['shape'][0] * case['shape'][1]:
                raise ValueError('Standalone native prefix output changed')
        if case['square_shape_stride'] != case['native_shape_stride']:
            raise ValueError('Standalone square reduction layout changed')
    all_words = validation['all_bf16_words_square']
    bounded = all_words['bounded_finite_inputs_abs_le_2pow60']
    if (not bounded['raw_bits_equal'] or not bounded['finite'] or bounded['elements'] != 47874
            or bounded['dtype'] != 'torch.float32' or bounded['different_values'] != 0):
        raise ValueError('Bounded finite all-word prefix square gate failed')
    if all_words['all_patterns']['elements'] != 65536 or not all_words['all_patterns']['raw_bits_equal']:
        raise ValueError('Retained square-only all-word raw identity changed')
    readonly = json.loads((root / 'validation02.json').read_text())
    if not readonly.get('completed') or readonly.get('failure') or readonly['exceptional_add_equivalence_claimed']:
        raise ValueError('Readonly/extreme add control incomplete or exceptional claim expanded')
    for name, digest in readonly['sources'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Readonly/extreme probe source changed')
    expected = {(name, active) for name in ('bounded', 'extreme_pairs') for active in (False, True)}
    if len(readonly['cases']) != 4 or {(case['name'], case['residual']) for case in readonly['cases']} != expected:
        raise ValueError('Missing readonly or extreme add fixture')
    for case in readonly['cases']:
        if not case['x_readonly'] or not case['residual_readonly'] or not case['outputs_aligned_dense'] or any(fmt not in (0, 2) for fmt in case['output_formats']):
            raise ValueError('Source mutation or unsafe output format in native prefix probe')
        if case['name'] == 'bounded' and (not case['square_raw_bits_equal'] or not case['updated_raw_bits_equal']):
            raise ValueError('Bounded native update/square control failed')
        if case['name'] == 'extreme_pairs' and case['residual'] and (case['square_raw_bits_equal'] or case['updated_raw_bits_equal']):
            raise ValueError('Retained exceptional NaN-add payload counterexample was hidden')
    result['standalone_all_bf16_square_probe'] = all_words
    result['readonly_inputs_and_output_format_guards_passed'] = True
    result['exceptional_add_payload_identity_not_accepted'] = True
    return result


if __name__ == '__main__':
    r = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps({key: value for key, value in r.items() if key != 'cases'}, indent=2))
    for name, case in r['cases'].items():
        print(name, {key: value for key, value in case.items() if key != 'profiles'})
