#!/usr/bin/env python3
"""Aggregate exact resident text-cache A/B without inventing service latency."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
MODES = ('baseline', 'activation_only', 'weight_shadow', 'shared_qkv', 'restored')


def aggregate(rows):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique rank receipts')
    if any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Incomplete/failed run cannot be accepted')
    if any(row['parameters_before'] != row['parameters_after'] or row['inputs_before'] != row['inputs_after'] for row in rows):
        raise ValueError('Input or parameter bytes changed')
    if any(row['inputs_before'] != rows[0]['inputs_before'] for row in rows):
        raise ValueError('Actual replicated text inputs differ across ranks')
    if any(not row['baseline_vs_round60_full_output']['raw_bits_equal'] for row in rows):
        raise ValueError('Baseline does not reproduce prior stable complete text output')
    result = {'scope': '50-layer resident experimental text-only TP8, actual captured2043-token inputs. Excludes processor/vision/offload/entire H3 service; not originalBF16/HF-oracle or official generation lossless proof.',
              'all_inputs_and_parameter_bytes_immutable': True, 'cases': {},
              'production_deployed': False, 'whole_service_text_under2s_proven': False, 'strict_goal_complete': False}
    final_hash = None
    for mode in MODES:
        cases = [row['cases'][mode] for row in rows]
        if any(len(case['warm_calls']) != 3 for case in cases):
            raise ValueError('Need three clean warm calls on every rank')
        if any(not case['first_vs_baseline']['raw_bits_equal'] or not case['first_vs_baseline']['finite'] for case in cases):
            raise ValueError('First candidate output differs')
        outputs = [call for case in cases for call in case['warm_calls']]
        if any(not call['vs_baseline']['raw_bits_equal'] or not call['vs_baseline']['finite']
               or call['vs_baseline']['elements'] != 10460160 or call['vs_baseline']['max_abs_error'] != 0.
               or call['vs_baseline']['relative_rms'] != 0. for call in outputs):
            raise ValueError('Candidate warm tensor differs or incomplete element coverage')
        hashes = {call['output_sha256'] for call in outputs}
        if len(hashes) != 1:
            raise ValueError('Rank/repeat output identity failed')
        observed = hashes.pop()
        if final_hash is None:
            final_hash = observed
        if observed != final_hash:
            raise ValueError('Cross-mode final tensor hash mismatch')
        checks = [check for case in cases for check in case['projection_checks']]
        expected = 200 if mode == 'activation_only' else (300 if mode in ('weight_shadow', 'shared_qkv') else 0)
        if any(len(case['projection_checks']) != expected for case in cases):
            raise ValueError('Incomplete same-input projection checks: ' + mode)
        if any(not check['raw_bits_equal'] or not check['finite'] or check['max_abs_error'] != 0.
               or check['relative_rms'] != 0. for check in checks):
            raise ValueError('Same-input local/FP32 partial identity failed')
        shadow_bytes = [case['shadow_cache']['cached_fp32_bytes'] for case in cases]
        shadow_entries = [case['shadow_cache']['entries'] for case in cases]
        if mode in ('weight_shadow', 'shared_qkv', 'restored'):
            if shadow_entries != [200] * 8 or shadow_bytes != [11927552000] * 8:
                raise ValueError('Unexpected shadow footprint / bypassing / cache thrash')
        elif any(shadow_entries):
            raise ValueError('Baseline/activation-only unexpectedly allocated FP32 weight shadows')
        if any(case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000} for case in cases):
            raise ValueError('Stable K/V geometry/cache changed')
        if any(not case['shared_activation']['scope_closed'] for case in cases):
            raise ValueError('Activation reference leaked across a forward')
        groups = [max(case['warm_calls'][i]['seconds'] for case in cases) for i in range(3)]
        result['cases'][mode] = {'warm_per_call_rank_max_s': groups, 'median_rank_max_s': statistics.median(groups),
                                 'first_forward_rank_max_s_checks_and_cache_excluded': max(case['first_s'] for case in cases),
                                 'full_output_raw_bits_equal': True, 'full_output_elements_per_rank': outputs[0]['vs_baseline']['elements'],
                                 'same_input_projection_checks': len(checks), 'projection_scalar_outputs_checked': sum(check['elements'] for check in checks),
                                 'nonkv_shadow_bytes_per_rank': shadow_bytes, 'nonkv_shadow_entries_per_rank': shadow_entries,
                                 'hbm_allocated_bytes_per_rank': [case['hbm_allocated_bytes'] for case in cases],
                                 'hbm_reserved_bytes_per_rank': [case['hbm_reserved_bytes'] for case in cases],
                                 'all_clean_resident_calls_under2s': all(value < 2 for value in groups)}
    result['final_tensor_sha256_all_modes_ranks_repeats'] = final_hash
    baseline = result['cases']['baseline']['median_rank_max_s']
    for case in result['cases'].values():
        case['speedup_vs_baseline'] = baseline / case['median_rank_max_s']
    result['baseline_restored_timing_drift'] = result['cases']['restored']['median_rank_max_s'] / baseline - 1
    return result


def summarize(root=ROOT):
    rows = [json.loads((root / 'ranks' / ('shadow_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        for name, digest in row['sources'].items():
            path = root / ('source/candidate_padded_text_encoder.py' if name == 'candidate_padded_text_encoder.py' else name)
            if name == 'evidence_io65.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Executed source fingerprint differs: ' + name)
    return aggregate(rows)


if __name__ == '__main__':
    result = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
