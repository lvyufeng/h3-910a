#!/usr/bin/env python3
"""Attribute profile forwards without adding parent/child or different ranks."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
TOP_COUNTS = {'top.local_q': 50, 'top.local_kv_padded': 100, 'top.local_gate_up': 50,
              'top.row_o': 50, 'top.row_down': 50, 'top.norm_q_head': 50, 'top.norm_k_head': 50,
              'top.norm_hidden_input': 50, 'top.norm_hidden_residual': 50,
              'top.rope': 50, 'top.causal_fp32': 50, 'top.swiglu': 50, 'top.deepstack': 3}
CHILD_COUNTS = {'child.linear_1024x5120': 150, 'child.linear_6400x5120': 50,
                'child.linear_5120x1024': 50, 'child.linear_5120x3200': 50, 'child.all_reduce': 100}


def group_events(profile, combined):
    expected = {**TOP_COUNTS, **CHILD_COUNTS}
    if combined:
        expected['top.shared_activation_cast'] = 50
    if Counter(event['name'] for event in profile['events']) != expected:
        raise ValueError('Missing/duplicate/unexpected actual component event counts')
    if profile['stream_begin'] != profile['stream_end']:
        raise ValueError('Mixed current streams in profile')
    grouped = defaultdict(lambda: {'count': 0, 'current_stream_elapsed_s': 0., 'host_dispatch_s': 0.})
    for event in profile['events']:
        elapsed, host = event['current_stream_elapsed_ms'] / 1000, event['host_dispatch_s']
        if not all(math.isfinite(value) and value >= 0 for value in (elapsed, host)):
            raise ValueError('Invalid event timing')
        value = grouped[event['name']]
        value['count'] += 1
        value['current_stream_elapsed_s'] += elapsed
        value['host_dispatch_s'] += host
    top_sum = sum(value['current_stream_elapsed_s'] for name, value in grouped.items() if name.startswith('top.'))
    return {'categories': dict(sorted(grouped.items())), 'top_nonoverlap_event_sum_s': top_sum,
            'profile_wall_s': profile['seconds'], 'wall_minus_top_event_sum_s': profile['seconds'] - top_sum,
            'interpretation': 'Current-stream elapsed intervals can include stalls/host gaps/dependency waits. child.* is nested and NEVER added to top.*; host wall overlaps device intervals and is NEVER added to them. Difference is not assigned to CPU or network.'}


def rank_max(cases, phase):
    return [max(case[phase][i]['seconds'] for case in cases) for i in range(3)]


def summarize(root=ROOT):
    rows = [json.loads((root / 'ranks' / ('profile_rank%d.json' % rank)).read_text()) for rank in range(8)]
    if any(row['rank'] != rank or not row.get('completed') or row.get('failure') for rank, row in enumerate(rows)):
        raise ValueError('Need complete successful eight-rank receipts')
    for row in rows:
        if set(row['sources']) != {'text_profile_tp8.py', 'event_profile66.py', 'candidate_padded_text_encoder.py'}:
            raise ValueError('Missing executed source identities')
        for name, digest in row['sources'].items():
            source = root / name if name != 'candidate_padded_text_encoder.py' else root.parent / 'round65' / 'source' / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('Executed source mismatch')
        if row['parameters_before'] != row['parameters_after'] or row['inputs_before'] != row['inputs_after'] or row['inputs_before'] != rows[0]['inputs_before']:
            raise ValueError('Changed/mismatched text input or weights')
        if not row['baseline_vs_round60_full_output']['raw_bits_equal']:
            raise ValueError('Lost stable baseline')
    r = {'scope': 'Same resident50-layer experimental TP8 text-only stack as Round65, actual2043-token captured inputs. Not full encoder stage, originalBF16/HF quality or official inference proof.',
         'all_inputs_and_weights_immutable': True, 'all_complete_outputs_raw_bits_equal': True,
         'cases': {}, 'profile_not_a_new_optimization': True, 'official_goal_complete': False}
    hashes = set()
    for mode in ('baseline', 'shared_qkv', 'restored'):
        cases = [row['cases'][mode] for row in rows]
        for case in cases:
            first = case['first_vs_baseline']
            if (not first['raw_bits_equal'] or not first['finite'] or first['elements'] != 10460160
                    or first['max_abs_error'] != 0 or first['relative_rms'] != 0 or len(case['clean_before']) != 3):
                raise ValueError('Wrong first output or missing clean repeats')
            if len(case['profiles']) != (0 if mode == 'restored' else 3) or len(case['clean_after']) != (0 if mode == 'restored' else 3):
                raise ValueError('Missing profiled/after repeats')
            for phase in ('clean_before', 'profiles', 'clean_after'):
                for call in case[phase]:
                    proof = call['vs_baseline']
                    if not proof['raw_bits_equal'] or not proof['finite'] or proof['elements'] != 10460160 or proof['max_abs_error'] != 0 or proof['relative_rms'] != 0:
                        raise ValueError('Full text output changed or incomplete')
                    hashes.add(call['output_sha256'])
            if not case['shared_activation_scope_closed']:
                raise ValueError('Leaked activation cache scope')
        clean_before = rank_max(cases, 'clean_before')
        report = {'clean_before_rank_max_s': clean_before, 'median_clean_before_s': statistics.median(clean_before),
                  'restored_math_only_shadows_still_resident': mode == 'restored'}
        if mode != 'restored':
            after = rank_max(cases, 'clean_after')
            profile = rank_max(cases, 'profiles')
            report.update(clean_after_rank_max_s=after, median_clean_after_s=statistics.median(after),
                          profile_rank_max_s=profile, median_profile_s=statistics.median(profile),
                          profile_wall_overhead_vs_clean_before=statistics.median(profile) / statistics.median(clean_before) - 1,
                          clean_after_drift_vs_before=statistics.median(after) / statistics.median(clean_before) - 1)
            # Preserve EACH actual rank vector. Representative vectors come from
            # the rank with the slowest complete wall of that SAME forward.
            all_vectors = []
            for i in range(3):
                vectors = [group_events(case['profiles'][i], mode == 'shared_qkv') for case in cases]
                slow = max(range(8), key=lambda rank: cases[rank]['profiles'][i]['seconds'])
                all_vectors.append({'profile_call': i, 'slowest_wall_rank': slow, 'representative': vectors[slow],
                                    'all_rank_vectors': vectors})
            report['profiles'] = all_vectors
        r['cases'][mode] = report
    if len(hashes) != 1:
        raise ValueError('Final output hash changed across ranks/calls/modes')
    r['all_complete_text_output_sha256'] = hashes.pop()
    r['clean_resident_speedup'] = r['cases']['baseline']['median_clean_before_s'] / r['cases']['shared_qkv']['median_clean_before_s']
    r['clean_resident_under2s'] = all(value < 2 for value in r['cases']['shared_qkv']['clean_before_rank_max_s'])
    return r


if __name__ == '__main__':
    result = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))
    for name, case in result['cases'].items():
        print(name, {key: value for key, value in case.items() if key != 'profiles'})
