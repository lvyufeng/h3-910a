#!/usr/bin/env python3
"""Independent vision event profile; parent/children and clean wall remain separate."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from summarize_vision71 import expected_modules, MAX_BYTES
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'round70'))
from summarize import exact, validate_identities

ROOT = Path(__file__).resolve().parent
MODES = ('native', 'both', 'restored')
PHASES = ('clean_before', 'event_profile', 'clean_after')


def category(name):
    return name if name == 'parent.vision' else name.split('.')[0]


def event_vector(records):
    counts = Counter(category(row['name']) for row in records)
    expected = {'parent.vision': 1, 'patch_embed': 1, 'pos_interpolate': 1, 'rot_pos': 1,
                'apply_rope': 27, 'attention': 54, 'linear': 116, 'norm': 58, 'gelu': 31}
    if counts != expected:
        raise ValueError('Missing or duplicate actual flat vision profile events')
    if Counter(row['name'][7:] for row in records if row['name'].startswith('linear.')) != Counter({name: 1 for name in expected_modules()}):
        raise ValueError('Missing actual vision Linear profile layers')
    norms = {f'blocks.{layer}.norm{side}' for layer in range(27) for side in (1, 2)} | {'merger.norm', *(f'deepstack_merger_list.{i}.norm' for i in range(3))}
    gelus = {f'blocks.{layer}.mlp.act_fn' for layer in range(27)} | {'merger.act_fn', *(f'deepstack_merger_list.{i}.act_fn' for i in range(3))}
    for prefix, names in (('norm.', norms), ('gelu.', gelus)):
        if Counter(row['name'][len(prefix):] for row in records if row['name'].startswith(prefix)) != Counter({name: 1 for name in names}):
            raise ValueError('Missing actual vision Norm/GELU profile layers')
    sums, host = {}, {}
    for row in records:
        if any(not math.isfinite(v) or v < 0 for v in (row['current_stream_elapsed_ms'], row['host_dispatch_s'])):
            raise ValueError('Invalid NPU event or host dispatch duration')
        name = category(row['name'])
        sums[name] = sums.get(name, 0.) + row['current_stream_elapsed_ms'] / 1000
        host[name] = host.get(name, 0.) + row['host_dispatch_s']
    parent = sums['parent.vision']
    flat = sum(value for name, value in sums.items() if name != 'parent.vision')
    return {'current_stream_category_s': sums, 'host_dispatch_category_s': host,
            'vision_parent_s': parent, 'flat_child_sum_s': flat,
            'vision_parent_minus_flat_children_s': parent - flat,
            'note': 'Parent contains children and is NOT added to their sum. Device events include dependencies/host gaps/stalls; host intervals are not additive with device events. Remainder is unassigned, not clamped.'}


def aggregate_profile(rows, main):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)) or any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Need eight unique successful profile ranks')
    rows = sorted(rows, key=lambda row: row['rank'])
    old = {row['rank']: row for row in main}
    reference = old[0]['cases']['native']['first_identities']
    identities, proofs = [], []
    result = {'scope': 'Unchanged-math isolated vision event profile after complete actual activation/shadow A/B. Three clean/profile/clean calls per mode; no per-op synchronization. Parent vision event is inclusive and never added to child sums. Complete wall includes encode_ids CPUoutput and excludes extra proof/event-readback plus upstreamHTTPprocessor/broadcast/load/offload.',
              'cases': {}, 'complete_output_identity_accepted': False, 'whole_service_text_under2s_proven': False,
              'official_goal_complete': False, 'production_deployed': False}
    for row in rows:
        previous = old[row['rank']]
        if (row['parameters_before'] != row['parameters_after'] or row['parameters_before'] != previous['parameters_before']
                or row['actual_inputs_before'] != row['actual_inputs_after'] or row['actual_inputs_before'] != previous['actual_inputs_before']):
            raise ValueError('Profile changed original input or parameter bytes')
        if row['math_flags_before'] != row['math_flags_after'] or row['math_flags_before'] != previous['math_flags_before']:
            raise ValueError('Profile changed actual matmul/conv precision settings')
        if not row['vision_wrappers_restored'] or not row['vision_profile_restored'] or row['vision_shadow_bytes_after_clear'] != 0:
            raise ValueError('Profile or vision state was not restored')
        if not exact(row['first_native_vs_round70_fresh_reference'], 10460160):
            raise ValueError('Profile native encoder reference differs from Round70')
        for field in ('checkpoint_root', 'checkpoint_index_sha256', 'component_parameter_bytes', 'parameter_dtypes', 'loaded_parameter_tensors', 'capture_file_sha256', 'environment'):
            if row[field] != previous[field]:
                raise ValueError('Profile original checkpoint/model/precision environment changed')
    for mode in MODES:
        cases = [row['cases'][mode] for row in rows]
        for case in cases:
            if case['vision_mode'] != mode or case['fixed_text_mode'] != 'shared_qkv_add_square_norm_tail_both_casts':
                raise ValueError('Profile mixed vision/text factors')
            validate_identities(case['first_identities'])
            identities.append(case['first_identities']); proofs.append(case['first_vs_baseline'])
            for phase in PHASES:
                if len(case[phase]) != 3:
                    raise ValueError('Need three full clean/profile/clean calls')
                for call in case[phase]:
                    validate_identities(call['identities'])
                    if call['output'] != call['identities']['complete_output'] or not math.isfinite(call['seconds']) or call['seconds'] <= 0:
                        raise ValueError('Invalid complete wall/CPU output receipt')
                    identities.append(call['identities']); proofs.append(call['vs_baseline'])
                    if phase == 'event_profile':
                        event_vector(call['event_records'])
                    elif 'event_records' in call:
                        raise ValueError('Clean call unexpectedly includes profile event recording')
            expected_cache = {'entries': 232 if mode == 'both' else 0, 'cached_fp32_bytes': MAX_BYTES if mode == 'both' else 0,
                              'misses': 232 if mode == 'both' else 0, 'hits': 2088 if mode == 'both' else 0,
                              'invalidations': 0, 'uncacheable_bypasses': 0, 'max_bytes': MAX_BYTES}
            if case['vision_shadow_cache'] != expected_cache:
                raise ValueError('Profile exact vision shadow cache policy changed')
            if case['text_shadow_cache']['entries'] != 200 or case['text_shadow_cache']['cached_fp32_bytes'] != 11927552000:
                raise ValueError('Profile fixed text shadow stack changed')
            if case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000}:
                raise ValueError('Profile fixed text padding geometry changed')
        maxima = {phase: [max(case[phase][i]['seconds'] for case in cases) for i in range(3)] for phase in PHASES}
        report = {phase + '_per_call_rank_max_s': values for phase, values in maxima.items()}
        report.update({phase + '_median_rank_max_s': statistics.median(values) for phase, values in maxima.items()})
        report['profile_perturbation_vs_clean_before'] = report['event_profile_median_rank_max_s'] / report['clean_before_median_rank_max_s'] - 1
        report['clean_after_drift_vs_before'] = report['clean_after_median_rank_max_s'] / report['clean_before_median_rank_max_s'] - 1
        vectors = []
        for i in range(3):
            per_rank = []
            for rank, case in enumerate(cases):
                call = case['event_profile'][i]
                per_rank.append({'rank': rows[rank]['rank'], 'complete_encode_ids_wall_s': call['seconds'], **event_vector(call['event_records'])})
            slow = max(range(8), key=lambda rank: per_rank[rank]['complete_encode_ids_wall_s'])
            vectors.append({'call': i, 'slowest_complete_wall_rank': rows[slow]['rank'], 'representative': per_rank[slow], 'all_rank_vectors': per_rank})
        report['profile_vectors'] = vectors
        result['cases'][mode] = report
    result['intermediate_identity_against_main_run'] = {name: all(item[name] == reference[name] for item in identities) for name in reference}
    result['complete_outputs_compared'] = len(identities)
    result['complete_output_identity_accepted'] = all(result['intermediate_identity_against_main_run'].values()) and all(exact(proof, 10460160) for proof in proofs)
    result['complete_encode_ids_under2s_accepted'] = result['complete_output_identity_accepted'] and all(v < 2 for v in result['cases']['both']['clean_before_per_call_rank_max_s'])
    result['profile_events_verified'] = 290 * 3 * 3 * 8
    result['runtime_math_flags'] = rows[0]['math_flags_before']
    return result


def summarize_profile(root=ROOT):
    rows = [json.loads((root / 'profile' / ('vision_rank%d.json' % rank)).read_text()) for rank in range(8)]
    main = [json.loads((root / 'ranks' / ('vision_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources']) != {'vision_profile_tp8.py', 'vision_linear71.py', 'vision_events71.py', 'candidate_padded_text_encoder.py', 'evidence_io71.py'}:
            raise ValueError('Missing exact profile executed-source fingerprints')
        for name, digest in row['sources'].items():
            path = root / name
            if name == 'candidate_padded_text_encoder.py':
                path = root.parent / 'round65' / 'source' / name
            elif name == 'evidence_io71.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Profile executed source differs from frozen local source')
    return aggregate_profile(rows, main)


if __name__ == '__main__':
    result = summarize_profile()
    (ROOT / 'profile_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))
    for mode, report in result['cases'].items():
        print(mode, {key: value for key, value in report.items() if key != 'profile_vectors'})
