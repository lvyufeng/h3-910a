#!/usr/bin/env python3
"""Reconcile clean and synchronized component probes without mixing their costs."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from summarize import EXPECTED_INPUTS, aggregate, exact, tensor_contract, validate_identities

ROOT = Path(__file__).resolve().parent
LABELS = ('baseline', 'fast', 'restored')
PHASES = ('clean_before', 'component_probes', 'clean_after')


def aggregate_components(rows, main):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique complete-encoder probe ranks')
    if any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Failed/incomplete component probes cannot be accepted')
    rows = sorted(rows, key=lambda row: row['rank'])
    main_rows = {row['rank']: row for row in main}
    all_ids, all_proofs = [], []
    for row in rows:
        old = main_rows[row['rank']]
        if (row['parameters_before'] != row['parameters_after'] or row['parameters_before'] != old['parameters_before']
                or row['actual_inputs_before'] != row['actual_inputs_after'] or row['actual_inputs_before'] != old['actual_inputs_before']):
            raise ValueError('Probe changed complete-model parameters or actual input bytes')
        if row['checkpoint_root'] != old['checkpoint_root'] or row['checkpoint_index_sha256'] != old['checkpoint_index_sha256']:
            raise ValueError('Probe checkpoint provenance differs from main A/B')
        if row['parameter_dtypes'] != old['parameter_dtypes'] or row['loaded_parameter_tensors'] != old['loaded_parameter_tensors']:
            raise ValueError('Probe did not load the same real full encoder')
        if (row['component_parameter_bytes'] != old['component_parameter_bytes']
                or row['bf16_complete_encoder_parameter_bytes'] != old['bf16_complete_encoder_parameter_bytes']):
            raise ValueError('Probe omitted real embedding/vision/text parameter storage')
        flag_names = ('VLLM_OMNI_910A_NO_FUSED_NORM', 'VLLM_OMNI_910A_NO_FUSED_ACT', 'VLLM_OMNI_910A_TEXT_HF_MATH',
                      'VLLM_OMNI_910A_TEXT_FP32_LINEAR', 'VLLM_OMNI_910A_TEXT_PAD_KV', 'VLLM_OMNI_910A_VISION_FP32')
        if any(row['environment'].get(name) != old['environment'].get(name) for name in flag_names) or 'VLLM_OMNI_910A_TE_DTYPE' in row['environment']:
            raise ValueError('Probe changed actual vision/text precision flags')
        if row['capture_file_sha256'] != old['capture_file_sha256']:
            raise ValueError('Probe actual original transport fixture differs from main run')
    result = {'scope': 'Independent fresh-run complete resident encode_ids clean/probe/clean controls. Component probes introduce per-component drains; they do not replace clean performance or official full-service measurement.',
              'all_actual_inputs_and_parameter_bytes_match_main_run': True, 'cases': {},
              'complete_output_identity_accepted': False, 'whole_service_text_under2s_proven': False,
              'official_goal_complete': False, 'production_deployed': False}
    for label in LABELS:
        cases = [row['cases'][label] for row in rows]
        for case in cases:
            if case['fast_enabled'] != (label == 'fast') or not exact(case['first_vs_baseline'], 10460160):
                raise ValueError('Wrong first probe output or candidate mode')
            validate_identities(case['first_identities'])
            all_ids.append(case['first_identities'])
            all_proofs.append(case['first_vs_baseline'])
            for phase in PHASES:
                if len(case[phase]) != 3:
                    raise ValueError('Need three before/probe/after calls on every rank')
                for call in case[phase]:
                    validate_identities(call['identities'])
                    if call['output'] != call['identities']['complete_output']:
                        raise ValueError('Complete probe output receipt mismatch')
                    all_ids.append(call['identities'])
                    all_proofs.append(call['vs_baseline'])
                    if not math.isfinite(call['seconds']) or call['seconds'] <= 0:
                        raise ValueError('Invalid complete wall timer')
                    if phase == 'component_probes':
                        if Counter(part['name'] for part in call['parts']) != {'embedding': 1, 'vision': 1, 'text': 1}:
                            raise ValueError('Missing or duplicate real component probe')
                        for part in call['parts']:
                            if not all(math.isfinite(value) and value >= 0 for value in (part['synchronized_wall_s'], part['current_stream_elapsed_ms'])):
                                raise ValueError('Invalid synchronized component timing')
                    elif 'parts' in call:
                        raise ValueError('Clean call unexpectedly recorded component synchronization')
            if case['shadow_cache']['entries'] != 200 or case['shadow_cache']['cached_fp32_bytes'] != 11927552000:
                raise ValueError('Fixed shadow residency changed')
            if case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000}:
                raise ValueError('Fixed padded K/V geometry changed')
        maxima = {phase: [max(case[phase][i]['seconds'] for case in cases) for i in range(3)] for phase in PHASES}
        report = {phase + '_per_call_rank_max_s': values for phase, values in maxima.items()}
        report.update({phase + '_median_rank_max_s': statistics.median(values) for phase, values in maxima.items()})
        report['probe_wall_perturbation_vs_clean_before'] = statistics.median(maxima['component_probes']) / statistics.median(maxima['clean_before']) - 1
        report['clean_after_drift_vs_before'] = statistics.median(maxima['clean_after']) / statistics.median(maxima['clean_before']) - 1
        report['all_clean_before_calls_under2s'] = all(value < 2 for value in maxima['clean_before'])
        # Each representative is the actual slowest COMPLETE wall rank from the
        # same call. Never add unrelated cross-rank component maxima.
        vectors = []
        for i in range(3):
            slow = max(range(8), key=lambda rank: cases[rank]['component_probes'][i]['seconds'])
            per_rank = []
            for rank, case in enumerate(cases):
                call = case['component_probes'][i]
                parts = {part['name']: part for part in call['parts']}
                subtotal = sum(part['synchronized_wall_s'] for part in parts.values())
                per_rank.append({'rank': rows[rank]['rank'], 'complete_encode_ids_wall_s': call['seconds'],
                                 'synchronized_components': parts, 'component_wall_sum_s': subtotal,
                                 'wall_minus_component_sum_s': call['seconds'] - subtotal})
            vectors.append({'call': i, 'slowest_complete_wall_rank': rows[slow]['rank'],
                            'representative': per_rank[slow], 'all_rank_vectors': per_rank})
        report['probe_vectors'] = vectors
        result['cases'][label] = report
    ref = main_rows[0]['cases']['baseline']['first_identities']
    result['complete_outputs_compared'] = len(all_ids)
    result['intermediate_identity_against_main_run'] = {name: all(ids[name] == ref[name] for ids in all_ids)
                                                      for name in ref}
    result['complete_output_identity_accepted'] = (all(result['intermediate_identity_against_main_run'].values())
                                                  and all(exact(proof, 10460160) for proof in all_proofs))
    result['complete_encode_ids_under2s_accepted'] = (result['complete_output_identity_accepted']
                                                    and result['cases']['fast']['all_clean_before_calls_under2s'])
    result['clean_encode_ids_speedup'] = result['cases']['baseline']['clean_before_median_rank_max_s'] / result['cases']['fast']['clean_before_median_rank_max_s']
    result['full_encode_ids_output_sha256'] = ref['complete_output']['sha256']
    result['saved_fresh_reference_rank0'] = rows[0]['saved_fresh_reference']
    return result


def summarize_components(root=ROOT):
    main = [json.loads((root / 'ranks' / ('encoder_rank%d.json' % rank)).read_text()) for rank in range(8)]
    aggregate(main)  # Main A/B gate must remain satisfied independently.
    rows = [json.loads((root / 'components' / ('encoder_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources']) != {'warm_components70.py', 'candidate_padded_text_encoder.py', 'evidence_io70.py'}:
            raise ValueError('Missing component-run executed-source fingerprints')
        for name, digest in row['sources'].items():
            path = root / name
            if name == 'candidate_padded_text_encoder.py':
                path = root.parent / 'round65' / 'source' / name
            elif name == 'evidence_io70.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Component-run executed source differs')
    return aggregate_components(rows, main)


if __name__ == '__main__':
    result = summarize_components()
    (ROOT / 'component_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: value for name, value in result.items() if name != 'cases'}, indent=2))
    for name, case in result['cases'].items():
        print(name, {key: value for key, value in case.items() if key != 'probe_vectors'})
