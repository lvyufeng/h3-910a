#!/usr/bin/env python3
"""Independent full encode_ids timing/identity gate, not full HTTP or HF quality."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
EXPECTED_INPUTS = {'input_ids': ([2043], 'torch.int64', 16344),
                   'pixel_values': ([8064, 1536], 'torch.float16', 24772608),
                   'image_grid_thw': ([2, 3], 'torch.int64', 48)}
TEXT_SHAPES = {'hidden_input': [1, 2043, 5120], 'hidden_residual': [1, 2043, 5120],
               'q_head': [1, 2043, 8, 128], 'k_head': [1, 2043, 1, 128]}


def exact(proof, elements):
    return (proof['raw_bits_equal'] and proof['finite'] and proof['elements'] == elements
            and proof['max_abs_error'] == 0 and proof['relative_rms'] == 0)


def tensor_contract(value, shape, dtype, byte_count):
    if (value['shape'] != shape or value['dtype'] != dtype or value['bytes'] != byte_count
            or not value['finite'] or re.fullmatch('[0-9a-f]{64}', value['sha256']) is None):
        raise ValueError('Incomplete or wrong tensor contract: ' + str(shape))


def validate_identities(identities):
    expected_keys = {'embedding_ids', 'embedding_output', 'vision_inputs', 'vision_output',
                     'text_inputs', 'text_output', 'complete_output'}
    if set(identities) != expected_keys:
        raise ValueError('Missing actual per-call embedding/vision/text identities')
    tensor_contract(identities['embedding_ids'], [1, 2043], 'torch.int64', 16344)
    tensor_contract(identities['embedding_output'], [1, 2043, 5120], 'torch.bfloat16', 20920320)
    tensor_contract(identities['vision_inputs']['pixels'], [8064, 1536], 'torch.bfloat16', 24772608)
    tensor_contract(identities['vision_inputs']['grid'], [2, 3], 'torch.int64', 48)
    merged, deep = identities['vision_output']
    if len(deep) != 3:
        raise ValueError('Three complete DeepStack outputs are required')
    for value in (merged, *deep):
        tensor_contract(value, [2016, 5120], 'torch.bfloat16', 20643840)
    text = identities['text_inputs']
    if set(text) != {'inputs_embeds', 'positions', 'visual_pos_masks', 'deepstack_visual_embeds'}:
        raise ValueError('Missing actual multimodal text-forward inputs')
    tensor_contract(text['inputs_embeds'], [1, 2043, 5120], 'torch.bfloat16', 20920320)
    tensor_contract(text['positions'], [3, 1, 2043], 'torch.int64', 49032)
    tensor_contract(text['visual_pos_masks'], [1, 2043], 'torch.bool', 2043)
    if len(text['deepstack_visual_embeds']) != 3:
        raise ValueError('Missing text-forward DeepStack input')
    if text['deepstack_visual_embeds'] != deep:
        raise ValueError('Text did not consume the fresh vision DeepStack outputs')
    tensor_contract(identities['text_output'], [1, 2043, 5120], 'torch.bfloat16', 20920320)
    tensor_contract(identities['complete_output'], [2043, 5120], 'torch.bfloat16', 20920320)
    if identities['text_output']['sha256'] != identities['complete_output']['sha256']:
        raise ValueError('Complete CPU output differs from actual final layer-50 output')


def validate_first_checks(case):
    # Native/restored still use the same FP32 projection/cache reference path;
    # only fast's first call double-computes same-input candidate proofs.
    norms = case['norm_checks']
    casts = case['cast_checks']
    projections = case['projection_checks']
    if not case['fast_enabled']:
        if norms or casts or projections:
            raise ValueError('Unexpected double-compute checks in baseline/restored')
        return
    names = {'%02d.%s' % (layer, family) for layer in range(50) for family in TEXT_SHAPES}
    if len(norms) != 200 or {check['name'] for check in norms} != names:
        raise ValueError('Missing actual complete-encoder Norm layers/families')
    for check in norms:
        family = check['name'].split('.')[1]
        shape = TEXT_SHAPES[family]
        elements = 1
        for dimension in shape:
            elements *= dimension
        if not exact(check['output'], elements):
            raise ValueError('Same-input complete-encoder Norm output changed')
        if family == 'hidden_residual':
            if not exact(check['residual'], elements):
                raise ValueError('Original BF16 residual update changed')
        elif 'residual' in check:
            raise ValueError('Unexpected residual proof')
    expected_casts = {('shared', (1, 2043, 5120)): 50, ('fallback', (1, 2043, 5120)): 50,
                      ('fallback', (1, 2043, 1024)): 50, ('fallback', (1, 2043, 3200)): 50}
    if len(casts) != 200 or Counter((check['kind'], tuple(check['input_shape'])) for check in casts) != expected_casts:
        raise ValueError('Actual cast operand coverage changed')
    for check in casts:
        count = 1
        for dimension in check['input_shape']:
            count *= dimension
        if (not check['candidate_enabled'] or not check['input_readonly'] or check['input_dtype'] != 'torch.bfloat16'
                or not exact(check['proof'], count) or check['candidate_layout'] != check['native_layout']):
            raise ValueError('Actual activation cast identity/layout/input failed')
    expected_slots = {(family, layer): 1 for layer in range(50) for family in ('q', 'gate_up', 'o', 'down')}
    expected_slots['padded_kv'] = 100
    if len(projections) != 300 or Counter(tuple(check['slot']) if isinstance(check['slot'], list) else check['slot'] for check in projections) != expected_slots:
        raise ValueError('Missing complete-encoder projection proof layer/slice')
    counts = {'q': 2092032, 'gate_up': 13075200, 'o': 10460160, 'down': 10460160, 'padded_kv': 261504}
    for check in projections:
        slot = check['slot'][0] if isinstance(check['slot'], list) else check['slot']
        kind = 'row_partial_fp32' if slot in ('o', 'down') else 'local'
        if check['kind'] != kind or not exact(check, counts[slot]):
            raise ValueError('Same-input projection output or full FP32 partial changed')


def aggregate(rows):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique rank receipts')
    if any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Failed/incomplete full encoder experiment cannot be accepted')
    for row in rows:
        if row['parameters_before'] != row['parameters_after'] or row['actual_inputs_before'] != row['actual_inputs_after'] or row['actual_inputs_before'] != rows[0]['actual_inputs_before']:
            raise ValueError('Actual encoder input/parameter bytes changed or mismatched')
        if set(row['actual_inputs_before']) != set(EXPECTED_INPUTS):
            raise ValueError('Actual processor tensor set changed')
        for name, (shape, dtype, count) in EXPECTED_INPUTS.items():
            tensor_contract(row['actual_inputs_before'][name], shape, dtype, count)
        if row['capture_file_sha256'] != 'ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc':
            raise ValueError('Original processor/transport input fixture changed')
        if row['checkpoint_root'] != '/home/mseco/models/MiniMax-H3/FL2VA/text_encoder' or row['checkpoint_index_sha256'] != '06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a':
            raise ValueError('BF16 checkpoint source changed; converted FP16 source is not equivalent')
        if set(row['parameter_dtypes']) != {'torch.bfloat16'} or row['loaded_parameter_tensors'] != sum(row['parameter_dtypes'].values()):
            raise ValueError('Missing real embedding/vision/text parameter loads or dtype changed')
        expected_component_bytes = {'vision': 1190533600, 'embedding': 194478080, 'text_all_including_embedding': 6290375680}
        if (row['component_parameter_bytes'] != expected_component_bytes or row['bf16_complete_encoder_parameter_bytes'] != 7480909280
                or row['parameter_dtypes'] != {'torch.bfloat16': 752} or row['loaded_parameter_tensors'] != 752):
            raise ValueError('Real complete vision/text/vocabulary checkpoint footprint changed')
        flags = {'VLLM_OMNI_910A_NO_FUSED_NORM': '1', 'VLLM_OMNI_910A_NO_FUSED_ACT': '1',
                 'VLLM_OMNI_910A_TEXT_HF_MATH': '1', 'VLLM_OMNI_910A_TEXT_FP32_LINEAR': '1',
                 'VLLM_OMNI_910A_TEXT_PAD_KV': '1', 'VLLM_OMNI_910A_VISION_FP32': '1'}
        if any(row['environment'].get(name) != value for name, value in flags.items()) or 'VLLM_OMNI_910A_TE_DTYPE' in row['environment']:
            raise ValueError('Experimental full encoder precision/vision/padding flags changed')
    result = {'scope': 'Complete resident encode_ids, actual original FP16 transport pixels+IDs/grids, real BF16 checkpoint embedding and fresh replicated vision, positions/DeepStack/text50 and CPUoutput. Fixed experimental FP32vision/text math. Excludes outerHTTPprocessor/tokenization/TPinputdistribute/finalFP16broadcast/load/offload; not originalBF16/HF or official generation quality.',
              'all_actual_inputs_and_parameters_immutable': True, 'cases': {}, 'complete_output_identity_accepted': False,
              'complete_encode_ids_under2s_accepted': False, 'whole_service_text_under2s_proven': False,
              'official_goal_complete': False, 'production_deployed': False,
              'actual_norm_checks': 0, 'actual_residual_checks': 0, 'actual_cast_checks': 0, 'actual_projection_checks': 0}
    all_identities, output_proofs = [], []
    for label in ('baseline', 'fast', 'restored'):
        cases = [row['cases'][label] for row in rows]
        for case in cases:
            if case['fast_enabled'] != (label == 'fast') or len(case['warm_calls']) != 3:
                raise ValueError('Wrong complete-encoder mode or repeat count')
            validate_first_checks(case)
            if label == 'fast':
                result['actual_norm_checks'] += len(case['norm_checks'])
                result['actual_residual_checks'] += sum('residual' in check for check in case['norm_checks'])
                result['actual_cast_checks'] += len(case['cast_checks'])
                result['actual_projection_checks'] += len(case['projection_checks'])
            validate_identities(case['first_identities'])
            all_identities.append(case['first_identities'])
            output_proofs.append(case['first_vs_baseline'])
            if Counter(part['name'] for part in case['first_components']) != {'embedding': 1, 'vision': 1, 'text': 1}:
                raise ValueError('First diagnostic must actually run embedding, vision and text')
            for call in case['warm_calls']:
                validate_identities(call['identities'])
                all_identities.append(call['identities'])
                output_proofs.append(call['vs_baseline'])
                if call['output'] != call['identities']['complete_output']:
                    raise ValueError('Complete CPU-output receipt mismatch')
            if case['shadow_cache']['entries'] != 200 or case['shadow_cache']['cached_fp32_bytes'] != 11927552000:
                raise ValueError('Fixed FP32-shadow text stack changed')
            if case['existing_padding_cache'] != {'entries': 100, 'cached_fp32_bytes': 2097152000}:
                raise ValueError('Actual K/V padding geometry changed')
        maxima = [max(case['warm_calls'][i]['seconds'] for case in cases) for i in range(3)]
        result['cases'][label] = {'warm_complete_encode_ids_rank_max_s': maxima, 'median_complete_encode_ids_s': statistics.median(maxima),
                                 'all_warm_complete_encode_ids_under2s': all(value < 2 for value in maxima),
                                 'first_rank_max_s_excluded_component_probes_checks': max(case['first_s_excluded_checks_and_component_probes'] for case in cases),
                                 'allocated_hbm_bytes_per_rank': [case['hbm_allocated_bytes'] for case in cases],
                                 'reserved_hbm_bytes_per_rank': [case['hbm_reserved_bytes'] for case in cases]}
    reference = all_identities[0]
    result['intermediate_identity'] = {key: all(identity[key] == reference[key] for identity in all_identities)
                                        for key in ('embedding_ids', 'embedding_output', 'vision_inputs', 'vision_output', 'text_inputs', 'text_output', 'complete_output')}
    result['complete_outputs_compared'] = len(all_identities)
    result['same_input_mode_rank_repeat_complete_output_identity'] = all(exact(proof, 10460160) for proof in output_proofs)
    result['complete_output_identity_accepted'] = (all(result['intermediate_identity'].values())
                                                    and result['same_input_mode_rank_repeat_complete_output_identity'])
    result['complete_encode_ids_under2s_accepted'] = result['complete_output_identity_accepted'] and result['cases']['fast']['all_warm_complete_encode_ids_under2s']
    result['speedup_vs_same_run_experimental_baseline'] = result['cases']['baseline']['median_complete_encode_ids_s'] / result['cases']['fast']['median_complete_encode_ids_s']
    result['restored_timing_drift'] = result['cases']['restored']['median_complete_encode_ids_s'] / result['cases']['baseline']['median_complete_encode_ids_s'] - 1
    result['full_embedding_vision_text_parameter_bytes_per_rank'] = [row['bf16_complete_encoder_parameter_bytes'] for row in rows]
    result['component_parameter_bytes_rank0'] = rows[0]['component_parameter_bytes']
    result['first_component_probes_rank0_excluded_not_warm_attribution'] = {label: rows[0]['cases'][label]['first_components'] for label in ('baseline', 'fast', 'restored')}
    return result


def summarize(root=ROOT):
    rows = [json.loads((root / 'ranks' / ('encoder_rank%d.json' % rank)).read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources']) != {'complete_encode_tp8.py', 'candidate_padded_text_encoder.py', 'evidence_io70.py'}:
            raise ValueError('Missing executed-source fingerprints')
        for name, digest in row['sources'].items():
            path = root / name
            if name == 'candidate_padded_text_encoder.py':
                path = root.parent / 'round65' / 'source' / name
            elif name == 'evidence_io70.py':
                path = root.parent / 'round63' / 'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Executed source fingerprint mismatch')
    return aggregate(rows)


if __name__ == '__main__':
    result = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
