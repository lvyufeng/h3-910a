#!/usr/bin/env python3
"""Exact partial-tail, realpacked fixture and complete encode_ids acceptance gates."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'round70'))
from summarize import exact, validate_identities, tensor_contract

ROOT = Path(__file__).resolve().parent
MODES = ('native', 'tail', 'restored')
LAYERS = (0, 8, 16, 26)


def role_proof(proof):
    return (proof['shape'] == [8064,16,72] and proof['dtype'] == 'torch.bfloat16' and proof['elements'] == 9289728
            and proof['raw_bits_equal'] and proof['finite'] and proof['different_words'] == 0 and proof['max_abs_error'] == 0)


def check_layout(value, shape, stride, dtype):
    if value['shape'] != shape or value['stride'] != stride or value['dtype'] != dtype or value['physical_format'] not in (0,2):
        raise ValueError('Incorrect measured vision RoPE layout')


def validate_fixtures(root):
    captures = [json.loads((root/'capture'/f'capture_rank{rank}.json').read_text()) for rank in range(8)]
    if any(not row.get('completed') or row.get('failure') or row['actual_vision_rope_calls'] != 27 for row in captures):
        raise ValueError('Actual full encoder capture failed/incomplete')
    if any(not exact(row['capture_full_output_vs_round70_fresh_reference'],10460160) for row in captures):
        raise ValueError('Captured full encoder output differs from stable reference')
    manifests = [json.loads((root/'capture'/f'rope_layer{layer:02d}.json').read_text()) for layer in LAYERS]
    if sum(meta['file_bytes'] for meta in manifests) >= 512*1024*1024:
        raise ValueError('Capture total exceeds declared budget')
    for meta, layer in zip(manifests, LAYERS):
        if meta['layer'] != layer or meta['file_bytes'] >= 128*1024*1024:
            raise ValueError('Wrong captured layer or per-file budget')
        for name, offset in (('query_layout',0),('key_layout',1152)):
            value=meta[name]
            check_layout(value,[8064,16,72],[3456,72,1],'torch.bfloat16')
            if value['storage_offset'] != offset or value['storage_bytes'] != 55738368 or not value['pointer_aligned32']:
                raise ValueError('Actual packed Q/K storage bounds or offset changed')
        for name in ('cos_layout','sin_layout'):
            check_layout(meta[name],[8064,72],[72,1],'torch.float32')
        for name in ('query_output_layout','key_output_layout'):
            check_layout(meta[name],[8064,16,72],[1152,72,1],'torch.bfloat16')
        if not all(meta[name] for name in ('query_values_match_full_storage','key_values_match_full_storage','cos_halves_equal','sin_halves_equal')):
            raise ValueError('Fullpacked source/real coefficient contract failed')
        expected={'packed_qkv':([8064,3,16,72],'torch.bfloat16',55738368),'cos':([8064,72],'torch.float32',2322432),
                  'sin':([8064,72],'torch.float32',2322432),'query_out':([8064,16,72],'torch.bfloat16',18579456),'key_out':([8064,16,72],'torch.bfloat16',18579456)}
        for name,(shape,dtype,byte_count) in expected.items():tensor_contract(meta['tensor_identities'][name],shape,dtype,byte_count)
        if max(meta['max_abs_values'].values()) >= 2**120 or meta['max_abs_values']['cos'] > 1 or meta['max_abs_values']['sin'] > 1:
            raise ValueError('Actual fixture not within certified finite non-overflow range')
    validation=json.loads((root/'validation01.json').read_text())
    if not validation.get('completed') or validation.get('failure') or {row['layer'] for row in validation['layers']} != set(LAYERS):
        raise ValueError('Real fullpacked standalone validation failed')
    for row in validation['layers']:
        for name in ('native_query_vs_capture','native_key_vs_capture','candidate_query','candidate_key'):
            if not role_proof(row[name]):raise ValueError('Real complete RoPE tensor proof failed')
        if row['output_strides'] != row['native_output_strides'] or row['output_formats'] != row['native_output_formats']:
            raise ValueError('Standalone native layout changed')
        if row['wrong_dense_pitch_query']['raw_bits_equal'] or not row['packed_and_coefficients_readonly'] or not row['unaligned_coefficient_rejected']:
            raise ValueError('Wrong-pitch/readonly/alignment counterexample gate missing')
        if len(row['warm']) != 3 or any(not all(role_proof(p) for p in call['candidate_proofs']) for call in row['warm']):
            raise ValueError('Standalone clean repeat proof failed')
    micro=json.loads((root/'micro_tail01.json').read_text())
    boundary=json.loads((root/'micro_tail02.json').read_text())
    if not micro.get('completed') or micro.get('failure') or set(micro['cases'])!={'1','5','8'} or not boundary.get('completed') or boundary.get('failure'):
        raise ValueError('Actual tiny or odd-row tail micro gates failed')
    odd=2*boundary['runtime_core_budget']+1
    if set(boundary['cases'])!={str(odd)}:
        raise ValueError('Odd-row micro does not exercise the actual core budget')
    for n_text,row in list(micro['cases'].items())+list(boundary['cases'].items()):
        n=int(n_text)
        for name in ('query','key'):
            p=row[name]
            if p['shape']!=[n,16,72] or p['elements']!=n*1152 or not p['raw_bits_equal'] or not p['finite'] or p['different_words']!=0 or p['max_abs_error']!=0:
                raise ValueError('Tiny/odd-row native-bit micro proof failed')
        if not row['packed_readonly'] or row['output_stride']!=row['native_output_stride'] or row['output_format']!=row['native_output_format']:
            raise ValueError('Tiny/odd-row layout/readonly gate failed')
    b=boundary['cases'][str(odd)]
    if not b['multiple_steps_and_inactive_last_rows_exercised'] or not b['four_fp32_inputs_readonly'] or not b['direct_tail_query']['raw_bits_equal'] or not b['direct_tail_query']['finite']:
        raise ValueError('Multistep/inactive-row/directFP32readonly gate missing')
    failure=json.loads((root/'gather32_failure.json').read_text())
    support=json.loads((root/'gather_support.json').read_text())
    if failure.get('completed') or failure['failure']['type']!='SystemExit' or 't.vgather' not in failure['failure']['message'] or support['gather_path_enabled'] or any(support['queries'].values()):
        raise ValueError('Unsupported gather counterexample was hidden or enabled')
    return {'captured_layers':list(LAYERS),'fullpacked_capture_file_bytes':sum(meta['file_bytes'] for meta in manifests),
            'tiny_and_odd_micro_tokens':[1,5,8,odd],'runtime_core_budget':boundary['runtime_core_budget'],
            'multistep_inactive_rows_and_direct_fp32_readonly_verified':True,
            'unsupported_gather_counterexample_retained':True,
            'all_real_standalone_outputs_exact':True,'wrong_pitch_and_alignment_counterexamples_retained':True,
            'standalone_wrapper_medians':{str(row['layer']):{'native_s':row['native_median_s'],'tail_s':row['tail_median_s']} for row in validation['layers']}}


def aggregate(rows, previous):
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)) or any(not row.get('completed') or row.get('failure') for row in rows):
        raise ValueError('Need eight successful genuine NPU rank receipts')
    rows=sorted(rows,key=lambda row:row['rank']);old={row['rank']:row for row in previous}
    reference=old[0]['cases']['both']['first_identities']
    result={'scope':'Partial vision RoPE tail only: original Torch packed BF16->FP32 cast and half36 cat/negate remain. TBE separateFP32mul/mul/add+oneBF16RNE. FixedR71bothvisionLinear/R70fasttext, true fresh fullencoder and CPUoutput; excludes upstreamHTTPprocessor/broadcast/load/offload. Stableexperimentalreference, not HForofficialquality.',
            'cases':{},'actual_rope_pair_checks':0,'actual_query_key_output_checks':0,'complete_output_identity_accepted':False,
            'complete_encode_ids_under2s_accepted':False,'whole_service_text_under2s_proven':False,'official_goal_complete':False,'production_deployed':False,
            'full_packed_gather_fusion_supported':False}
    all_ids,proofs=[],[]
    for row in rows:
        prev=old[row['rank']]
        if row['parameters_before'] != row['parameters_after'] or row['parameters_before'] != prev['parameters_before'] or row['actual_inputs_before'] != row['actual_inputs_after'] or row['actual_inputs_before'] != prev['actual_inputs_before']:
            raise ValueError('Actual complete encoder input or parameter bytes changed')
        for field in ('checkpoint_root','checkpoint_index_sha256','capture_file_sha256','component_parameter_bytes','parameter_dtypes','loaded_parameter_tensors','environment'):
            if row[field] != prev[field]:raise ValueError('Fixed original BF16 checkpoint/input/precision configuration changed')
        if row['math_flags_before'] != row['math_flags_after'] or row['math_flags_before'] != prev['math_flags_before']:
            raise ValueError('Actual NPU math flags changed')
        if not row['vision_rope_restored'] or not row['vision_wrappers_restored'] or row['vision_shadow_bytes_after_clear'] != 0:
            raise ValueError('Isolated vision state did not restore')
        if not exact(row['first_native_vs_round70_fresh_reference'],10460160):raise ValueError('Full native RoPE baseline differs from stable reference')
    for mode in MODES:
        cases=[row['cases'][mode] for row in rows]
        for case in cases:
            if case['rope_mode'] != mode or case['fixed_vision_linear_mode'] != 'both' or case['fixed_text_mode'] != 'shared_qkv_add_square_norm_tail_both_casts' or case['first_actual_rope_calls'] != 27:
                raise ValueError('RoPE factor or fixed existing Linear/text stack changed')
            checks=case['rope_checks']
            if mode=='tail':
                if len(checks)!=27 or Counter(check['layer'] for check in checks)!=Counter({layer:1 for layer in range(27)}):raise ValueError('Missing actual full-vision RoPE layers')
                for check in checks:
                    if not exact(check['query'],9289728) or not exact(check['key'],9289728) or not check['packed_and_coefficients_readonly'] or check['output_layouts']!=check['native_output_layouts']:
                        raise ValueError('Actual same-input full vision RoPE bit/layout/readonly proof failed')
                    for name in ('query_input_layout','key_input_layout'):check_layout(check[name],[8064,16,72],[3456,72,1],'torch.bfloat16')
                    for name in ('cos_layout','sin_layout'):check_layout(check[name],[8064,72],[72,1],'torch.float32')
                    for layout in check['output_layouts']:check_layout(layout,[8064,16,72],[1152,72,1],'torch.bfloat16')
            elif checks:raise ValueError('Native/restored unexpectedly contains candidate double-compute')
            result['actual_rope_pair_checks']+=len(checks);result['actual_query_key_output_checks']+=2*len(checks)
            validate_identities(case['first_identities']);all_ids.append(case['first_identities']);proofs.append(case['first_vs_baseline'])
            if len(case['warm_calls'])!=3:raise ValueError('Need three clean complete encode_ids calls')
            for call in case['warm_calls']:
                validate_identities(call['identities'])
                if call['actual_rope_calls']!=27 or call['output']!=call['identities']['complete_output'] or not math.isfinite(call['seconds']) or call['seconds']<=0:raise ValueError('Wrong actual RoPE call count/CPUoutput/timer')
                all_ids.append(call['identities']);proofs.append(call['vs_baseline'])
            if case['vision_shadow_cache']!={'entries':232,'cached_fp32_bytes':2362750400,'hits':696,'misses':232,'invalidations':0,'uncacheable_bypasses':0,'max_bytes':2362750400}:raise ValueError('Fixed vision Linear shadow cache policy changed')
            if case['text_shadow_cache']['entries']!=200 or case['text_shadow_cache']['cached_fp32_bytes']!=11927552000 or case['existing_padding_cache']!={'entries':100,'cached_fp32_bytes':2097152000}:raise ValueError('Fixed text shadow/KV cache changed')
        maxima=[max(case['warm_calls'][i]['seconds'] for case in cases) for i in range(3)]
        result['cases'][mode]={'warm_complete_encode_ids_rank_max_s':maxima,'median_complete_encode_ids_s':statistics.median(maxima),
                              'first_rank_max_s_excluded_checks_compile':max(case['first_s_excluded_checks_compile_and_component_probes'] for case in cases),
                              'allocated_hbm_bytes_per_rank':[case['hbm_allocated_bytes'] for case in cases],
                              'reserved_hbm_bytes_per_rank':[case['hbm_reserved_bytes'] for case in cases]}
    result['complete_outputs_compared']=len(all_ids)
    result['intermediate_identity_against_round71']={name:all(ids[name]==reference[name] for ids in all_ids) for name in reference}
    result['complete_output_identity_accepted']=all(result['intermediate_identity_against_round71'].values()) and all(exact(proof,10460160) for proof in proofs)
    result['complete_encode_ids_under2s_accepted']=result['complete_output_identity_accepted'] and all(value<2 for value in result['cases']['tail']['warm_complete_encode_ids_rank_max_s'])
    native=result['cases']['native']['median_complete_encode_ids_s']
    result['speedup_vs_same_run_native']=native/result['cases']['tail']['median_complete_encode_ids_s']
    result['restored_timing_drift']=result['cases']['restored']['median_complete_encode_ids_s']/native-1
    result['runtime_math_flags']=rows[0]['math_flags_before']
    return result


def summarize(root=ROOT):
    rows=[json.loads((root/'ranks'/f'rope_rank{rank}.json').read_text()) for rank in range(8)]
    previous=[json.loads((root.parent/'round71'/'ranks'/f'vision_rank{rank}.json').read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources'])!={'vision_tail_tp8.py','candidate_padded_text_encoder.py','evidence_io72.py'}:
            raise ValueError('Missing exact complete encoder executed-source fingerprints')
        for name,digest in row['sources'].items():
            path=root/name
            if name=='candidate_padded_text_encoder.py':path=root.parent/'round65'/'source'/name
            elif name=='evidence_io72.py':path=root.parent/'round63'/'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Executed complete RoPE A/B source fingerprint differs')
    result=aggregate(rows,previous)
    result['real_fixture_gate']=validate_fixtures(root)
    return result


if __name__=='__main__':
    result=summarize()
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
