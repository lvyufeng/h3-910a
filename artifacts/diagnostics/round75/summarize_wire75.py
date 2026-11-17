#!/usr/bin/env python3
"""Genuine fixed-size HCCL gather proof; not arithmetic ISA or whole-encoder speed."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
CASES={(128,544),(2016,432),(8064,432),(8064,144),(8064,538),(8064,544)}


def summarize_mode(mode,root=ROOT):
    rows=[json.loads((root/('wire_'+mode)/f'wire_rank{i}.json').read_text())for i in range(8)]
    main_sha=hashlib.sha256((root/'wire_probe75.py').read_bytes()).hexdigest()
    count=0;roundtrips=0;guard_rejections=0
    for row in rows:
        if not row.get('completed')or row.get('failure')or row['mode']!=mode or row['world_size']!=8 or row['device_index']!=row['rank']:raise ValueError('Need actual eight-rank successful device gathers')
        if row['source_sha256']!=main_sha or row['device_group_backend']!='hccl'or row['actual_group_class']!='vllm_omni.diffusion.distributed.group_coordinator.GroupCoordinator'or row['integer_transport_only']!=(mode=='int32'):raise ValueError('Wrong actual communicator/source/backend/dtype mode')
        if row['production_communicator_proven']or row['whole_encoder_acceleration_proven']or row['official_goal_complete']:raise ValueError('Transport probe promoted to production/full quality')
        if {(case['rows'],case['bf16_columns'])for case in row['cases']}!=CASES or len(row['cases'])!=6:raise ValueError('Missing genuine payload shapes')
        for case in row['cases']:
            n,c=case['rows'],case['bf16_columns']
            if case['source_physical_format']not in(0,2)or case['source_ptr_mod32']or case['source_storage_offset']or not case['source_readonly']or case['distinct_bf16_words_in_first65568_source_words']!=65536:raise ValueError('Source layout/readonly/word-domain proof changed')
            p=case['raw_device_view_roundtrip']
            if not p['raw_pairs_equal']or p['different_int32_pairs']or p['different_bf16_words']or p['shape_int32']!=[n,c//2]:raise ValueError('Raw device view changed BF16 words')
            roundtrips+=1
            if set(case['directions'])!={'0','-1'}:raise ValueError('Both rank-major and row-major gathers required')
            for direction,value in case['directions'].items():
                proofs=[value['proof'],*value['repeat_proofs']]
                if len(proofs)!=4 or len(value['warm_same_call_s'])!=3:raise ValueError('Missing coldcontrol/threewarmwordproofs')
                shape=[8*n,c//2]if direction=='0'else[n,8*c//2]
                for p in proofs:
                    if p['shape_int32']!=shape or p['int32_elements']!=8*n*c//2 or p['bf16_words']!=8*n*c or not p['raw_pairs_equal']or p['different_int32_pairs']or p['different_bf16_words']or p['observed_sha256']!=p['expected_sha256']:raise ValueError('Gather rank tags/order/raw words changed')
                    count+=1
                if any(not math.isfinite(t)or t<=0 for t in value['warm_same_call_s']):raise ValueError('Invalid raw-gather timer')
            if n==128:
                if set(case['guard_rejections'])!={'odd_columns','odd_offset','noncontiguous'}or not all(v['rejected']for v in case['guard_rejections'].values()):raise ValueError('Bad raw view accepted')
                guard_rejections+=len(case['guard_rejections'])
    results=[]
    for n,c in sorted(CASES):
        for direction in('0','-1'):
            values=[next(case for case in row['cases']if(case['rows'],case['bf16_columns'])==(n,c))['directions'][direction]for row in rows]
            maxima=[max(value['warm_same_call_s'][i]for value in values)for i in range(3)]
            results.append({'rows':n,'bf16_columns':c,'direction':direction,'same_call_rank_max_s':maxima,'median_s':statistics.median(maxima),
                            'bytes_per_rank_input':n*c*2,'full_output_bytes_per_rank':8*n*c*2})
    return {'mode':mode,'all_tested_raw_words_equal':True,'gather_proofs_verified':count,'raw_view_roundtrips':roundtrips,'guard_rejections_verified':guard_rejections,
            'guard_negative_fixture_rows':128,'host_CPU_reference_cohosting':True,'dtype_latency_comparison_accepted':False,
            'actual_encoder_group':'vllm_omni.diffusion.distributed.group_coordinator.GroupCoordinator',
            'group_source_sha256':rows[0]['actual_group_source_sha256'],'results':results,
            'scope':'Raw all65536BF16encodings,finite+Inf+NaNpayloads,throughfixedgatherson8genuineNPUs. Not BF16arithmetic/productionvLLMcommunicator/wholeencoderperformance proof.'}


def summarize(root=ROOT):
    modes={mode:summarize_mode(mode,root)for mode in('int32','bf16')}
    if modes['int32']['group_source_sha256']!=modes['bf16']['group_source_sha256']:raise ValueError('Communicator source changed between dtype controls')
    cross_mode_proofs=0
    for rank in range(8):
        a=json.loads((root/'wire_int32'/f'wire_rank{rank}.json').read_text());b=json.loads((root/'wire_bf16'/f'wire_rank{rank}.json').read_text())
        byshape={(case['rows'],case['bf16_columns']):case for case in b['cases']}
        for case in a['cases']:
            other=byshape[(case['rows'],case['bf16_columns'])]
            if case['raw_device_view_roundtrip']['expected_sha256']!=other['raw_device_view_roundtrip']['expected_sha256']:raise ValueError('Cross-mode source payload differs')
            for direction in('0','-1'):
                if case['directions'][direction]['proof']['expected_sha256']!=other['directions'][direction]['proof']['expected_sha256']:raise ValueError('Cross-mode gather payload/order differs')
                cross_mode_proofs+=1
    return {'modes':modes,'same_payload_cross_mode_gather_identity_checks':cross_mode_proofs,
            'current_encoder_actual_FP32_all_reduce_not_covered_by_this_gather_probe':True,
            'direct_bf16_gather_bits_proven_for_current_experimental_encoder_group':True,
            'int32_raw_pair_gather_bits_proven':True,'integer_codec_required_for_this_direct_gather':False,
            'all_gather_was_not_observed_to_cast_to_fp16':True,'all_reduce_bf16_fp16_path_is_a_different_operation':True,
            'production_communicator_proven':False,'whole_encoder_acceleration_proven':False,'official_goal_complete':False}


if __name__=='__main__':
    r=summarize();(ROOT/'wire_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
