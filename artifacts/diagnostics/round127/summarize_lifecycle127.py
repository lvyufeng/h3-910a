#!/usr/bin/env python3
"""Summarize completed R127 same-worker lifecycle from actual audit rows only.

No model execution, tensor reads, or automatic performance/precision acceptance.
The observer's current-stream event intervals are inclusive and not additive.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import statistics

LABELS={1:'r01_cold',2:'r02_warm',3:'r03_native_reload'}


def median_rank_max(rows,key):
    by_step=defaultdict(list)
    for row in rows:by_step[row['step_index']].append(row[key])
    assert sorted(by_step)==list(range(49)) and all(len(v)==8 for v in by_step.values())
    maxima=[max(by_step[index]) for index in range(49)]
    return {'first_step_s':maxima[0],'warm_steps_1_to_48_median_s':statistics.median(maxima[1:]),
            'warm_steps_min_s':min(maxima[1:]),'warm_steps_max_s':max(maxima[1:]),
            'every_step_rank_max_s':maxima}


def summarize(root:Path):
    result=json.loads((root/'lifecycle_result127.json').read_text())
    assert result['completed'] and len(result['requests'])==3
    audits=sorted(root.glob('worker_runtime_pid*127.jsonl'));assert len(audits)>=8
    rows=[json.loads(line) for path in audits for line in path.read_text().splitlines()]
    constructed=[r for r in rows if r['event']=='matched_transformer_constructed']
    assert len(constructed)==8 and all(r['main_layers']==50 for r in constructed)
    for row in constructed:
        counts=row['parameter_dtype_counts']
        assert set(counts)<= {'torch.float16','torch.float32'}
        assert 2*counts.get('torch.float16',0)+4*counts.get('torch.float32',0)==33168334080,'True TP2 size differs'
    expected_pids=None;expected_ids=None;cases=[]
    for index,label in LABELS.items():
        records=[row for row in rows if row['request_index']==index]
        begin=[r for r in records if r['event']=='native_request_begin']
        end=[r for r in records if r['event']=='native_request_end']
        steps=[r for r in records if r['event']=='dit_step_end']
        pids={r['pid'] for r in begin};assert len(begin)==len(end)==len(pids)==8
        assert len(steps)==8*49
        model_ids={r['pid']:(r['pipeline_id'],r['transformer_id']) for r in begin}
        if expected_pids is None:expected_pids=pids;expected_ids=model_ids
        else:assert pids==expected_pids and model_ids==expected_ids
        assert {r['pid'] for r in end}==pids
        for pid in pids:
            current=[r for r in steps if r['pid']==pid]
            assert sorted(r['step_index'] for r in current)==list(range(49))
            assert sorted(r['total_step_index'] for r in current)==list(range((index-1)*49,index*49))
            assert all(r['full50_main_layers_executed'] and r['actual_Half_matrix_modules_this_step']==155 for r in current)
            ending=next(r for r in end if r['pid']==pid)
            assert ending['checks_this_request']==7595 and ending['total_steps']==index*49
        assert len([r for r in records if r['event']=='live_NPU_encoder_hidden'])==8
        assert len([r for r in records if r['event']=='denoise_inputs_ready'])==8
        assert all(r['sigma_points']==50 and r['own_NPU_conditioning'] for r in records if r['event']=='denoise_inputs_ready')
        assert not any(r['event'] in ('half_matrix_range_warning','native_request_failure') for r in records)
        state_rows=[r['state'] for r in begin+end]
        assert all(not r['HF32_matmul'] and not r['HF32_conv'] for r in state_rows)
        transitions=[r for r in records if r['event']=='stager_transition']
        loads=[r for r in transitions if r['method']=='_load_once']
        offloads=[r for r in transitions if r['method']=='offload' and r['before']['loaded']]
        for row in transitions:
            assert row['before']['master_metadata_sha256']==row['after']['master_metadata_sha256']
            if row['method']=='_load_once':assert not row['before']['loaded'] and row['after']['loaded']
            if row['method']=='offload' and row['before']['loaded']:assert not row['after']['loaded']
        stage=defaultdict(list)
        for row in records:
            if row['event']=='stage_end':stage[row['stage']].append(row['host_elapsed_s'])
        stage_max={name:max(values) for name,values in stage.items()}
        native_decodes=[r for r in records if r['event']=='native_NPU_VAE_decode_end'];assert len(native_decodes)==8
        assert all(not r['operand_cache_enabled'] for r in native_decodes)
        prefetch=[v for row in end for v in row['prefetch']]
        assert prefetch and all(v['calls']>0 for v in prefetch)
        copies_shared=Counter('shared' if v['streams']['copy_is_current'] else 'independent' for v in prefetch)
        per_rank_loads={str(pid):Counter(r['label'] for r in loads if r['pid']==pid) for pid in pids}
        pins=[s for row in begin+end for s in row['state']['stagers']]
        case={'index':index,'label':label,'generation_wall_s':result['requests'][index-1]['generation_wall_s'],
              'native_worker_request_rank_max_s':max(r['host_elapsed_s'] for r in end),
              'whole_forward_hook':median_rank_max(steps,'elapsed_s_includes_offload_and_D2H_audit'),
              'same_scope_as_R124_prediction_D2H_safety_hook_not_pure_NPU_compute':True,
              'original_strict_Half_checks_passed':60760,'actual_steps_each_rank':49,'worker_pids':sorted(pids),
              'host_stage_rank_max_s':stage_max,
              'cold_or_warm_native_VAE_combined_rank_max_s':max(r['elapsed_s'] for r in native_decodes),
              'actual_stager_load_once_count':len(loads),'actual_stager_offload_count':len(offloads),
              'actual_loads_by_rank_and_component':per_rank_loads,
              'layerwise_prefetch_copy_stream_roles':dict(copies_shared),
              'actual_layerwise_prefetch_calls_total':sum(v['calls'] for v in prefetch),
              'pinned_master_metadata_stayed_stable_across_each_transition':True,
              'pinned_master_full_byte_comparison_per_transition_performed':False,
              'all_stager_masters_pinned':all(s['all_masters_pinned'] for s in pins),
              'thread_affinity_counts':sorted({len(r['streams']['affinity']) for r in begin}),
              'stage_event_scope':'current consumer stream inclusive elapsed, includes waits/host gaps; not summed or called pure copy',
              'NUMA_scope':'whole VMA node statistics for sampled pinned pointers; exact per-tensor pages remain unknown',
              'peak_allocated_rank_max_bytes':max((r['state']['memory']['peak_allocated_bytes'] for r in end),default=None)
                  if all('peak_allocated_bytes' in r['state']['memory'] for r in end) else None,
              'peak_reserved_rank_max_bytes':max((r['state']['memory']['peak_reserved_bytes'] for r in end),default=None)
                  if all('peak_reserved_bytes' in r['state']['memory'] for r in end) else None,
              'new_precision_policy_accepted':False}
        if index>1:assert loads,'Warm compiler-cache request must still prove native weight reload'
        cases.append(case)
    assert sum(c['original_strict_Half_checks_passed'] for c in cases)==182280
    return {'scope':__doc__,'completed':True,'same_worker_and_model_objects_all_three_requests':True,
            'engine_initialization_s':result['engine_initialization_s'],'actual_full_requests':3,
            'total_strict_Half_checks':182280,'cases':cases,
            'warm_requests_have_native_stage_reload_not_full_model_residency':True,
            'third_request_keeps_existing_compiler_cache':True,
            'no_new_runtime_or_precision_optimization_applied':True,
            'full_performance_goal_complete':False,'numeric_candidate_automatically_accepted':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists(),'Use fresh output to preserve historical receipt'
    result=summarize(args.root);args.output.write_text(json.dumps(result,indent=2)+'\n')
    brief={k:v for k,v in result.items() if k!='cases'}
    brief['cases']=[{k:v for k,v in c.items() if k not in ('whole_forward_hook','actual_loads_by_rank_and_component')}
                    for c in result['cases']]
    print(json.dumps(brief,indent=2))


if __name__=='__main__':main()
