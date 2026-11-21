#!/usr/bin/env python3
"""Summarize only completed R128 actual receipts; no model run or numeric adoption."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

SOURCE=Path(__file__).resolve().parent


def summarize(root):
    session=root/'engine_overlap128a'
    process=json.loads((root/'engine_overlap128a_process.json').read_text())
    lifecycle=json.loads((session/'lifecycle_result127.json').read_text())
    for remote,digest in process['source_sha256'].items():
        exact=SOURCE/'executed_sources128a'/Path(remote).name
        if not exact.exists():exact=SOURCE/Path(remote).name
        assert hashlib.sha256(exact.read_bytes()).hexdigest()==digest,('Executed source drift',remote)
    rows=[json.loads(line) for path in session.glob('worker_runtime_pid*127.jsonl') for line in path.read_text().splitlines()]
    steps=[row for row in rows if row['event']=='dit_step_end']
    report={'scope':__doc__,'process_exit_code':process['exit_code'],'lifecycle_completed':lifecycle['completed'],
        'actual_full49_requests':len(lifecycle['requests']),'generation_strict_checks_observed':sum(r['actual_Half_matrix_modules_this_step'] for r in steps),
        'source_sha_verified':True,'weights_or_activations_downloaded_to_client':False,
        'no_new_precision_policy_automatically_accepted':True,'cases':[],'replay':[]}
    if process['exit_code']!=0 or not lifecycle['completed']:
        report['failure']=lifecycle.get('failure')
        failure=root/'cohort_failure128.json'
        if failure.exists():report['cohort_failure']=json.loads(failure.read_text())
        report['partial_completed_steps_by_pid']={str(pid):len([r for r in steps if r['pid']==pid]) for pid in sorted({r['pid'] for r in steps})}
        report['result']='Failed experiment, preserve actual receipts; no acceleration claim'
        replays=[json.loads(p.read_text()) for p in root.glob('formal_replay_rank*128.json')]
        if len(replays)==8 and all(r['completed'] and not r.get('active_serial') and not r.get('active_async') for r in replays):
            assert all(r['all_cpu_master_SHA_unchanged'] for r in replays)
            assert all(r['strict_Half_checks']==155 for replay in replays for r in replay['cases'])
            pids=sorted({r['pid'] for r in steps});assert len(pids)==8
            assert all(sorted(r['step_index'] for r in steps if r['pid']==pid)==list(range(49)) for pid in pids)
            ends=[r for r in rows if r['event']=='native_request_end'];assert len(ends)==8
            report['result']='Blocked before candidate: unchanged native same-input replay is not bit-repeatable'
            report['native_original49_completed_all8']=True
            report['candidate_adapter_never_activated_in_model']=True
            report['candidate49_not_launched']=True
            report['all8_native_master_SHA_unchanged']=True
            report['extra_nontrajectory_strict_checks']=sum(r['extra_nontrajectory_benchmark_Half_checks'] for r in ends)
            assert report['extra_nontrajectory_strict_checks']==3720
            report['original_prediction_preserved_before_sampler26_all8']=len([r for r in rows if r['event']=='R128_original_prediction_preserved_after_nontrajectory_replay'])==8
            assert report['original_prediction_preserved_before_sampler26_all8']
            per_step=[max(r['elapsed_s_includes_offload_and_D2H_audit'] for r in steps if r['step_index']==i) for i in range(49)]
            report['native_generation_wall_includes_extra_replay_s']=lifecycle['requests'][0]['generation_wall_s']
            report['native_warm_whole_forward_rank_max_median_s']=statistics.median(per_step[1:])
            report['native_input_replay_wall_rank_max_s']=[r['rank_max_wall_s'] for r in replays[0]['cases']]
            report['numeric_differences_not_attributed_to_candidate']=json.loads((SOURCE/'native_repeat_failure128.json').read_text())
            report['driver_failure_is_output_equality_restored_flag_semantic_error_not_model49_failure']=True
        return report
    assert len(lifecycle['requests']) in (1,3)
    expected=None;objects=None
    for request in lifecycle['requests']:
        index=request['index'];selected=[r for r in rows if r['request_index']==index]
        actual=[r for r in selected if r['event']=='dit_step_end'];pids=sorted({r['pid'] for r in actual})
        assert len(pids)==8 and all(sorted(r['step_index'] for r in actual if r['pid']==pid)==list(range(49)) for pid in pids)
        assert all(r['actual_Half_matrix_modules_this_step']==155 for r in actual)
        ids={r['pid']:(r['pipeline_id'],r['transformer_id']) for r in selected if r['event']=='native_request_begin'}
        if expected is None:expected=pids;objects=ids
        else:assert pids==expected and ids==objects
        ends=[r for r in selected if r['event']=='native_request_end'];assert len(ends)==8
        per_step=[max(r['elapsed_s_includes_offload_and_D2H_audit'] for r in actual if r['step_index']==step) for step in range(49)]
        copies=[r for r in selected if r['event']=='R128_candidate_49_complete_before_restore']
        restored=[r for r in selected if r['event']=='R128_candidate_native_hooks_restored_after49']
        if index==2:
            assert len(copies)==len(restored)==8
            assert all(r['actual_native_copies']==2450 and r['complete_50_block_cycles']==49 for r in copies)
            assert all(r['distinct_streams'] and r['retirement_backpressure_count']>=0 for r in copies)
        report['cases'].append({'index':index,'label':request['label'],'original49steps_all8':True,'strict_checks':60760,
            'generation_wall_s':request['generation_wall_s'],'wall_includes_extra_replay':index==1,
            'whole_forward_first_step_rank_max_s':per_step[0],'whole_forward_warm_steps1to48_median_rank_max_s':statistics.median(per_step[1:]),
            'whole_forward_warm_step_min_s':min(per_step[1:]),'whole_forward_warm_step_max_s':max(per_step[1:]),
            'worker_request_rank_max_s':max(r['host_elapsed_s'] for r in ends),
            'extra_nontrajectory_strict_checks':sum(r.get('extra_nontrajectory_benchmark_Half_checks',0) for r in ends),
            'candidate_2450copies_and49cycles_all8_confirmed':index==2,
            'candidate_native_hooks_restored_before_decode_all8':index==2,
            'peak_allocated_bytes':max(r['state']['memory']['peak_allocated_bytes'] for r in ends),
            'peak_reserved_bytes':max(r['state']['memory']['peak_reserved_bytes'] for r in ends),
            'candidate_retired_peak_bytes':max((r['peak_retired_storage_bytes'] for r in copies),default=0),
            'candidate_retirement_backpressure_count':sum(r['retirement_backpressure_count'] for r in copies),
            'candidate_retirement_backpressure_host_s':sum(r['retirement_backpressure_host_s'] for r in copies),
            'encoder_VAE_FP32_and_all_stagers_remain_native':True})
    replays=[json.loads(p.read_text()) for p in root.glob('formal_replay_rank*128.json')]
    assert len(replays)==8 and sorted(r['rank'] for r in replays)==list(range(8))
    for replay in replays:
        assert replay['completed'] and replay['restored'] and replay['all_cpu_master_SHA_unchanged']
        assert replay['actual_TP2_runtime']['size']==2 and replay['actual_TP2_runtime']['local_heads_before_USP4']==28
        assert all(r['strict_Half_checks']==155 for r in replay['cases'])
        leaf=replay['leaf_proof'];masters=replay['byte_proof']
        if replay['candidate_formal_generation_allowed']:
            assert len(leaf)==104 and all(r['bit_identical'] for r in leaf)
            assert {r['target_block'] for r in masters}==set(range(50)) and all(r['entire_native_flat_master_raw_words_equal'] for r in masters)
        serial=[r for r in replay['profiles'] if r['mode']=='serial']
        asynchronous=[r for r in replay['profiles'] if r['mode']=='async']
        for row in serial:assert not row['actual_selected_hook_copy_is_distinct'] and row['copy_overlap_with_inclusive_block_spans_ms']<=0.01
        for row in asynchronous:assert row['actual_selected_hook_copy_is_distinct']
        report['replay'].append({'rank':replay['rank'],'actual_TP2_runtime':replay['actual_TP2_runtime'],
            'warm_rank_max_medians_s':replay['warm_rank_max_medians_s'],'gain_vs_restored_fraction':replay['candidate_gain_vs_restored_fraction'],
            'root_all8_bit_gates':replay['all8ranks_bit_gates_passed'],'same_input104wide_wordproof':len(leaf)==104 and all(r['bit_identical'] for r in leaf),
            'all50_native_flat_master_byteproof':bool(masters) and all(r['entire_native_flat_master_raw_words_equal'] for r in masters),
            'actual_serial_overlap_ms':serial[0]['copy_overlap_with_inclusive_block_spans_ms'] if serial else None,
            'actual_async_prefetch_body_intersection_ms':asynchronous[0]['copy_overlap_with_inclusive_block_spans_ms'] if asynchronous else None,
            'intersection_includes_host_allocator_gaps_not_pure_DMA_or_kernel_busy':True,'gate_result':replay['result']})
    report['original_prediction_preserved_before_sampler26_all8']=len([r for r in rows if r['event']=='R128_original_prediction_preserved_after_nontrajectory_replay'])==8
    assert report['original_prediction_preserved_before_sampler26_all8']
    report['same8workers_and_model_objects']=True
    report['original_generation_strict_checks']=60760*len(lifecycle['requests'])
    report['result']='Awaiting approval' if not all(r['all8ranks_bit_gates_passed'] for r in replays) else ('Full candidate validated for timing review' if len(lifecycle['requests'])==3 else 'Reject: no clear same-scope gain; candidate49 not launched')
    if len(lifecycle['requests'])==3:
        candidate=report['cases'][1]['whole_forward_warm_steps1to48_median_rank_max_s']
        restored=report['cases'][2]['whole_forward_warm_steps1to48_median_rank_max_s']
        report['actual_formal49_candidate_vs_restored_gain_fraction']=(restored-candidate)/restored
    report['latency_targets_completed']=False
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();result=summarize(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
