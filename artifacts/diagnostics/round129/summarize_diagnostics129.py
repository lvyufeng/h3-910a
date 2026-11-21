#!/usr/bin/env python3
"""Summarize actual original-only RPC controls/traces without promoting precision."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SOURCE=Path(__file__).resolve().parent


def summarize(root,label):
    process=json.loads((root/(label+'_process.json')).read_text())
    lifecycle=json.loads((root/label/'diagnostic_lifecycle129.json').read_text())
    result={'scope':__doc__,'label':label,'process_exit_code':process['exit_code'],'lifecycle_completed':lifecycle['completed'],
        'engine_initialization_s':lifecycle.get('engine_initialization_s'),'no_generation_or_sampler_steps':True,
        'copy_candidate_not_enabled':True,'no_new_precision_policy_accepted':True,'controls':[],'traces':[]}
    archive=SOURCE/('executed_sources129a' if label.endswith('a') else 'executed_sources129b')
    for path,digest in process['source_sha256'].items():
        local=archive/Path(path).name
        if not local.exists():local=SOURCE/Path(path).name
        assert hashlib.sha256(local.read_bytes()).hexdigest()==digest,('Executed source differs',path)
    result['exact_executed_sources_SHA_verified']=True
    total=0;pids=None
    for mode in ('repeat','trace','detailed'):
        reports=[json.loads(p.read_text()) for p in root.glob(mode+'_rank*129.json')]
        if not reports:continue
        assert len(reports)==8 and sorted(r['rank'] for r in reports)==list(range(8))
        current={r['pid'] for r in reports};assert len(current)==8
        if pids is None:pids=current
        else:assert pids==current
        completed=all(r['completed'] for r in reports)
        passes=min(len(r['passes']) for r in reports)
        if completed:assert passes== (3 if mode=='repeat' else 2)
        for report in reports:
            for row in report['passes']:
                assert row['strict_Half_checks']==155
                total+=155
                assert row['input_values_and_layout_pre_post_equal']
                assert row['actual_FA_activity']['actual_ACL_launch_count']==50
                assert row['actual_FA_activity']['FA_wrapper_error_count']==0
        item={'mode':mode,'all8_reports_completed':completed,'actual_completed_passes_each_rank':passes,
            'actual_checks':sum(sum(p['strict_Half_checks'] for p in r['passes']) for r in reports),
            'actual_FA_launches_each_pass':50,'FA_wrapper_errors':0,
            'all_input_values_and_layout_pre_post_stable':True,
            'CPU_master_metadata_and_SHA_stable':all(r.get('all_native_CPU_masters_metadata_and_words_same',False) for r in reports),
            'resident_parameter_buffer_logical_words_stable':all(r.get('all_materialized_resident_parameters_buffers_same',False) for r in reports),
            'per_pass_rank_max_delta':[]}
        for index in range(passes):
            values=[r['passes'][index] for r in reports]
            item['per_pass_rank_max_delta'].append({'index':index,'full_forward_time_includes_D2H_and_trace_s':max(v['local_full_forward_and_original_D2H_audit_s'] for v in values),
                'video_relative_RMS_error':max(v['output_comparison_to_pass0'][0]['relative_RMS_error'] for v in values),
                'audio_relative_RMS_error':max(v['output_comparison_to_pass0'][1]['relative_RMS_error'] for v in values),
                'video_max_abs_difference':max(v['output_comparison_to_pass0'][0]['max_abs_difference'] for v in values),
                'audio_max_abs_difference':max(v['output_comparison_to_pass0'][1]['max_abs_difference'] for v in values),
                'all8_video_audio_output_words_equal_to_pass0':all(all(x['all_words_equal'] for x in v['output_comparison_to_pass0']) for v in values)})
        if mode=='repeat':result['controls'].append(item)
        else:
            item['first_boundary_changes_by_rank']=[{'rank':r['rank'],'first':r['trace_compare'][0]['first_different_trace_boundary'] if r.get('trace_compare') else None} for r in reports]
            item['trace_SHA_is_region_fingerprint_not_complete_wordproof']=True
            item['trace_D2H_sync_changes_schedule_not_clean_performance']=True
            result['traces'].append(item)
    result['actual_original155checks_total']=total
    result['actual_same8worker_PIDs_across_RPCs']=sorted(pids) if pids else []
    failure=root/'cohort_failure128.json'
    if failure.exists():result['cohort_failure']=json.loads(failure.read_text())
    result['first_changing_operator_fully_attributed']=False
    result['candidate_tolerance_not_relaxed']=True
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--label',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();result=summarize(args.root,args.label)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
