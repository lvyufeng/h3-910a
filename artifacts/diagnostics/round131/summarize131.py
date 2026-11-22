#!/usr/bin/env python3
"""Summarize actual full50 diagnostics; no tolerance, flag or baseline adoption."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

SOURCE=Path(__file__).resolve().parent


def summarize(root):
    process=json.loads((root/'engine_deterministic131a_process.json').read_text())
    lifecycle=json.loads((root/'engine_deterministic131a/diagnostic_lifecycle131.json').read_text())
    reports=[json.loads(p.read_text()) for p in root.glob('full50_rank*131.json')]
    result={'scope':__doc__,'process_exit_code':process.get('exit_code'),'completion_unknown':process.get('completion_unknown',False),
        'lifecycle_completed':lifecycle['completed'],'engine_initialization_s':lifecycle.get('engine_initialization_s'),
        'model_output_scope':'Prediction tensors of full50 forward on own-live step25 input, not49-step sampled/decoded media',
        'candidate_adopted':False,'copy_candidate_enabled':False,'generation_or_sampler_steps':0,
        'cases':[],'total_original155checks':0,'reported_rank_count':len(reports)}
    if lifecycle.get('failure'):result['lifecycle_failure']=lifecycle['failure']
    if (root/'cohort_failure128.json').exists():result['cohort_failure']=json.loads((root/'cohort_failure128.json').read_text())
    for path,digest in process['source_sha256'].items():
        source=SOURCE/'executed_sources131a'/Path(path).name
        if not source.exists():source=SOURCE/Path(path).name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,('Executed source mismatch',path)
    result['executed_source_sha_verified']=True
    if not reports:return result
    assert len({r['rank'] for r in reports})==len(reports) and all(0<=r['rank']<8 for r in reports)
    if lifecycle['completed']:
        assert sorted(r['rank'] for r in reports)==list(range(8))
        assert len(reports)==8 and all(r['completed'] and len(r['passes'])==9 for r in reports)
        assert process['exit_code']==0 and all(r['private_flags_physically_restored'] for r in reports)
    result['same8_actual_worker_pids']=sorted({r['pid'] for r in reports})
    result['actual_groups_by_rank']=[{'rank':r['rank'],'TP_group':r['TP_group'],'SP_group':r['SP_group'],
        'TP':r['TP_actual'],'SP':r['SP_actual'],'Ulysses':r['Ulysses_actual'],'Ring':r['Ring_actual'],'local_heads':r['actual_local_heads']} for r in reports]
    result['all_master_reference_checks_verified']=all(r.get('original_master_reference_verified',{}).get('all50_original_R129_master_logical_words_and_metadata_same',False) for r in reports)
    result['all_input_before_after_stable']=all(p['input_values_and_layout_pre_post_equal'] for r in reports for p in r['passes'])
    result['all_original_master_and_resident_state_unchanged']=all(r.get('native_CPU_masters_words_metadata_unchanged',False) and r.get('materialized_parameters_buffers_unchanged',False) for r in reports)
    result['all_private_flags_restored']=all(r.get('private_flags_physically_restored',False) for r in reports)
    result['actual_FA_launches']=sum(p['actual_FA_activity']['actual_ACL_launch_count'] for r in reports for p in r['passes'])
    result['actual_FA_errors']=sum(p['actual_FA_activity']['FA_wrapper_error_count'] for r in reports for p in r['passes'])
    for report in reports:
        for row in report['passes']:
            assert row['strict_Half_checks']==155
            assert row['actual_FA_activity']['actual_ACL_launch_count']==50 and row['actual_FA_activity']['FA_wrapper_error_count']==0
            result['total_original155checks']+=155
    for case in ('native','deterministic','restored'):
        per_rank=[[p for p in r['passes'] if p['case']==case] for r in reports]
        if not all(per_rank):continue
        repeats=min(len(rows) for rows in per_rank);rows=[p for group in per_rank for p in group]
        entry={'case':case,'complete_rank_repeats':repeats,'actual_flags':sorted({p['actual_deterministic_flag'] for p in rows}),
            'all8_video_audio_same_case_repeats_bit_identical':all(all(v['all_words_equal'] for v in p['output_vs_same_case_first']) for p in rows),
            'all8_time_same_case_repeats_bit_identical':all(p['time_vs_same_case_first']['all_words_equal'] for p in rows),
            'per_repeat_actual_rank_max_forward_s':[max(rank_rows[i]['local_full50_including_original_audit_and_output_D2H_s'] for rank_rows in per_rank) for i in range(repeats)],
            'numeric_max_by_modality':[]}
        if repeats>1:entry['warm_median_of_coordinated_per_repeat_rank_max_s']=statistics.median(entry['per_repeat_actual_rank_max_forward_s'][1:])
        for modality,index in (('video',0),('audio',1)):
            modality_row={'modality':modality}
            for key in ('output_vs_same_case_first','output_vs_native_first','output_vs_native_last','output_vs_deterministic_first'):
                values=[p[key][index] for p in rows if p.get(key) is not None]
                if values:modality_row[key]={'max_relative_RMS_ratio':max(v['relative_RMS_error'] for v in values),
                    'max_relative_RMS_percent':100*max(v['relative_RMS_error'] for v in values),
                    'max_abs_difference':max(v['max_abs_difference'] for v in values),'max_different_FP32_words':max(v['different_FP32_words'] for v in values),
                    'all_words_equal_all_rank_repeats':all(v['all_words_equal'] for v in values)}
            entry['numeric_max_by_modality'].append(modality_row)
        for key in ('time_vs_same_case_first','time_vs_native_first','time_vs_native_last'):
            values=[p[key] for p in rows if p.get(key) is not None]
            if values:entry[key]={'max_relative_RMS_ratio':max(v['relative_RMS_error'] for v in values),
                'max_abs_difference':max(v['max_abs_difference'] for v in values),'max_different_FP32_words':max(v['different_FP32_words'] for v in values),
                'all_words_equal_all_rank_repeats':all(v['all_words_equal'] for v in values)}
        memories=[p['memory'] for p in rows if 'peak_allocated_bytes' in p['memory']]
        if memories:entry['rank_max_peak_allocated_GiB']=max(p['peak_allocated_bytes'] for p in memories)/(1024**3)
        else:entry['memory_unavailable']=True
        result['cases'].append(entry)
    result['new_numeric_tolerance_granted']=False
    result['clean_performance_or_decoded_media_quality_claimed']=False
    result['flag_scope_process_wide_not_time_only']=True
    if lifecycle['completed']:assert result['total_original155checks']==11160 and result['actual_FA_launches']==3600
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.root)
    if args.output.exists():assert json.loads(args.output.read_text())==result,'Preserve incompatible prior receipt'
    else:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
