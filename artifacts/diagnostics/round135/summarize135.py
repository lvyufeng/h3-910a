#!/usr/bin/env python3
"""Summarize actual full50 candidate diagnostics; no tolerance or baseline adoption."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

SOURCE=Path(__file__).resolve().parent


def summarize(root):
    process=json.loads((root/'engine_candidate135a_process.json').read_text());life=json.loads((root/'engine_candidate135a/diagnostic_lifecycle135.json').read_text())
    ranks=sorted([json.loads(p.read_text()) for p in root.glob('full50_rank*135.json')],key=lambda r:r['rank'])
    result={'scope':__doc__,'process_exit_code':process.get('exit_code'),'completion_unknown':process.get('completion_unknown',False),
        'lifecycle_completed':life['completed'],'engine_initialization_s':life.get('engine_initialization_s'),
        'model_output_scope':'Own-live step25 full50 predictions, not49-step decoded media',
        'all_case_flag_scope':'Same temporaryTrue; original reference is not defaultFalse accepted generation',
        'candidate_not_adopted':True,'new_numeric_tolerance':False,'copy_candidate_enabled':False,'sampler_steps':0,
        'reported_ranks':len(ranks),'strict_Half_checks':0,'actual_FA_launches':0,'actual_FA_errors':0,'cases':[]}
    for path,digest in process['source_sha256'].items():
        p=SOURCE/'executed_sources135a'/Path(path).name
        if not p.exists():p=SOURCE/Path(path).name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('Executed source mismatch',path)
    result['exact_execution_source_SHA_verified']=True
    if life.get('failure'):result['lifecycle_failure']=life['failure']
    if (root/'cohort_failure128.json').exists():result['cohort_failure']=json.loads((root/'cohort_failure128.json').read_text())
    if not ranks:return result
    assert len({r['rank'] for r in ranks})==len(ranks)
    if life['completed']:
        assert len(ranks)==8 and [r['rank'] for r in ranks]==list(range(8)) and process['exit_code']==0
        assert all(r['completed'] and len(r['passes'])==9 for r in ranks)
    result['actual_same8_worker_pids']=sorted({r['pid'] for r in ranks})
    result['actual_groups_by_rank']=[{'rank':r['rank'],'TP':r['TP_actual'],'TP_group':r['TP_group'],'SP':r['SP_actual'],
        'SP_group':r['SP_group'],'Ulysses':r['Ulysses_actual'],'Ring':r['Ring_actual'],'local_heads':r['actual_local_heads']} for r in ranks]
    result['input_logical_values_and_layout_stable']=all(p['input_values_and_layout_pre_post_equal'] for r in ranks for p in r['passes'])
    result['input_allocation_metadata_equal_in_saved_fingerprints']=all(p['input_before']==p['input_after'] for r in ranks for p in r['passes'])
    result['masters_and_resident_parameters_buffers_stable']=all(r.get('native_CPU_masters_words_metadata_unchanged',False) and r.get('materialized_parameters_buffers_unchanged',False) for r in ranks)
    result['original_master_reference_verified']=all(r.get('original_master_reference_verified',{}).get('all50_original_R129_master_logical_words_and_metadata_same',False) for r in ranks)
    result['all_candidate_methods_physically_restored']=all(r.get('candidate_method_physically_restored',False) for r in ranks)
    result['all_private_flags_restored_false']=all(r.get('private_flags_physically_restored',False) and not r.get('final_runtime',{}).get('deterministic',True) for r in ranks)
    result['actual_candidate_Python_local_calls']=sum(r.get('candidate_actual_local_calls',0) for r in ranks)
    result['candidate_call_counts_are_not_CANN_launch_proof']=True
    for report in ranks:
        for row in report['passes']:
            assert row['strict_Half_checks']==155
            result['strict_Half_checks']+=155;result['actual_FA_launches']+=row['actual_FA_activity']['actual_ACL_launch_count'];result['actual_FA_errors']+=row['actual_FA_activity']['FA_wrapper_error_count']
    for case in ('original','candidate','restored'):
        per_rank=[[p for p in r['passes'] if p['case']==case] for r in ranks]
        if not all(per_rank):continue
        n=min(len(r) for r in per_rank);rows=[p for group in per_rank for p in group]
        item={'case':case,'completed_rank_repeats':n,'actual_flags':sorted({p['actual_deterministic_flag'] for p in rows}),
            'per_repeat_coordinated_rank_max_s':[max(group[i]['local_full50_including_original_audit_and_output_D2H_s'] for group in per_rank) for i in range(n)],'modalities':[]}
        if n>1:item['warm_median_of_coordinated_per_repeat_rank_max_s']=statistics.median(item['per_repeat_coordinated_rank_max_s'][1:])
        for modality,index in (('video',0),('audio',1)):
            values={'modality':modality}
            for key in ('output_vs_same_case_first','output_vs_original_first','output_vs_original_last','output_vs_candidate_first'):
                comparisons=[p[key][index] for p in rows if p.get(key) is not None]
                if comparisons:values[key]={'all_rank_words_equal':all(c['all_words_equal'] for c in comparisons),
                    'max_different_FP32_words':max(c['different_FP32_words'] for c in comparisons),'max_abs_difference':max(c['max_abs_difference'] for c in comparisons),
                    'max_relative_RMS_ratio':max(c['relative_RMS_error'] for c in comparisons),'max_relative_RMS_percent':100*max(c['relative_RMS_error'] for c in comparisons)}
            item['modalities'].append(values)
        for key in ('time_vs_same_case_first','time_vs_original_first'):
            comparisons=[p[key] for p in rows]
            item[key]={'all_rank_words_equal':all(c['all_words_equal'] for c in comparisons),'max_different_FP32_words':max(c['different_FP32_words'] for c in comparisons),
                'max_abs_difference':max(c['max_abs_difference'] for c in comparisons)}
        memories=[p['memory'] for p in rows if 'peak_allocated_bytes' in p['memory']]
        if memories:item['rank_max_PyTorch_peak_allocated_GiB']=max(p['peak_allocated_bytes'] for p in memories)/(1024**3)
        result['cases'].append(item)
    if life['completed']:
        assert result['strict_Half_checks']==11160 and result['actual_FA_launches']==3600 and result['actual_FA_errors']==0 and result['actual_candidate_Python_local_calls']==24
    result['nonclean_forward_scope_not_49step_media_or_quality_claim']=True
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=summarize(args.root)
    if args.output.exists():assert json.loads(args.output.read_text())==result
    else:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
