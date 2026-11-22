#!/usr/bin/env python3
"""Summarize executed audio first-change diagnostics, not a candidate tolerance."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

SOURCE=Path(__file__).resolve().parent


def summarize(root):
    process=json.loads((root/'engine_audio132a_process.json').read_text())
    lifecycle=json.loads((root/'engine_audio132a/diagnostic_lifecycle132.json').read_text())
    reports=[json.loads(p.read_text()) for p in root.glob('audio_tail_rank*132.json')]
    result={'scope':__doc__,'process_exit_code':process.get('exit_code'),'completion_unknown':process.get('completion_unknown',False),
        'lifecycle_completed':lifecycle['completed'],'engine_initialization_s':lifecycle.get('engine_initialization_s'),
        'model_output_scope':'Own-live step25 original50 prediction tensors, no49-step decoded media',
        'new_numeric_baseline_adopted':False,'copy_candidate_enabled':False,'generation_or_sampler_steps':0,
        'reported_ranks':len(reports),'phases':[],'original_strict_checks':0,'actual_FA_launches':0,'actual_FA_errors':0}
    for path,digest in process['source_sha256'].items():
        source=SOURCE/'executed_sources132a'/Path(path).name
        if not source.exists():source=SOURCE/Path(path).name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,('Executed source mismatch',path)
    result['exact_execution_source_sha_verified']=True
    if lifecycle.get('failure'):result['lifecycle_failure']=lifecycle['failure']
    if (root/'cohort_failure128.json').exists():result['cohort_failure']=json.loads((root/'cohort_failure128.json').read_text())
    if not reports:return result
    assert len({r['rank'] for r in reports})==len(reports) and all(0<=r['rank']<8 for r in reports)
    if lifecycle['completed']:
        assert sorted(r['rank'] for r in reports)==list(range(8)) and process['exit_code']==0
        assert all(r['completed'] and len(r['passes'])==9 and r['private_flags_physically_restored'] for r in reports)
    result['same8_actual_worker_pids']=sorted({r['pid'] for r in reports})
    result['actual_parallel_groups']=[{'rank':r['rank'],'TP':r['TP_actual'],'TP_group':r['TP_group'],
        'SP':r['SP_actual'],'SP_group':r['SP_group'],'Ulysses':r['Ulysses_actual'],'Ring':r['Ring_actual']} for r in reports]
    for report in reports:
        for row in report['passes']:
            assert row['strict_Half_checks']==155
            result['original_strict_checks']+=155;result['actual_FA_launches']+=row['actual_FA_activity']['actual_ACL_launch_count']
            result['actual_FA_errors']+=row['actual_FA_activity']['FA_wrapper_error_count']
    result['all_input_values_layout_unchanged']=all(p['input_values_and_layout_unchanged'] for r in reports for p in r['passes'])
    result['all_master_resident_state_unchanged']=all(r.get('original_master_and_resident_state_unchanged',False) for r in reports)
    result['all_private_flags_restored_false']=all(r.get('private_flags_physically_restored',False) and not r.get('final_runtime',{}).get('deterministic',True) for r in reports)
    for phase in ('native_first','true_control','true_tail_trace','true_restored','native_final'):
        per_rank=[[p for p in r['passes'] if p['phase']==phase] for r in reports]
        if not all(per_rank):continue
        count=min(len(rows) for rows in per_rank);rows=[p for group in per_rank for p in group]
        entry={'phase':phase,'completed_rank_repeats':count,'actual_flags':sorted({p['actual_deterministic_flag'] for p in rows}),
            'observer_enabled':any(p['trace_enabled'] for p in rows),
            'per_repeat_coordinated_rank_max_s':[max(g[index]['full50_original_audit_D2H_and_observers_s'] for g in per_rank) for index in range(count)],
            'modality_numeric':[]}
        if count>1:entry['median_of_per_repeat_coordinated_rank_max_s']=statistics.median(entry['per_repeat_coordinated_rank_max_s'][1:])
        else:entry['single_forward_not_self_repeat_proof']=True
        for modality,index in (('video',0),('audio',1)):
            numeric={'modality':modality}
            for key in ('output_vs_phase_first','output_vs_native_first','output_vs_true_control_first'):
                values=[p[key][index] for p in rows if p.get(key) is not None]
                if values:numeric[key]={'all_rank_words_equal':all(v['all_words_equal'] for v in values),
                    'max_different_FP32_words':max(v['different_FP32_words'] for v in values),
                    'max_abs_difference':max(v['max_abs_difference'] for v in values),
                    'max_relative_RMS_ratio':max(v['relative_RMS_error'] for v in values),
                    'max_relative_RMS_percent':100*max(v['relative_RMS_error'] for v in values)}
            entry['modality_numeric'].append(numeric)
        memories=[p['memory'] for p in rows if 'peak_allocated_bytes' in p['memory']]
        if memories:entry['rank_max_PyTorch_peak_allocated_GiB']=max(p['peak_allocated_bytes'] for p in memories)/(1024**3)
        result['phases'].append(entry)
    traced=[r for r in reports if r.get('tail_trace_compare')]
    if traced:
        result['first_tail_changes_by_rank']=[{'rank':r['rank'],'first_boundary':r['tail_trace_compare']['first_different_boundary']} for r in traced]
        proofs=[]
        for report in traced:
            local=report['paired_audio_local_apply'][-1];post=report['paired_audio_post_TP_gather'][-1]
            proofs.append({'rank':report['rank'],'full_input':local['full_input'],'audio_row_input':local['audio_row_input'],
                'weight':local['weight'],'bias':local['bias'],'local_full_output_preTP':local['full_local_output_before_TP_gather'],
                'local_audio_output_preTP':local['audio_row_local_output_before_TP_gather'],
                'full_output_postTP':post['full_post_TP_output'],'audio_output_postTP':post['audio_rows_post_TP_output']})
        result['complete_word_proofs_by_rank']=proofs
    locals_=[r for r in reports if r.get('fixed_full_input_audio_local_replay')]
    if locals_:
        result['actual_additional_local_audio_method_calls']=sum(r['fixed_full_input_audio_local_replay']['actual_local_method_calls'] for r in locals_)
        result['fixed_local_replay_by_rank']=[]
        for report in locals_:
            local=report['fixed_full_input_audio_local_replay'];rows=local['rows']
            result['fixed_local_replay_by_rank'].append({'rank':report['rank'],'repeats':len(rows),
                'input_shape':local['input_shape'],'all_full_outputs_bit_identical':all(p['full_output_vs_replay_first']['all_words_equal'] for p in rows),
                'all_audio_row_outputs_bit_identical':all(p['audio_rows_vs_replay_first']['all_words_equal'] for p in rows),
                'max_full_output_different_FP32_words':max(p['full_output_vs_replay_first']['different_FP32_words'] for p in rows),
                'max_audio_rows_different_FP32_words':max(p['audio_rows_vs_replay_first']['different_FP32_words'] for p in rows),
                'max_audio_rows_abs_difference':max(p['audio_rows_vs_replay_first']['max_abs_difference'] for p in rows),
                'max_audio_rows_relative_RMS_ratio':max(p['audio_rows_vs_replay_first']['relative_RMS_ratio'] for p in rows),
                'original_vs_frozen_input_all_words_equal':local['original_vs_frozen_full_word_proof']['input']['all_words_equal'],
                'dispatcher':local['dispatcher']})
    result['nonclean_trace_not_a_model_speedup']=True;result['first_different_device_kernel_not_attributed']=True
    if lifecycle['completed']:assert result['original_strict_checks']==11160 and result['actual_FA_launches']==3600 and result['actual_FA_errors']==0 and result['actual_additional_local_audio_method_calls']==80
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=summarize(args.root)
    if args.output.exists():assert json.loads(args.output.read_text())==result
    else:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
