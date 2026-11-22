#!/usr/bin/env python3
"""Verify actual full50 candidate control and numeric receipts; never adopt them."""
import argparse
import json
from pathlib import Path
from fingerprints129 import normalized_fingerprint
from summarize135 import summarize

SOURCE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    summary=summarize(args.root)
    assert summary['lifecycle_completed'] and summary['process_exit_code']==0 and not summary['completion_unknown']
    assert summary['strict_Half_checks']==11160 and summary['actual_FA_launches']==3600 and summary['actual_FA_errors']==0
    assert len(summary['actual_same8_worker_pids'])==8 and summary['actual_candidate_Python_local_calls']==24
    assert summary['input_logical_values_and_layout_stable'] and summary['masters_and_resident_parameters_buffers_stable']
    assert summary['all_candidate_methods_physically_restored'] and summary['all_private_flags_restored_false'] and summary['original_master_reference_verified']
    process=json.loads((args.root/'engine_candidate135a_process.json').read_text())
    assert process['all_private_children_exited'] and not process.get('cohort_monitor_errors') and not process.get('cleanup_failure')
    assert not process.get('private_children_after_parent_exit')
    reports=sorted([json.loads(p.read_text()) for p in args.root.glob('full50_rank*135.json')],key=lambda r:r['rank'])
    for r in reports:
        rank=r['rank'];assert r['completed'] and r['TP_actual']==2 and r['SP_actual']==4 and r['Ulysses_actual']==4 and r['Ring_actual']==1
        assert r['TP_group']==[rank//2*2,rank//2*2+1] and r['SP_group']==list(range(rank%2,8,2)) and r['actual_local_heads']==28
        assert not r['final_runtime']['deterministic'] and not r['final_runtime']['warn_only']
        assert not r['final_runtime']['HF32_matmul'] and not r['final_runtime']['HF32_conv'] and r['candidate_actual_local_calls']==3
        assert r['model_before_state']['native_masters']==r['model_after_state']['native_masters']
        assert normalized_fingerprint(r['model_before_state']['materialized_parameters_and_buffers'])==normalized_fingerprint(r['model_after_state']['materialized_parameters_and_buffers'])
        assert [p['case'] for p in r['passes']]==['original']*3+['candidate']*3+['restored']*3
        assert [p['repeat'] for p in r['passes']]==[0,1,2]*3
        assert [p['actual_candidate_calls_so_far'] for p in r['passes']]==[0,0,0,1,2,3,3,3,3]
        contract=r['candidate_actual_contract'];assert contract['operand_dtype']==contract['weight_dtype']==contract['returned_dtype']=='torch.float16'
        assert contract['GEMM_output_dtype']==contract['bias_add_dtype']=='torch.float32' and contract['one_final_narrowing']
        assert contract['input_shape']==[3,2688] and contract['local_output_shape']==[3,5376]
        for p in r['passes']:
            assert p['strict_Half_checks']==155 and p['actual_deterministic_flag'] and p['runtime']['deterministic'] and not p['runtime']['warn_only']
            assert p['candidate_adapter_installed']==(p['case']=='candidate')
            assert normalized_fingerprint(p['input_before'])==normalized_fingerprint(p['input_after'])
            fa=p['actual_FA_activity'];assert fa['actual_ACL_launch_count']==50 and fa['FA_wrapper_error_count']==0
            shapes=[a['q_shape'] for a in fa['activity'] if a['kind']=='FA.wrapper'];assert len(shapes)==50
            assert all(shape==[1,7,41792,128] for shape in shapes)
            for comparisons in (p['output_vs_same_case_first'],p['output_vs_original_first']):
                assert len(comparisons)==2
                assert all(c['dtype']=='torch.float32' for c in comparisons)
                assert comparisons[0]['shape']==[39312,96] and comparisons[1]['shape']==[414,32]
            if p['case']=='original':assert p['output_vs_original_last'] is None and p['time_vs_original_last'] is None
    proof={'verified_actual_full50_candidate_diagnostic':True,'same8_workers9_full50_forwards_each':True,
        'actual_strict_Half_checks':11160,'actual_FA_launches':3600,'actual_FA_errors':0,'actual_candidate_local_Python_calls':24,
        'TP2_USP4_and_actual_FA_heads_verified':True,'original_cpu_masters_resident_parameters_buffers_and_input_unchanged':True,
        'all_candidate_methods_and_initial_false_warn_flags_physically_restored':True,
        'observed_case_repeatability':[{'case':c['case'],'modalities':[{'modality':m['modality'],
            'bit_repeatable':m['output_vs_same_case_first']['all_rank_words_equal']} for m in c['modalities']]} for c in summary['cases']],
        'same_flag_original_reference_not_default_false_reference':True,'exact_execution_source_SHA_verified':True,
        'all_private_children_exited':True,'candidate_not_adopted_and_no_new_tolerance':True,
        'no49_step_media_or_copy_candidate_quality_claim':True,'performance_roadmap_still_open':True}
    path=SOURCE/'verified135.json'
    if path.exists():assert json.loads(path.read_text())==proof
    else:path.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
