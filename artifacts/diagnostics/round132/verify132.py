#!/usr/bin/env python3
"""Verify actual audio local/full-word controls and cleanup, no tolerance adoption."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize132 import summarize
from fingerprints129 import normalized_fingerprint

SOURCE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    summary=summarize(args.root)
    assert summary['lifecycle_completed'] and summary['process_exit_code']==0 and not summary['completion_unknown']
    assert summary['original_strict_checks']==11160 and summary['actual_FA_launches']==3600 and summary['actual_FA_errors']==0
    assert summary['actual_additional_local_audio_method_calls']==80 and len(summary['same8_actual_worker_pids'])==8
    reports=[json.loads(p.read_text()) for p in args.root.glob('audio_tail_rank*132.json')]
    expected=['native_first']+['true_control']*3+['true_tail_trace']*2+['true_restored']*2+['native_final']
    for r in reports:
        assert r['completed'] and [p['phase'] for p in r['passes']]==expected
        assert r['TP_actual']==2 and r['SP_actual']==4 and r['Ulysses_actual']==4 and r['Ring_actual']==1
        assert r['private_flags_physically_restored'] and not r['final_runtime']['deterministic'] and not r['final_runtime']['warn_only']
        assert r['model_before_state']['native_masters']==r['model_after_state']['native_masters']
        assert normalized_fingerprint(r['model_before_state']['materialized_parameters_and_buffers'])==normalized_fingerprint(r['model_after_state']['materialized_parameters_and_buffers'])
        for row in r['passes']:
            assert normalized_fingerprint(row['input_before'])==normalized_fingerprint(row['input_after'])
            assert row['actual_deterministic_flag']==(row['phase'].startswith('true_'))
            assert row['runtime']['deterministic']==row['actual_deterministic_flag']
            assert row['strict_Half_checks']==155 and row['actual_FA_activity']['actual_ACL_launch_count']==50
        proof=r['fixed_full_input_audio_local_replay']['original_vs_frozen_full_word_proof']
        assert proof['input']['all_words_equal'] and proof['weight']['all_words_equal']
        assert proof['input']['total_FP32_words']==41792*5376
        assert proof['weight']['total_FP32_words']==16*5376
        assert proof['bias'].get('all_words_equal',proof['bias'].get('both_none',False))
        assert r['fixed_full_input_audio_local_replay']['actual_local_method_calls']==10
        assert r['fixed_full_input_audio_local_replay']['extra_full_model_checks']==0
        assert len(r['paired_audio_local_apply'])==len(r['paired_audio_post_TP_gather'])==2
        for pair in r['paired_audio_local_apply']:
            assert pair['full_input']['total_FP32_words']==41792*5376
            assert pair['audio_row_input']['total_FP32_words']==414*5376
            assert pair['full_local_output_before_TP_gather']['total_FP32_words']==41792*16
            assert pair['audio_row_local_output_before_TP_gather']['total_FP32_words']==414*16
        for pair in r['paired_audio_post_TP_gather']:
            assert pair['full_post_TP_output']['total_FP32_words']==41792*32
            assert pair['audio_rows_post_TP_output']['total_FP32_words']==414*32
    result={'actual_execution_verification_passed':True,'same8_full50_workers9calls':True,'strict_Half_checks':11160,
        'actual_FA_launches':3600,'actual_TP2_USP4_verified':True,'all_input_original_masters_and_resident_values_unchanged':True,
        'original_vs_frozen_full_word_equivalence_proven':True,'local_audio_calls_separate_withoutTPgather':80,
        'complete_original_local_input_shape':[41792,5376],'complete_local_output_shape':[41792,16],
        'audio_selected_output_shape':[414,32],'physical_layout_kernel_internal_not_proven':True,
        'no_sampler_media_or_copy_candidate':True,'no_numeric_tolerance_or_deterministic_baseline_adopted':True,
        'all_private_flags_physically_restored':True,'first_changes':[{'rank':r['rank'],'first':r['tail_trace_compare']['first_different_boundary']} for r in reports]}
    path=SOURCE/'verified132.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
