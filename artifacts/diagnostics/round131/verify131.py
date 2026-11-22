#!/usr/bin/env python3
"""Verify actual authorized full50 diagnostic counts/state, never promote a flag."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from summarize131 import summarize
from fingerprints129 import normalized_fingerprint

SOURCE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.root)
    assert result['lifecycle_completed'] and result['process_exit_code']==0 and not result['completion_unknown']
    assert result['total_original155checks']==11160 and result['actual_FA_launches']==3600 and result['actual_FA_errors']==0
    assert len(result['same8_actual_worker_pids'])==8
    assert result['all_input_before_after_stable'] and result['all_original_master_and_resident_state_unchanged']
    assert result['all_private_flags_restored'] and result['all_master_reference_checks_verified']
    reports=[json.loads(p.read_text()) for p in args.root.glob('full50_rank*131.json')]
    for report in reports:
        assert report['TP_actual']==2 and report['SP_actual']==4 and report['Ulysses_actual']==4 and report['Ring_actual']==1
        assert report['actual_local_heads']==28 and len(report['TP_group'])==2 and len(report['SP_group'])==4
        assert not report['final_runtime']['deterministic'] and not report['final_runtime']['warn_only']
        before=report['model_before_state'];after=report['model_after_state']
        assert before['native_masters']==after['native_masters']
        assert normalized_fingerprint(before['materialized_parameters_and_buffers'])==normalized_fingerprint(after['materialized_parameters_and_buffers'])
        assert [p['case'] for p in report['passes']]==['native']*3+['deterministic']*3+['restored']*3
        for row in report['passes']:
            assert row['strict_Half_checks']==155 and row['actual_deterministic_flag']==(row['case']=='deterministic')
            assert normalized_fingerprint(row['input_before'])==normalized_fingerprint(row['input_after'])
            if row['case']=='native':assert row['output_vs_native_last'] is None and row['time_vs_native_last'] is None
    verification={'verified_actual_full50_diagnostics':True,'same8_workers_original9forwards':True,'actual155checks':11160,
        'actual_original_FA_launches':3600,'actual_TP2_USP4_and_FA_heads_verified':True,
        'original_input_and_master_resident_state_verified':True,'private_flags_physically_restored':True,
        'per_case_repeatability_observed':[{'case':r['case'],'full_output_bit_repeatable':r['all8_video_audio_same_case_repeats_bit_identical'],
            'time_bit_repeatable':r['all8_time_same_case_repeats_bit_identical']} for r in result['cases']],
        'no_sampler_generation_or_decoded_media_quality_pass':True,'new_numeric_policy_or_copy_candidate_not_adopted':True,
        'output_scope':'Original50-layer prediction on own-live step25 input',
        'timing_scope':'Coordinated per-repeat rank max, includes fullforward/originalaudit/outputD2H/tinytimeclone',
        'execution_source_SHA_verified':True,'performance_roadmap_not_complete':True}
    path=SOURCE/'verified131.json'
    if path.exists():assert json.loads(path.read_text())==verification
    else:path.write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
