#!/usr/bin/env python3
"""Verify actual authorized local numerical test; no tolerance or baseline adoption."""
from pathlib import Path
import hashlib
import json
import statistics
from fingerprints129 import normalized_fingerprint

ROOT=Path(__file__).resolve().parent


def main():
    process=json.loads((ROOT/'candidate_process134.json').read_text());assert process['exit_code']==0 and process['all_private_processes_exited']
    for name,digest in process['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    reports=sorted([json.loads(p.read_text()) for p in ROOT.glob('candidate_rank*134.json')],key=lambda r:r['rank'])
    assert len(reports)==8 and sorted(r['rank'] for r in reports)==list(range(8))
    result={'actual_execution_verified':True,'local_test_only_no_full50_or49media':True,'candidate_not_adopted':True,
        'new_numeric_tolerance':False,'actual_module_calls_with_TP_gather':3*32*8,'actual_direct_local_calls':3*128*8,
        'actual_candidate_successful_calls':160*8,'same_original_parameters_and_time_input_verified':True,'cases':[]}
    for r in reports:
        assert r['completed'] and r['candidate_actual_calls']==160 and r['TP_actual']==2 and r['SP_actual']==4 and r['Ulysses_actual']==4
        assert r['actual_formal_t_emb_SHA_matched'] and r['original_vs_frozen_input']['all_words_equal']
        assert r['actual_local_x_weight_bias_before']==r['actual_local_x_weight_bias_after']
        assert normalized_fingerprint(r['parameters_before'])==normalized_fingerprint(r['parameters_after'])
        assert [c['case'] for c in r['cases']]==['original','candidate','restored']
        assert r['physical_restore']['method_descriptor_restored'] and not r['physical_restore']['deterministic_flag'] and not r['physical_restore']['warn_only']
        contract=r['candidate_actual_operand_output_contract']
        assert contract['operand_dtype']==contract['weight_dtype']=='torch.float16'
        assert contract['GEMM_output_dtype']==contract['bias_add_dtype']=='torch.float32' and contract['returned_dtype']=='torch.float16'
        for c in r['cases']:
            assert c['actual_deterministic_flag'] and len(c['module_repeats'])==32 and len(c['local_repeats'])==128
            assert all(p['input_vs_original_first']['all_words_equal'] for p in c['module_repeats'])
    for case in ('original','candidate','restored'):
        cases=[next(c for c in r['cases'] if c['case']==case) for r in reports]
        local=[p for c in cases for p in c['local_repeats']];module=[p for c in cases for p in c['module_repeats']]
        row={'case':case,'all_local_self_repeats_bit_identical':all(p['output_vs_case_first']['all_words_equal'] for p in local),
            'all_Module_local_self_repeats_bit_identical':all(p['local_output_vs_case_first']['all_words_equal'] for p in module),
            'all_Module_postgather_self_repeats_bit_identical':all(p['postgather_vs_case_first']['all_words_equal'] for p in module),
            'local_changed_repeat_counts_by_rank':[sum(not p['output_vs_case_first']['all_words_equal'] for p in c['local_repeats']) for c in cases],
            'max_local_self_changed_Half_words':max(p['output_vs_case_first']['different_words'] for p in local),
            'max_local_self_abs_difference':max(p['output_vs_case_first']['max_abs_difference'] for p in local),
            'local_vs_original_first_max_Half_words_changed':max(p['output_vs_original_first']['different_words'] for p in local),
            'local_vs_original_first_max_abs_difference':max(p['output_vs_original_first']['max_abs_difference'] for p in local),
            'local_vs_original_first_max_relative_RMS_ratio':max(p['output_vs_original_first']['relative_RMS_ratio'] for p in local),
            'postgather_vs_original_first_max_Half_words_changed':max(p['postgather_vs_original_first']['different_words'] for p in module),
            'postgather_vs_original_first_max_abs_difference':max(p['postgather_vs_original_first']['max_abs_difference'] for p in module),
            'postgather_vs_original_first_max_relative_RMS_ratio':max(p['postgather_vs_original_first']['relative_RMS_ratio'] for p in module),
            'max_of_rank_local_medians_sync_D2H_s':max(statistics.median(p['explicit_sync_and_D2H_s'] for p in c['local_repeats'][1:]) for c in cases),
            'timing_not_clean_kernel_or_model_latency':True}
        result['cases'].append(row)
    result['all_private_flags_and_method_descriptor_restored']=True
    result['candidate_FP32_output_not_internal_accumulation_kernel_proof']=True
    target=ROOT/'summary134.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
