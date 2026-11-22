#!/usr/bin/env python3
"""Verify and summarize original small final-AdaLN repetitions, no baseline adoption."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    reports=[json.loads(p.read_text()) for p in ROOT.glob('component_rank*133.json')]
    process=json.loads((ROOT/'component_process133.json').read_text());assert process['exit_code']==0 and process['all_private_processes_exited']
    assert len(reports)==8 and sorted(r['rank'] for r in reports)==list(range(8))
    for name,digest in process['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    summary={'verification_passed':True,'same_original_input_SHA_and_parameter_reference_verified':True,
        'native_config_factory_used_before_constructing_original_components':True,
        'CANN_kernel_and_formal_AdaLN_full_runtime_context_not_identified':True,'new_arithmetic_or_flag_baseline_adopted':False,
        'no_full50_or_155checks_or49media_execution':True,'process_exit_code':0,'ranks':[]}
    for report in reports:
        assert report['completed'] and report['actual_formal_t_emb_SHA_matched'] and report['original_parameters_unchanged'] and report['private_flags_restored_false']
        assert report['TP_actual']==2 and report['SP_actual']==4 and report['Ulysses_actual']==4
        assert report['source_input_SHA_verified'] and report['same_python_default_dispatch_as_R132_audio_head']
        assert len(report['module_repeats'])==32 and len(report['local_repeats'])==128 and len(report['restored_module_repeats'])==4
        assert report['original_vs_frozen_input']['all_words_equal']
        assert report['actual_local_x_weight_bias_before']==report['actual_local_x_weight_bias_after']
        row={'rank':report['rank'],'TP_group':report['TP_group'],'SP_group':report['SP_group'],
            'actual_module_calls':32,'actual_local_calls_noTPgather':128,'actual_restored_module_calls':4,
            'module_input_changed_repeats':sum(not r['input']['all_words_equal'] for r in report['module_repeats']),
            'module_local_output_changed_repeats':sum(not r['local_before_TP_gather']['all_words_equal'] for r in report['module_repeats']),
            'module_postgather_changed_repeats':sum(not r['post_gather_output']['all_words_equal'] for r in report['module_repeats']),
            'max_module_local_different_Half_words':max(r['local_before_TP_gather']['different_words'] for r in report['module_repeats']),
            'max_module_local_abs_difference':max(r['local_before_TP_gather']['max_abs_difference'] for r in report['module_repeats']),
            'fixed_local_changed_repeats':sum(not r['output']['all_words_equal'] for r in report['local_repeats']),
            'fixed_local_max_different_Half_words':max(r['output']['different_words'] for r in report['local_repeats']),
            'fixed_local_max_abs_difference':max(r['output']['max_abs_difference'] for r in report['local_repeats']),
            'restored_module_changed_repeats':sum(not r['output']['all_words_equal'] for r in report['restored_module_repeats']),
            'first_fixed_local_changed_coordinates':next((r['output']['first_changed_coordinates'] for r in report['local_repeats'] if not r['output']['all_words_equal']),[])}
        summary['ranks'].append(row)
    summary['actual_total_module_calls_with_TP_gather']=36*8;summary['actual_total_local_only_calls']=128*8
    summary['all_Module_inputs_bit_identical']=all(r['module_input_changed_repeats']==0 for r in summary['ranks'])
    summary['any_fixed_local_output_changed']=any(r['fixed_local_changed_repeats'] for r in summary['ranks'])
    target=ROOT/'summary133.json'
    if target.exists():assert json.loads(target.read_text())==summary
    else:target.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
