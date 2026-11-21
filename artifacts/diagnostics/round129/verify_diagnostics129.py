#!/usr/bin/env python3
"""Verify original-only repeat/first-region evidence, not a numerical promotion."""
import hashlib
import json
from pathlib import Path
from summarize_diagnostics129 import summarize
from fingerprints129 import normalized_fingerprint

ROOT=Path(__file__).resolve().parent


def main():
    result=summarize(ROOT/'executed_diagnostics129b','engine_repeat129b')
    assert result==json.loads((ROOT/'summary129b.json').read_text())
    assert result['process_exit_code']==0 and result['actual_original155checks_total']==8680
    for trace in result['traces']:
        assert trace['all8_reports_completed']
        assert all(row['first']['module']=='time_embedder' and row['first']['boundary']=='output' for row in trace['first_boundary_changes_by_rank'])
        assert trace['all_input_values_and_layout_pre_post_stable'] and trace['CPU_master_metadata_and_SHA_stable']
        assert trace['resident_parameter_buffer_logical_words_stable'] and trace['FA_wrapper_errors']==0
    components=[json.loads(p.read_text()) for p in ROOT.glob('time_component_rank*129.json')]
    assert len(components)==8
    for item in components:
        assert item['completed'] and item['TP_actual']==2 and item['FP32_parameters_unchanged']
        base={row['region']:row for row in item['passes'][0]['all_regions']}
        for repetition in item['passes'][1:]:
            rows={row['region']:row for row in repetition['all_regions']}
            for region in ('frequency_construction','time_embedder.proj_in:local_projection','time_embedder.proj_in:post_TP'):
                assert normalized_fingerprint(rows[region])==normalized_fingerprint(base[region])
            region='time_embedder.proj_out:local_projection'
            assert normalized_fingerprint(rows[region]['input'])==normalized_fingerprint(base[region]['input'])
            assert normalized_fingerprint(rows[region]['output'])!=normalized_fingerprint(base[region]['output'])
    component=json.loads((ROOT/'time_component_summary129.json').read_text())
    assert component['source_sha256']==hashlib.sha256((ROOT/'time_component129.py').read_bytes()).hexdigest()
    assert component['changed_BEFORE_TP_reduce'] and not component['exact_inner_GEMM_or_accumulate_root_cause_known']
    cleanup=json.loads((ROOT/'cleanup129b.json').read_text())
    assert cleanup['completed'] and cleanup['all_private_exited'] and cleanup['private_key_removed']
    report={'verification_passed':True,'original8rank_7full50diagnostic_forwards':True,'strict_checks':8680,
        'same_inputs_and_weights_no_modification_proved_by_fingerprints':True,
        'actual_FA50launch_each_forward_no_errors':True,'first_changed_fullgraph_region':'time_embedder',
        'first_changed_smallcomponent_region':'time_embedder.proj_out local output BEFORE TP reduce',
        'time_output_max_delta':component['maximum_time_output_difference'],
        'time_output_max_relative_RMS':component['maximum_time_output_relative_RMS'],
        'inner_four_GEMMs_or_FP32_add_root_cause_still_unknown':True,
        'candidate_copy_stream_not_tested_or_promoted':True,'no_sampler_or_generation_steps':True,
        'no_precision_tolerance_relaxed':True,'a_failure_and_cleanup_not_hidden':True,
        'source_SHA_and_private_cleanup_verified':True,'whole_latency_goal_not_complete':True}
    path=ROOT/'verified_diagnostics129.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
