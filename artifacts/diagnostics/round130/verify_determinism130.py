#!/usr/bin/env python3
"""Validate R130 deterministic-control receipts without adopting their numeric output."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parent


def main():
    reports=[json.loads(p.read_text()) for p in ROOT.glob('deterministic_rank*130.json')]
    assert len(reports)==8 and sorted(r['rank'] for r in reports)==list(range(8))
    source=ROOT/'executed_deterministic_probe130.py'
    assert source.read_bytes()==(ROOT/'deterministic_probe130.py').read_bytes()
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    summary=json.loads((ROOT/'determinism_summary130.json').read_text())
    assert summary['completed'] and not summary['candidate_policy_adopted']
    for report in reports:
        assert report['completed'] and report['TP_actual']==2 and len(report['TP_group'])==2
        assert report['source_sha256']==sha and not report['initial_flag']
        assert not report['initial_warn_only'] and report['restored_original_deterministic_flags']
        assert report['input_before']==report['input_after']
        assert [r['mode'] for r in report['cases']]==['native','deterministic','restored']
        for case in report['cases']:
            assert not case['errors']
            assert case['flag_actually_enabled']==(case['mode']=='deterministic')
            assert [r['op'] for r in case['operations']]==['mm0','mm1','mm2','mm3','full_original_guard']
            for op in case['operations']:
                assert op['actual_repeats']==10 and len(op['results'])==10 and op['operands_unchanged']
                stable=all(r['self_repeat']['all_words_equal'] for r in op['results'])
                assert op['all_same_input_self_outputs_bit_identical']==stable
                assert stable==(case['mode']=='deterministic')
    for mode_summary in summary['cases']:
        mode=mode_summary['mode']
        for op_summary in mode_summary['operations']:
            op=op_summary['op']
            rows=[next(o for o in next(c for c in report['cases'] if c['mode']==mode)['operations'] if o['op']==op) for report in reports]
            assert op_summary['stable_rank_groups']==sum(r['all_same_input_self_outputs_bit_identical'] for r in rows)
            for field,container,value in (
                ('max_self_repeat_abs_difference','self_repeat','max_abs_difference'),
                ('max_difference_to_original_first_words','versus_native_first','different_FP32_words'),
                ('max_difference_to_original_first_abs','versus_native_first','max_abs_difference'),
                ('max_difference_to_original_first_relative_RMS','versus_native_first','relative_RMS_error'),
            ):
                assert op_summary[field]==max(p[container][value] for r in rows for p in r['results'])
            assert op_summary['rank_max_median_explicit_sync_and_D2H_ms']==max(statistics.median(r['times_include_explicit_D2H_sync_s'][1:])*1000 for r in rows)
    cleanup=json.loads((ROOT/'cleanup130.json').read_text())
    assert cleanup['completed'] and cleanup['all_private_time_stage_and_probe_processes_exited']
    assert cleanup['all8_deterministic_flags_restored_before_exit']
    guard=next(o for o in summary['cases'][1]['operations'] if o['op']=='full_original_guard')
    assert guard['max_difference_to_original_first_words']==2084
    result={'verification_passed':True,'actual8TP2_ranks_same_operands':True,
        'native0_deterministic40_restored0_bit_repeatable_groups':True,'each_op_repeats':10,
        'deterministic_full_local_guard_max_abs_vs_native':guard['max_difference_to_original_first_abs'],
        'deterministic_full_local_guard_max_relative_RMS_vs_native':guard['max_difference_to_original_first_relative_RMS'],
        'deterministic_full_local_guard_max_FP32_words_changed_vs_native':guard['max_difference_to_original_first_words'],
        'full_original_local_guard_output_words':8064,'full50_model_quality_and_repeatability_unknown':True,
        'candidate_flag_not_adopted_or_deployed':True,'all_private_flags_and_functions_restored':True,
        'source_sha_and_scalar_summaries_verified':True,'requires_user_decision_before_full50_diagnostic':True}
    path=ROOT/'verified_determinism130.json'
    if path.exists():assert json.loads(path.read_text())==result, 'Do not overwrite an earlier incompatible verification receipt'
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
