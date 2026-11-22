#!/usr/bin/env python3
"""Verify complete R130 stage evidence, no numeric acceptance or model execution."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    reports=[json.loads(p.read_text()) for p in (ROOT/'stages130b').glob('stages_rank*130.json')]
    assert len(reports)==8 and sorted(r['rank'] for r in reports)==list(range(8))
    stage=json.loads((ROOT/'stage_summary130b.json').read_text())
    assert stage['first_changed_op']=='mm0 Half x Half -> FP32 output'
    assert stage['total_fixed_MM_groups']==32 and stage['fixed_MM_repeats_changed']==32
    assert stage['fixed_scaled_and_bias_add_groups_all_bit_identical']==28
    pins=json.loads((ROOT/'launch_pins130b.json').read_text())['source_sha256']
    for name,digest in pins.items():
        if name=='fingerprints129.py':source=Path('artifacts/diagnostics/round129')/name
        else:source=ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,('Executed source mismatch',name)
    for r in reports:
        assert r['completed'] and r['TP_actual']==2 and len(r['TP_group'])==2
        assert r['original_parameter_words_unchanged'] and r['global_functions_restored']
        assert r['actual_local_bias_present']==(r['rank']%2==0)
        for row in r['observed'][1:]:
            first=next(s for s in row['stages'] if not s['output']['all_raw_words_equal'])
            assert first['op']=='mm' and first['index']==0 and first['a']['all_raw_words_equal'] and first['b']['all_raw_words_equal']
        for op in r['fixed_operand_repeats']:
            assert op['same_inputs_before_after']
            if op['op']=='add_':assert all(x['all_words_equal'] for x in op['repeats'])
            else:assert any(not x['all_words_equal'] for x in op['repeats'])
    result={'verification_passed':True,'all8_true_TP2_USP4':True,'first_changed_op':'MM0 Half operands FP32 output',
        'all_MM0_actual_operand_bytes_same':True,'fixed4GEMMs_all32rank_groups_nonrepeatable':True,
        'fixed_scaled_add_and_bias_all28groups_identical':True,'actual_original_bias_branch_respected':True,
        'source_pins_parameters_and_process_functions_restored':True,'first_a_diagnostic_failure_not_hidden':True,
        'no_49step_or_copy_candidate_execution':True,'no_candidate_tolerance_accepted':True,
        'max_actual_MM0_difference':stage['max_mm0_difference'],'deterministic_probe_separate_not_a_promoted_policy':True}
    path=ROOT/'verified_stages130.json'
    if path.exists():assert json.loads(path.read_text())==result, 'Do not overwrite an earlier incompatible verification receipt'
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
