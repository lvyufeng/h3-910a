#!/usr/bin/env python3
"""Summarize supplemental step0/48 FA leaf gates, never as adoption/speed measurements."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ranks=[json.loads((HERE/'rank_receipts'/f'candidate_rank{rank}.json').read_text()) for rank in range(8)]
assert all(row['completed'] for row in ranks)
assert all(row['tp_actual']==2 and row['parallel_config']['ulysses_degree']==4 for row in ranks)
gates=[]
for rank in ranks:
    assert len(rank['checks'])==6
    assert all(row['safety']['matrix_checks']==155 and row['safety']['mode']=='strict' and row['safety']['nonfinite_count']==row['safety']['boundary_count']==0 for row in rank['checks'])
    assert {gate['step'] for gate in rank['gate_rows219']}=={0,48}
    stage=rank['gate_stage219']
    assert stage['restored'] and stage['candidate_results_returned_to_model']==0
    assert stage['candidate_measurements_or_adoption'] is False
    for gate in rank['gate_rows219']:
        gates.append({'rank':rank['rank'],'step':gate['step'],
                      'candidate_calls':gate['candidate_FA_calls'],'original_calls':gate['original_FA_calls'],
                      'candidate_ACL_launches':gate['candidate_ACL_launches'],
                      'elements':gate['compared_elements'],'different_words':gate['differing_words'],
                      'passed':gate['passed'],'all_rank_passed':gate['all_rank_passed'],
                      'original_outputs_returned':gate['gate_forward_returned_original_outputs'],
                      'strict_checks':gate['strict_checks']['matrix_checks'],
                      'per_layer':gate['per_layer']})
all_pass=all(g['passed'] and g['all_rank_passed'] and g['different_words']==0 and g['candidate_calls']==g['candidate_ACL_launches']==g['original_calls']==50 for g in gates)
binaries=[entry for r in ranks for entry in r['gate_stage219']['candidate_kernel']]
if all_pass:
    assert all(entry['sha256']=='052c49f2df2235bac2136b4630a05a71bd1b9b454d88c354bbbf5e254ab6e327' for entry in binaries)
    assert sum(g['candidate_calls'] for g in gates)==800
older=json.loads((HERE.parent/'round218'/'trim_model_summary218.json').read_text())
# Only merge the gate counts, not step timing or exact-trajectory claims.
summary={'round':219,'scope':'supplemental step0/48 gate-only eight-rank DiT isolated replay',
         'tp':2,'usp':4,'candidate_outputs_used_by_model':0,'candidate_default_enabled':False,
         'no_adoption_or_composition':True,'whole_goal_complete':False,
         'gates':gates,'candidate_kernel':binaries,
         'total_compared_calls':sum(g['candidate_calls'] for g in gates),
         'total_compared_elements':sum(g['elements'] for g in gates),
         'total_differing_words':sum(g['different_words'] for g in gates),
         'all_step0_48_gate_passed':all_pass,
         'including_prior_step25_calls':sum(g['candidate_calls'] for g in gates)+older['total_gate_calls'],
         'including_prior_step25_elements':sum(g['elements'] for g in gates)+older['total_gate_elements'],
         'previous_measured_step_gain_s_not_remeasured':older['measured_rank_max_step_gain_s'],
         'full_model_prediction_noise_not_a_tolerance':True,
         'all_155_strict_checks_preserved':True,'all_50_blocks_retained':True,
         'decision':'Awaiting user decision; full49step trajectory proof not done' if all_pass else 'Reject/stop candidate: exact leaf gate mismatch'}
(HERE/'gate_summary219.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('gates','candidate_kernel')},indent=2))
