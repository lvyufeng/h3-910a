#!/usr/bin/env python3
"""Summarize independent eight-rank trim-only measurement after its real-input gate."""
import json
import statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
ranks=[json.loads((HERE/'trim_model_receipts'/f'candidate_rank{rank}.json').read_text()) for rank in range(8)]
assert all(r['completed'] for r in ranks)
assert all(r['tp_actual']==2 and r['parallel_config']['ulysses_degree']==4 for r in ranks)
gates=[]
for r in ranks:
    gate=r['trim_gate218']
    assert gate['actual_FA_calls']==gate['actual_candidate_ACL_launches']==50
    assert gate['all_rank_passed'] and gate['passed'] and gate['differing_words']==0
    assert gate['gate_forward_returned_original_outputs']
    assert gate['strict_checks']['matrix_checks']==155
    assert all(row['bit_identical'] and row['differing_words']==0 for row in gate['per_call'])
    assert len(r['checks'])==6
    assert all(row['safety']['matrix_checks']==155 and row['safety']['mode']=='strict' and row['safety']['nonfinite_count']==row['safety']['boundary_count']==0 for row in r['checks'])
    stage=r['trim_stage218']
    assert stage['restored'] and not stage['gate_failed']
    assert stage['actual_ACL_launches']['gate']==50
    assert stage['actual_ACL_launches']['measurement']==150 # one warmup plus two measured 50-block steps
    assert stage['calls']==stage['actual_ACL_launches']
    gates.append({'rank':r['rank'],'calls':50,'different_words':0,'elements':gate['compared_elements'],
                  'all_checks_retained':True,'candidate_ACL_launches_measured':100,
                  'candidate_ACL_launches_warmup':50,'restored':True,
                  'candidate_kernel':stage['candidate_kernel']})
cases=[]
for index,label in [(0,'baseline_pre'),(1,'trim_only_candidate'),(2,'restored_baseline')]:
    values=[row['rank_max_elapsed_s'] for row in ranks[0]['checks'] if row['case_index']==index]
    assert len(values)==2
    cases.append({'case':label,'rank_max_seconds':values,'rank_max_median_s':statistics.median(values),
                  'local_medians_by_rank_s':{str(r['rank']):statistics.median([row['local_elapsed_s'] for row in r['checks'] if row['case_index']==index]) for r in ranks},
                  'peak_allocated_bytes_by_rank':{str(r['rank']):max(row['peak_allocated_bytes'] for row in r['checks'] if row['case_index']==index) for r in ranks}})
baseline=cases[0]['rank_max_median_s'];candidate=cases[1]['rank_max_median_s'];restore=cases[2]['rank_max_median_s']
summary={'round':218,'scope':'single FA trim candidate at step25 in isolated8rank DiT, NOT full49step or live encoder pipeline',
         'TP2_USP4':True,'all50blocks_and_155strict_checks_retained':True,'only_FA_candidate_changed':True,
         'no_other_candidate_combined':True,'nothing_adopted':True,'whole_goal_complete':False,
         'cases':cases,'gate':gates,'total_gate_calls':sum(g['calls'] for g in gates),
         'total_gate_elements':sum(g['elements'] for g in gates),'total_different_words':0,
         'measured_rank_max_step_gain_s':baseline-candidate,
         'measured_rank_max_step_speedup':baseline/candidate,
         'measured_rank_max_step_gain_vs_restored_s':restore-candidate,
         'restore_time_ratio_to_pre':restore/baseline,
         'numerical_tolerance':None,'precision_gate_only_on_identical_real_leaf_inputs':True,
         'full_model_self_repeat_not_assumed_exact':True,
         'decision':'Awaiting user adoption decision and additional step0/48/full49step evidence' if candidate<min(baseline,restore) else 'Reject no demonstrated benefit',
         'candidate_default_enabled':False,
         'next':'Broader boundary and step0/48 real leaf gates, then user per-item decision before composition'}
(HERE/'trim_model_summary218.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('gate','cases')},indent=2))
print(json.dumps([{'case':c['case'],'rank_max_seconds':c['rank_max_seconds'],'rank_max_median_s':c['rank_max_median_s']} for c in cases],indent=2))
