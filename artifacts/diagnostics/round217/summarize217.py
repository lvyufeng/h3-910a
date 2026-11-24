#!/usr/bin/env python3
"""Summarize R217 receipts without treating empirical GEMM rate as a physical ceiling."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def load(name):
    return json.loads((HERE/name).read_text())

base=load('restored_final217.json')
restored=load('restored_end217.json')
candidates=[('grouped_k128',load('real_pvgroup_raw217.json')),
            ('grouped_k128_thread2',load('real_pvdouble_raw217.json'))]
instance=base['compiled_kernels'][0]['instance']
assert instance['has_attn_mask'] is False and instance['only_causal'] is False
assert instance['prev_block_num']==instance['next_block_num']==65536
n=instance['Nq'];nk=instance['Nk'];h=instance['B_heads'];d=instance['d']
padded=4*h*n*nk*d/1e12
useful=4*h*base['input_shape'][2]*base['valid_keys']*d/1e12
work={'mode':'dense attention','padded_dense_matmul_TFLOP_per_call':padded,
      'useful_dense_matmul_TFLOP_per_call':useful,
      'dense_tiles_per_head':instance['Tr']*instance['Tc'],
      'dense_tiles_all_heads':h*instance['Tr']*instance['Tc'],
      'padded_rate_TFLOPS_per_card':padded/(base['kernel_event_median_ms']/1000),
      'historical_best_GEMM_TFLOPS_per_card':96.9,
      'empirical_GEMM_reference_is_not_a_physical_ceiling':True,
      'not_measured_hardware_occupancy_or_MFU':True}
rows=[{'name':'accepted_baseline','wall_ms':base['warm_wrapper_wall_median_ms'],
       'kernel_ms':base['kernel_event_median_ms'],'disposition':'Keep baseline'}]
for name,rec in candidates:
    rows.append({'name':name,'wall_ms':rec['warm_wrapper_wall_median_ms'],
                 'kernel_ms':rec['kernel_event_median_ms'],
                 'kernel_time_ratio_to_baseline':rec['kernel_event_median_ms']/base['kernel_event_median_ms'],
                 'output_sha256_matches_baseline':rec['output_sha256']==base['output_sha256'],
                 'disposition':'Reject: slower; not adopted'})
rows.append({'name':'restored_baseline','wall_ms':restored['warm_wrapper_wall_median_ms'],
             'kernel_ms':restored['kernel_event_median_ms'],'disposition':'Keep baseline restored'})
# Pure scenarios, different-era region scopes must not be mistaken for a measured decomposition.
A=15.473;guard=1.665;F=7.454
allowance=10-(A-guard-F)
scenarios={'formula':'new_step=A-guard_gain-F+F/speedup-other_gain',
           'A_s':A,'guard_gain_assumed_s':guard,'historical_FA_scope_s':F,
           'no_combo_measured':True,'different_run_scopes':True,
           'FA_speedup_needed_with_guard_assumption':F/allowance,
           'FA_speedup_needed_without_guard':F/(10-(A-F)),
           'FA_speedup_1_5_assuming_guard_step_s':A-guard-F+F/1.5,
           'FA_speedup_1_89_assuming_guard_step_s':A-guard-F+F/1.89,
           'not_a_physical_lower_bound':True,
           'historical_fit_not_proven_removable_cost':True}
result={'schema':1,'round':217,'nothing_adopted':True,'whole_goal_complete':False,
        'scope':'single-card synthetic local FA with actual geometry and key limit; NOT 8-rank whole-step',
        'geometry':instance,'work':work,'measurements':rows,
        'numeric':load('exact_comparison217.json'),
        'failed_direct_PV_layout':load('comparison_pv217.json'),
        'scoped_sync_runtime_fault':'507015 L0A/L0B read/write conflict, not a timing result',
        'grouped_double_and_single_binary_equal':candidates[0][1]['compiled_kernels'][0]['object_sha256']==candidates[1][1]['compiled_kernels'][0]['object_sha256'],
        'conditional_scenarios_only':scenarios,
        'source_policy':'Only hashes of installed proprietary SDK references published; no SDK vendoring',
        'next_experiment':'Optimize vector memory passes with exact NZ layouts and tail handling; keep automatic dependency sync, and measure current same-run critical path before whole-step projection'}
(HERE/'summary217.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'measurements':rows,'work':work,'conditional_scenarios':scenarios},indent=2))
