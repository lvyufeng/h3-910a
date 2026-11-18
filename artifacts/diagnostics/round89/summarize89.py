#!/usr/bin/env python3
"""Verify original CPU-entry reset and real seven-operator Half controls."""
from pathlib import Path
import json
import hashlib
ROOT=Path(__file__).resolve().parent
p=ROOT/'results/vision02/vision89.json';r=json.loads(p.read_text())
assert r['completed']and r['original_parameters_unchanged']and r['pixels_unchanged']and r['CPU_reference_SHA_unchanged']
assert r['CPU_entry_NPU_transfer_and_all_calls_unchanged']and r['rebase_calls']==12
assert r['original_HF_CPU_entry_reference']['CPU_block0_matches_original_R78_output_SHA']
assert r['original_HF_CPU_entry_reference']['CPU_entry_repeat_bits']
assert r['native_recovery']['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in r['native_recovery']['deepstack'])
assert r['source_sha256']==hashlib.sha256((ROOT/'vision_probe_fixed89.py').read_bytes()).hexdigest()
initial=json.loads((ROOT/'results/vision01/vision89.json').read_text())
assert initial['original_HF_CPU_entry_reference']['CPU_block0']['sha256']==r['original_HF_CPU_entry_reference']['CPU_block0']['sha256']
assert not initial['completed']and 'Inference tensors cannot be saved for backward'in initial['failure']['message']
assert r['actual_patch_onecast_GEMM']['custom_vs_original_CPU_Half_Conv3d']['finite']
assert r['patch_final']['kernel_launches']==14
old=json.loads((ROOT.parent/'round87/results/vision01/vision87.json').read_text())
assert r['cases']['native']['merged_vs_CPU']['npu']['sha256']==old['cases']['native']['merged_vs_CPU']['npu']['sha256']
leaf=r['actual_nonlinear_leaf']
assert all(leaf[key]for key in('all_native_repeats_bit_identical','all_cpu_half_repeats_bit_identical','all_npu_float_diagnostics_supported','all_inputs_unchanged','all_leaf_parameters_unchanged'))
leafs={}
for name,c in leaf['controls'].items():
 leafs[name]={'kind':c['kind'],'approximate':c['approximate'],
              'native_CPU_Half_relative_RMS':c['native_vs_cpu_half']['relative_rms'],
              'NPU_Float_then_Half_CPU_Half_relative_RMS':c['npu_float_op_then_half_vs_cpu_half']['relative_rms'],
              'CPU_Float_then_Half_CPU_Half_relative_RMS':c['cpu_float_op_then_half_vs_cpu_half']['relative_rms'],
              'CPU_Float_then_Half_bits_equal_CPU_Half':c['cpu_float_op_then_half_vs_cpu_half']['raw_bits_equal'],
              'native_max_abs_error':c['native_vs_cpu_half']['max_abs_error'],
              'Float_variant_max_abs_error':c['npu_float_op_then_half_vs_cpu_half']['max_abs_error'],
              'same_actual_Half_inputs_and_params_unchanged':c['input_sha_unchanged']and c['parameters_sha_unchanged']}
entry={name:{key:c[key]for key in('relative_rms','max_abs_error','raw_bits_equal')}for name,c in r['actual_entry'].items()}
cases={}
for name,c in r['cases'].items():
 assert len(c['raw27_layers'])==27 and c['merged_vs_CPU']['finite']
 assert all(a['merged']['raw_bits_equal']and all(b['raw_bits_equal']for b in a['deepstack'])for a in c['repeat_bits'])
 cases[name]={'CPU_entry_reset':c['CPU_entry_reset'],'merged_relative_RMS':c['merged_vs_CPU']['relative_rms'],
              'merged_max_abs_error':c['merged_vs_CPU']['max_abs_error'],
              'DeepStack_relative_RMS':[a['relative_rms']for a in c['DeepStack_vs_CPU']],
              'resident_vision_median_s':c['median_resident_vision_s'],
              'not_full_encoder_or_speed_candidate_when_CPU_entry_reset':c['CPU_entry_reset'],
              'repeat_bits':True}
summary={'scope':__doc__,'CPU_entry_reference':r['original_HF_CPU_entry_reference'],'actual_entry':entry,
         'actual_nonlinear_leafs':leafs,'full_vision_cases':cases,'all_raw27_cases_finite_and_repeated':True,
         'native_and_original_parameter_input_CPUgold_recovery_verified':True,
         'CPU_entry_NPU_transfer_bit_exact_and_SHA_unchanged':True,'rebase_calls':r['rebase_calls'],
         'R88_slow_attention_disabled':True,'nonlinear_final':r['nonlinear_final'],'bias_final':r['bias_final'],
         'actual_patch_onecast_GEMM':r['actual_patch_onecast_GEMM'],'patch_final':r['patch_final'],
         'initial_failed_inference_context_call_preserved_and_CPU_block0_identical_to_fixed_run':True,
         'numerical_tolerance':None,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
         'frozen_payload_remote':'/tmp/h3_round89_vision02/CPUEntryAndActualLeaves89.pt','frozen_inputs_SHA256':r['frozen_inputs_SHA256'],
         'actual_receipt_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
with(ROOT/'summary89.json').open('x')as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:summary[key]for key in('actual_entry','actual_nonlinear_leafs','full_vision_cases')},indent=2))
