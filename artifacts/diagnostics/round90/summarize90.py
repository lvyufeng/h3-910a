#!/usr/bin/env python3
"""Verify real erfGELU controls and fullvisioncandidate; no fullprecision waiver."""
from pathlib import Path
import json
import hashlib
ROOT=Path(__file__).resolve().parent
pre=json.loads((ROOT/'results/preflight01/gelu_preflight90.json').read_text())
kpath=ROOT/'results/vision02/kernel_probe90.json';vpath=ROOT/'results/vision02/vision90.json'
k=json.loads(kpath.read_text());v=json.loads(vpath.read_text())
assert pre['completed']and k['completed']and v['completed']
assert k['improves_both_real_CPU_erf_leafs']
assert v['original_parameters_unchanged']and v['pixels_unchanged']and v['CPU_reference_SHA_unchanged']
assert v['native_recovery']['merged']['raw_bits_equal']and all(x['raw_bits_equal']for x in v['native_recovery']['deepstack'])
assert v['source_SHA256']==hashlib.sha256((ROOT/'vision_probe_fixed90.py').read_bytes()).hexdigest()
old=json.loads((ROOT.parent/'round89/results/vision02/vision89.json').read_text())
for current,previous in(('native','native'),('patch_and_linear_bias','patch_and_linear_bias')):
 a=v['cases'][current];b=old['cases'][previous]
 assert a['merged_vs_CPU']['npu']['sha256']==b['merged_vs_CPU']['npu']['sha256']
 assert all(x['npu']['sha256']==y['npu']['sha256']for x,y in zip(a['DeepStack_vs_CPU'],b['DeepStack_vs_CPU']))
leaves=[]
for name in('deepstack_merger_list.0.act_fn','merger.act_fn'):
 c=k['cases'][name];assert c['repeat_bits']['raw_bits_equal']and c['input_SHA_unchanged']
 p=pre['leafs'][name];assert p['native_none_vs_native_tanh']['raw_bits_equal']
 leaves.append({'name':name,'native_None_CPU_Half_erf_relative_RMS':p['native_none_vs_CPU_Half_erf']['relative_rms'],
                'explicit_Torch_erf_CPU_Half_relative_RMS':p['NPU_Float_explicit_erf_then_Half_vs_CPU_Half_erf']['relative_rms'],
                'TIK_erf_CPU_Half_relative_RMS':c['CPU_Half_erf']['relative_rms'],
                'TIK_max_abs_error':c['CPU_Half_erf']['max_abs_error'],'repeat_bits':True,'inputs_SHA_unchanged':True})
cases={}
for name,c in v['cases'].items():
 assert len(c['raw27_layers'])==27 and c['merged_vs_CPU']['finite']
 assert all(x['merged']['raw_bits_equal']and all(y['raw_bits_equal']for y in x['deepstack'])for x in c['repeat_bits'])
 cases[name]={'merged_relative_RMS':c['merged_vs_CPU']['relative_rms'],'max_abs_error':c['merged_vs_CPU']['max_abs_error'],
              'DeepStack_relative_RMS':[x['relative_rms']for x in c['DeepStack_vs_CPU']],
              'resident_vision_median_s':c['resident_vision_median_s'],'repeat_bit_equal':True}
assert v['TIK_erf_final']['patched_module_count']==4 and v['TIK_erf_final']['tanh_gelu_modules_left_untouched']==27
assert v['TIK_erf_final']['kernel_launches']==v['TIK_erf_final']['calls']==32
s={'scope':__doc__,'native_and_patchbias_reproduce_R89_output_SHA':True,'same_actual_Half_erf_leafs':leaves,
   'full_vision_cases':cases,'special_signed_extremes':k['cases']['signed_extreme_smoke']['special_values'],
   'TIK_adapter':v['TIK_erf_final'],'explicit_Torch_adapter':v['explicit_erf_final'],
   'original_Half_params_inputs_CPUgold_unchanged':True,'all_raw27_cases_finite_and_repeat_bit_equal':True,
   'native_recovery_verified':True,'only4merger_erf_modified_27MLPtanh_untouched':True,
   'no_Float_weight_teacher':True,'numerical_tolerance':None,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
   'not_full_encoder_timing':True,'kernel_receipt_SHA256':hashlib.sha256(kpath.read_bytes()).hexdigest(),
   'vision_receipt_SHA256':hashlib.sha256(vpath.read_bytes()).hexdigest()}
with(ROOT/'summary90.json').open('x')as f:f.write(json.dumps(s,indent=2)+'\n')
print(json.dumps({'leafs':leaves,'full_vision_cases':cases,'actual_kernel_launches':v['TIK_erf_final']['kernel_launches']},indent=2))
