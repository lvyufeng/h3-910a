#!/usr/bin/env python3
"""Summarize measured FP16 operator factors without changing the precision reference."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipts={name:json.loads((ROOT/file).read_text())for name,file in(('operators','operators.json'),('factors','factors.json'),('policy','policy.json'),('split','pv_split.json'),('vision','vision_comp.json'))}
for name,r in receipts.items():assert r['completed']and not r['production_changed']and not r['whole_goal_complete']
f=receipts['factors']['cases'];p=receipts['policy']['cases']['Attention_PV'];s=receipts['split']['cases'];v=receipts['vision']['phases']
result={'scope':__doc__,'CPU_reference_completed_and_verified':True,'reference_receipt_sha256':sha(ROOT.parent/'round78'/'reference.json'),
        'CPU_reference_unmodified_by_factor_experiments':True,
        'native_FP16_biased_linear_equals_FP16_no_bias_GEMM_plus_FP16_bias_add':f['Linear_native_with_bias_vs_NPU_two_rounding_factor']['raw_bits_equal'],
        'same_input_linear_relative_RMS':{'native_no_bias':f['Linear_no_bias_native_vs_CPU']['relative_RMS'],'native_bias':f['Linear_with_bias_native_vs_CPU']['relative_RMS'],'FP32_bias_proxy':f['Linear_FP32_bias_island_vs_CPU']['relative_RMS']},
        'LayerNorm_classified':{name:f[name]for name in('LayerNorm_native_constant','LayerNorm_stable_constant','LayerNorm_native_ordinary','LayerNorm_stable_ordinary','LayerNorm_stable_repeat')},
        'HF32_flag_control':{'before':receipts['policy']['policy_before'],'restored':receipts['policy']['policy_restored'],'PV_errors_identical_under_false_and_true':p['false']['NPU_FP32_vs_CPU_FP32']==p['true']['NPU_FP32_vs_CPU_FP32']},
        'grouped_matmul_FP16_inputs_FP32_output_supported':receipts['factors']['grouped_matmul_FP16_inputs_FP32_output']['supported'],
        'same_input_PV_relative_RMS':{'native_arbitrary_FP32_probabilities':s['native_FP32']['relative_RMS'],'split65536':s['split_scale_65536']['same_input_CPU_FP32']['relative_RMS']},
        'split65536_probability_reconstruction_not_exact':s['split_scale_65536']['FP32_probability_reconstruction']['relative_RMS']>0,
        'whole_vision_phases':{name:{'CPU_FP16_merged_relative_RMS':r['CPU_FP16_comparison']['merged']['relative_RMS'],'CPU_FP16_DeepStack_relative_RMS':[x['relative_RMS']for x in r['CPU_FP16_comparison']['deepstack']],'resident_vision_median_s':statistics.median(x['resident_vision_s']for x in r['calls']),'FP32_shadow_cache_bytes':r['FP32_cache_bytes']}for name,r in v.items()},
        'native_restored_bit_equal':receipts['vision']['native_restoration']['merged']['raw_bits_equal'],
        'full_encoder_precision_accepted':False,'whole_goal_complete':False,
        'remaining':['Theseleafcontrolsarenotfullencoderacceptance','FP32biasproxyandtwoPVgemmsnotproductionoptimizedkernels','CompensatedPVimprovesfullvisionRR1.46to1.18percentbutslowsvision1.45to2.25s','CompleteTP8baselinecurrentlyrunning','KeepCPUFP16asuserreference;noBF16requirementorafterthefactthreshold']}
with(ROOT/'summary80.json').open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
