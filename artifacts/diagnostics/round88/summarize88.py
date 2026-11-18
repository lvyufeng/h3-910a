#!/usr/bin/env python3
"""Actual real-QKV vision attention and four full27-layer case verification."""
from pathlib import Path
import json
import hashlib
ROOT=Path(__file__).resolve().parent
p=ROOT/'results/vision01/vision88.json';r=json.loads(p.read_text())
assert r['completed']and r['original_parameters_unchanged']and r['original_pixels_unchanged']and r['CPU_reference_SHA_unchanged']
assert r['native_recovery']['merged']['raw_bits_equal']and all(x['raw_bits_equal']for x in r['native_recovery']['deepstack'])
assert r['source_sha256']==hashlib.sha256((ROOT/'vision_probe88.py').read_bytes()).hexdigest()
old=json.loads((ROOT.parent/'round87/results/vision01/vision87.json').read_text())
for now,previous in(('native','native'),('bias_only','FP32_bias_one_Half')):
 a=r['cases'][now];b=old['cases'][previous]
 assert a['merged_vs_CPU']['npu']['sha256']==b['merged_vs_CPU']['npu']['sha256']
 assert all(x['npu']['sha256']==y['npu']['sha256']for x,y in zip(a['DeepStack_vs_CPU'],b['DeepStack_vs_CPU']))
cases={}
for name,c in r['cases'].items():
 assert len(c['raw27_layers'])==27 and c['merged_vs_CPU']['finite']
 assert all(x['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in x['deepstack'])for x in c['repeat_bits'])
 cases[name]={'merged_relative_RMS':c['merged_vs_CPU']['relative_rms'],'merged_max_abs_error':c['merged_vs_CPU']['max_abs_error'],
              'DeepStack_relative_RMS':[a['relative_rms']for a in c['DeepStack_vs_CPU']],
              'resident_vision_warm_s':c['warm_s'],'resident_vision_median_s':c['resident_vision_median_s'],
              'repeat_bits':True,'raw27_layers_finite':True}
leaf=[]
for c in r['real_vision_SDPA_leaf']:
 assert c['repeat_bits']['raw_bits_equal']and c['original_QKV_unchanged']and c['no_causal_mask']
 assert c['original_head_dim']==72
 assert all(c['padding_controls'][name]for name in('q_prefix_exact','q_width_tail_zero','k_key_tail_zero','k_prefix_exact','v_prefix_exact','v_width_tail_zero','packed_k_head_distinct'))
 leaf.append({'image':c['image'],'native_vs_CPU_Half_MATH_relative_RMS':c['native_vs_CPU_Half_MATH']['relative_rms'],
              'custom_vs_CPU_Half_MATH_relative_RMS':c['custom_vs_CPU_Half_MATH']['relative_rms'],
              'native_Float_PV_exact_CPU_P_vs_CPU_Float_PV_relative_RMS':c['native_Float_PV_with_exact_CPU_P_vs_CPU_Float_PV']['relative_rms'],
              'actual_padding_prefix_and_tail_checks':True,'QKV_SHA_unchanged':True,'repeat_bits':True})
stats=r['engine_final']['stats']
assert stats['calls_rejected']==0 and stats['kernel_launches']==24*stats['calls_enabled']
assert stats['float_before_half_chunks']==8*stats['calls_enabled']
summary={'scope':__doc__,'native_and_bias_only_reproduced_R87_output_SHA':True,'cases':cases,'actual_leafs':leaf,
         'engine_stats':stats,'bias_stats':r['bias_final'],
         'all_original_params_pixels_CPUgold_unchanged':True,'native_output_recovered':True,
         'not_full_encoder_timing':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
         'frozen_vision_QKV_remote':'/tmp/h3_round88_vision01/FrozenVisionQKV88.pt','frozen_capture_SHA256':r['frozen_capture_SHA256'],
         'actual_receipt_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
with(ROOT/'summary88.json').open('x')as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'cases':cases,'actual_leafs':leaf,'engine_stats':stats},indent=2))
