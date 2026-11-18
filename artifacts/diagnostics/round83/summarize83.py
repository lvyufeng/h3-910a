#!/usr/bin/env python3
"""Summarize verified R83 primitive results,not whole-encoder precision/speed."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
bias_path=ROOT/'results/bias01/bias_result83.json';bias=json.loads(bias_path.read_text())
assert bias['completed']and bias['kernel_launch_executed']and bias['repeat_vs_first']['raw_bits_equal']
assert bias['original_inputs_and_parameters_unchanged']and bias['GPU_operands_unchanged']
results={};receipts=[]
for folder in(ROOT/'results/blocked02',ROOT/'results/blocked_dynamic01'):
    if not folder.exists():continue
    for p in folder.glob('blocked_*_result83.json'):
        r=json.loads(p.read_text())
        if not r.get('completed')or not r.get('kernel_launch_executed'):continue
        assert r['kernel_repeat_vs_first']['raw_bits_equal']and r['original_inputs_and_weights_unchanged']
        assert r['padded_output_rows_max_abs']==0
        assert r['build']['dst_dtype']=='float32'
        if r['case']=='smoke':continue
        entry={'logical_geometry':r['logical_geometry'],'core_count':r['build']['block_dim'],
               'FP32_accumulator_diagnostic_relative_RMS':r['kernel_FP32_vs_accumulator_diagnostic']['relative_RMS'],
               'custom_final_Half_vs_CPU_Half_relative_RMS':r['kernel_final_Half_vs_CPU_Half']['relative_RMS'],
               'native_Half_vs_CPU_Half_relative_RMS':r['native_Half_vs_CPU_Half']['relative_RMS'],
               'custom_repeat_bit_equal':True,'native_repeat_bit_equal':r['native_repeat_vs_first']['raw_bits_equal'],
               'medians_s':r['medians_s'],'cached_Half_weight_bytes':r['cached_Half_weight_bytes'],
               'source':str(p.relative_to(ROOT)),'same_output_dtype_for_performance_comparison':False}
        results[r['case']]=entry;receipts.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
fullbias=[]
for folder in(ROOT/'results/blocked_bias01',):
    if folder.exists():
        for p in folder.glob('blocked_bias_*_result83.json'):
            r=json.loads(p.read_text())
            if not r.get('completed'):continue
            assert r['repeat_vs_first']['raw_bits_equal']and r['GPU_operands_unchanged']
            fullbias.append({'case':r['case'],'geometry':r['geometry'],'custom_vs_CPU_Half':r['kernel_vs_CPU_FP16']['relative_RMS'],
                             'native_vs_CPU_Half':r['native_vs_CPU_FP16']['relative_RMS'],'packed_only_median_s':r['packed_micro_median_s'],
                             'not_full_encoder_performance':True,'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
summary={'scope':__doc__,'portable_bias_micro':{'custom_vs_CPU_Half_relative_RMS':bias['kernel_vs_CPU_FP16']['relative_RMS'],
          'native_vs_CPU_Half_relative_RMS':bias['native_vs_CPU_FP16']['relative_RMS'],
          'error_reduction_ratio':bias['native_vs_CPU_FP16']['relative_RMS']/bias['kernel_vs_CPU_FP16']['relative_RMS'],
          'repeat_bit_identical':True,'original_bias_parameter_Half_values_unchanged':True},
         'H3_projection_cases':results,'blocked_bias_cases':fullbias,'receipts':receipts,
         'all_four_H3_projection_shapes_executed':set(results)=={'kv','q','up','row'},
         'CPU_FP16_reference_unchanged':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
         'no_whole_encoder_speedup_from_primitive_timing':True}
with(ROOT/'summary83.json').open('x')as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
