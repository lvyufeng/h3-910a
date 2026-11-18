#!/usr/bin/env python3
"""Separate exact CPU input-only amplification from same-input NPU arithmetic."""
from pathlib import Path
import json
import hashlib
ROOT=Path(__file__).resolve().parent
cpu=json.loads((ROOT/'results/cpu02/CPU_block25_reference92.json').read_text())
assert cpu['completed']and cpu['reference_gate']['matches_R78_raw25_SHA256']and cpu['reference_gate']['repeat_bits_equal']
rows=[];receipts=[]
for rank in range(8):
    p=ROOT/'results/block25_02/ranks'/('block25_rank%d.json'%rank);r=json.loads(p.read_text())
    assert r['rank']==rank and r['completed']and r['original_parameters_unchanged']and r['CPU_reference_SHA_unchanged']
    for case,c in r['cases'].items():
        assert c['native_recovery_and_input_SHA_unchanged']
        for mode in('native','precise_attention_native_rows','native_attention_FP32_rows','precise_attention_FP32_rows'):
            m=c[mode];assert m['repeat_bit_identical']and m['same_input_vs_original_CPU_Half_block25']['finite']
    rows.append(r);receipts.append({'rank':rank,'receiptSHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
summary={}
for case,c in rows[0]['cases'].items():
    modes={}
    for mode in('native','precise_attention_native_rows','native_attention_FP32_rows','precise_attention_FP32_rows'):
        same=c[mode]['same_input_vs_original_CPU_Half_block25'];original=c[mode]['vs_original_CPU_layer25']
        hashes={r['cases'][case][mode]['same_input_vs_original_CPU_Half_block25']['npu']['sha256']for r in rows}
        assert len(hashes)==1
        modes[mode]={'same_input_arithmetic_relative_RMS':same['relative_rms'],
                     'same_input_arithmetic_max_abs_error':same['max_abs_error'],
                     'full_output_vs_original_CPU25_relative_RMS':original['relative_rms'],
                     'all8_output_SHA256':next(iter(hashes))}
    summary[case]={'input_only_original_CPU_control':cpu['input_error_amplification'].get(case),
                   'input_SHA256':c['native']['original_input_SHA256'],'math_modes':modes}
out={'scope':__doc__,'CPU_original_raw25_twice_SHA_exact':True,'CPU_original_Half_weights_no_Float_teacher':True,
     'no_DeepStack_after24_or_finalnorm':True,'not_full_encoder_timing':True,
     'all8_same_input4cases4modes_repeat_native_recovery_and_params_inputs_unchanged':True,
     'cases':summary,'receipts':receipts,'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,
     'original_R91_fullvision_combination_not_default':True}
with(ROOT/'summary92.json').open('x')as f:f.write(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
