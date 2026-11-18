#!/usr/bin/env python3
"""Summarize CPU-layer24 rebased late stages and genuine CPUHalf leaf controls."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parent
rows=[]
for rank in range(8):
    p=ROOT/'results/late01/ranks'/('tp8_rank%d.json'%rank);r=json.loads(p.read_text())
    assert r['completed']and r['rank']==rank and r['CPU_layer24_transfer']['raw_bits_equal']
    assert r['original_parameters_unchanged']and r['original_CPU_layer24_input_unchanged']and r['CPU_reference_receipt_unchanged']
    assert r['native_recovered']['raw_bits_equal']and r['block25_leaf_controls']['frozen_unchanged']
    for case in r['cases'].values():
        assert len(case['raw25to49'])==25 and case['repeat_bits']['raw_bits_equal']
        assert case['no_DeepStack_or_final_norm_added']
    rows.append((r,hashlib.sha256(p.read_bytes()).hexdigest()))
first=rows[0][0];cases={}
for name,case in first['cases'].items():
    assert len({r['cases'][name]['final49_vs_CPU']['npu']['sha256']for r,_ in rows})==1
    cases[name]={'relative_RMS':case['final49_vs_CPU']['relative_rms'],'max_abs_error':case['final49_vs_CPU']['max_abs_error'],
                 'final49_info':case['final49_vs_CPU']['npu'],'repeat_bit_identical':True}
leaf=first['block25_leaf_controls']['controls']
report={'scope':__doc__,'all8_original_CPU_layer24_transfer_and_native_recovery_verified':True,
        'all8_original_inputs_and_parameters_unchanged':True,'rebased_late25to49_cases':cases,
        'leaf_rank0':leaf,'leaf_all8':[{'rank':r['rank'],'controls':r['block25_leaf_controls']['controls']}for r,_ in rows],
        'not_full_encoder_timing':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
        'receipts':[{'rank':r['rank'],'sha256':sha}for r,sha in rows]}
with(ROOT/'summary86.json').open('x')as f:f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps({'rebased_late_cases':cases,'block25_leaf':{k:v for k,v in leaf.items()if k in('input_layernorm','silu_on_actual_native_gate')}},indent=2))
