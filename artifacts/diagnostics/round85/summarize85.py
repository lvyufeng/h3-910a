#!/usr/bin/env python3
"""Verify precise-attention full TP8 results without inventing an accuracy tolerance."""
from pathlib import Path
import hashlib
import json
import statistics
ROOT=Path(__file__).resolve().parent
records=[]
for rank in range(8):
    path=ROOT/'results/full01/ranks'/('tp8_rank%d.json'%rank)
    r=json.loads(path.read_text())
    assert r['rank']==rank and r['completed']and r['frozen_inputs_unchanged']
    assert r['native_matches_round81_sha256']and r['native_after_row_AB']['raw_bits_equal']
    for case in r['row_full_cases'].values():
        assert case['repeat_vs_first']['raw_bits_equal']and len(case['raw_layers'])==50
        assert all(layer['npu']['finite']for layer in case['raw_layers'])
    assert r['row_full_cases']['native_rows_CPU_condition']['reproduces_R82_CPU_condition_sha256']
    engine=r['accurate_attention_final'];assert engine['kernel_calls']==12*engine['calls']
    assert engine['actual_FP32_before_Half_chunks']==engine['half_output_chunks']==4*engine['calls']
    assert r['stock_encode_ids_timed']and r['timing']['warm_outputs_bit_equal']
    assert r['cases']['stock_encode_ids_probe']['matches_equivalent_path']
    records.append((r,hashlib.sha256(path.read_bytes()).hexdigest()))
first=records[0][0];cases={}
for name,case in first['row_full_cases'].items():
    shas={r['row_full_cases'][name]['selected50_vs_CPU']['npu']['sha256']for r,_ in records};assert len(shas)==1
    cases[name]={'relative_RMS':case['selected50_vs_CPU']['relative_rms'],
                 'max_abs_error':case['selected50_vs_CPU']['max_abs_error'],
                 'token_subsets':{k:v['relative_rms']for k,v in case['token_subsets'].items()},
                 'repeat_bit_identical':True,'conditioning':case['conditioning'],'rows':case['rows']}
clocks=[max(r['timing']['samples_synchronized_wall_s_including_d2h'][i]for r,_ in records)for i in range(3)]
leaf=json.loads((ROOT/'results/batch01/attention_result85.json').read_text())
summary={'scope':__doc__,'all8_completed_repeated_and_native_recovered':True,'all8_parameters_and_inputs_unchanged':True,
         'full50_cases':cases,'actual_single_CPU_input_decoder0':first['single_actual_CPU_input_layer0'],
         'precise_attention_and_FP32_rows_full_encode_rank_max_s':clocks,
         'full_encode_rank_max_median_s':statistics.median(clocks),
         'engine_rank0':first['accurate_attention_final'],'row_adapter_rank0':first['row_adapter_final'],
         'leaf_CPU_Half_relative_RMS_custom':leaf['custom_vs_CPU_Half_MATH']['relative_rms'],
         'leaf_CPU_Half_relative_RMS_native':leaf['saved_native_vs_CPU_Half_MATH']['relative_rms'],
         'full_encoder_precision_accepted':False,'whole_goal_complete':False,'CPU_reference_unchanged':True,
         'receipts':[{'rank':r['rank'],'sha256':sha}for r,sha in records]}
with(ROOT/'summary85.json').open('x')as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'full50_cases':cases,'full_encode_median_s':statistics.median(clocks)},indent=2))
