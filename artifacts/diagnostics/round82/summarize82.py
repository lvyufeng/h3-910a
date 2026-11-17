#!/usr/bin/env python3
"""Verify exact-CPU conditioning factors;never call isolated text timing full encode."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
rows=[]
for rank in range(8):
    path=ROOT/'ranks'/('tp8_rank%d.json'%rank);r=json.loads(path.read_text())
    assert r['rank']==rank and r['world_size']==8 and r['completed']
    assert r['native_matches_round81_sha256'] and r['native_after_isolation']['raw_bits_equal']
    assert r['cases']['repeat_vs_first']['raw_bits_equal']and r['frozen_inputs_unchanged']
    assert r['CPU_condition_transfer_proof']['inputs_embeds']['raw_bits_equal']
    assert all(item['raw_bits_equal']for item in r['CPU_condition_transfer_proof']['deepstack'])
    for case in r['conditioning_isolation'].values():
        assert len(case['layers'])==50 and all(item['npu']['finite']for item in case['layers'])
        assert case['repeat_vs_first']['raw_bits_equal']and case['condition_inputs_unchanged']
        assert case['not_full_encoder_performance']
    assert r['source_sha256']==hashlib.sha256((ROOT/'condition_isolation82.py').read_bytes()).hexdigest()
    rows.append({'rank':rank,'r':r,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
base=rows[0]['r'];cases={}
for name,case in base['conditioning_isolation'].items():
    assert len({item['r']['conditioning_isolation'][name]['selected50_vs_CPU']['npu']['sha256']for item in rows})==1
    cases[name]={'relative_RMS':case['selected50_vs_CPU']['relative_rms'],
                 'max_abs_error':case['selected50_vs_CPU']['max_abs_error'],
                 'repeat_bits_equal':True,'native_initial_replaced':case['initial_condition']=='CPU_FP16',
                 'native_DeepStack_replaced':case['DeepStack_condition']=='CPU_FP16',
                 'token_subsets':{k:{'relative_RMS':v['relative_rms'],'tokens':v['npu']['shape'][0],'max_abs_error':v['max_abs_error']}for k,v in case['token_subsets'].items()},
                 'six_raw_layer_metrics':{item['layer']:item['full_compare']for item in case['layers']if 'full_compare'in item},
                 'not_full_encoder_performance':True}
maxima=[max(item['r']['timing']['samples_synchronized_wall_s_including_d2h'][i]for item in rows)for i in range(3)]
result={'scope':__doc__,'same_CPU_FP16_reference_and_native_weight_graph':True,'all8_native_matches_R81_and_restored':True,
        'native_selected50_relative_RMS':base['cases']['cpu_fp16_comparison']['selected50_pre_final_norm']['relative_rms'],
        'conditioning_factors':cases,'native_full_encode_same_call_rank_max_s':maxima,'native_full_encode_rank_max_median_s':statistics.median(maxima),
        'native_stock_encode_ids_timed':all(item['r']['stock_encode_ids_timed']for item in rows),
        'full_encoder_precision_accepted':False,'whole_goal_complete':False,
        'receipts':[{'rank':item['rank'],'sha256':item['sha256']}for item in rows]}
with(ROOT/'isolation_summary82.json').open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'native_RR':result['native_selected50_relative_RMS'],'cases':{name:{'relative_RMS':v['relative_RMS'],'subsets':v['token_subsets']}for name,v in cases.items()},'native_complete_encode_median_s':result['native_full_encode_rank_max_median_s']},indent=2))
