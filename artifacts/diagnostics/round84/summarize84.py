#!/usr/bin/env python3
"""Verify eight completed FP32-row TP8 A/B receipts without picking accuracy tolerances."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
records=[]
for rank in range(8):
    p=ROOT/'results/row_fp3202/ranks'/('tp8_rank%d.json'%rank);r=json.loads(p.read_text())
    assert r['rank']==rank and r['world_size']==8 and r['completed']
    assert r['native_matches_round81_sha256']and r['native_after_row_AB']['raw_bits_equal']
    assert r['frozen_inputs_unchanged']and r['cases']['repeat_vs_first']['raw_bits_equal']
    assert r['single_actual_CPU_input_layer0']['FP32_repeat_bits']['raw_bits_equal']
    assert r['row_full_cases']['native_rows_CPU_condition']['reproduces_R82_CPU_condition_sha256']
    for case in r['row_full_cases'].values():
        assert len(case['raw_layers'])==50 and all(item['npu']['finite']for item in case['raw_layers'])
        assert case['repeat_vs_first']['raw_bits_equal']
    assert r['stock_encode_ids_timed']and not r['cases']['stock_encode_ids_probe']['raised']
    assert r['cases']['stock_encode_ids_probe']['matches_equivalent_path']
    assert r['timing']['warm_outputs_bit_equal']
    assert not r['full_encoder_precision_accepted']and not r['whole_goal_complete']
    adapter=r['row_adapter_final'];stats=adapter['stats']
    assert adapter['weight_cache']['entries']==100 and adapter['weight_cache']['bytes']==2162688000
    for control in r['actual_activation_packing_controls'].values():
        assert control['a_pack_matches_verified_helper']and control['a_pack_roundtrip_exact']and control['a_pack_tail_rows_zero']
    assert stats['kernel_launches']==stats['fp32_collective_count']==stats['half_cast_count']
    assert stats['fp32_dtype_before_collective']=={'torch.float32':stats['fp32_collective_count']}
    assert stats['fp32_dtype_after_collective']=={'torch.float32':stats['fp32_collective_count']}
    assert stats['calls_rejected']==0
    records.append({'r':r,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
rank0=records[0]['r'];cases={}
for name,case in rank0['row_full_cases'].items():
    assert len({item['r']['row_full_cases'][name]['selected50_vs_CPU']['npu']['sha256']for item in records})==1
    cases[name]={'selected50_relative_RMS':case['selected50_vs_CPU']['relative_rms'],
                 'max_abs_error':case['selected50_vs_CPU']['max_abs_error'],
                 'token_subsets':{k:{'relative_RMS':v['relative_rms'],'tokens':v['npu']['shape'][0]}for k,v in case['token_subsets'].items()},
                 'rows':case['rows'],'conditioning':case['conditioning'],'repeat_bit_identical':True}
clocks=[max(item['r']['timing']['samples_synchronized_wall_s_including_d2h'][i]for item in records)for i in range(3)]
result={'scope':__doc__,'all8_native_baseline_and_recovery_verified':True,
        'all8_original_parameters_and_inputs_unchanged':True,
        'actual_CPU_input_layer0':rank0['single_actual_CPU_input_layer0'],
        'full50_cases':cases,'enabled_FP32_rows_complete_encode_rank_max_s':clocks,
        'enabled_FP32_rows_complete_encode_rank_max_median_s':statistics.median(clocks),
        'row_adapter_rank0':rank0['row_adapter_final'],
        'memory_before_row_cache_rank0':rank0['memory_before_row_cache_bytes'],
        'memory_after_row_cache_rank0':rank0['memory_after_row_cache_bytes'],
        'memory_after_forward_rank0':rank0['memory_after_all_forwards_bytes'],
        'full_encoder_precision_accepted':False,'whole_goal_complete':False,
        'CPU_FP16_reference_unchanged':True,
        'receipts':[{'rank':x['r']['rank'],'sha256':x['sha256']}for x in records]}
with(ROOT/'summary84.json').open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'one_layer':{k:v.get('relative_rms')if isinstance(v,dict)else v for k,v in result['actual_CPU_input_layer0'].items()},
                  'full_cases':cases,'complete_encode_rank_max_median_s':result['enabled_FP32_rows_complete_encode_rank_max_median_s']},indent=2))
