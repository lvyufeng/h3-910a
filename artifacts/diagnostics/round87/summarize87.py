#!/usr/bin/env python3
"""Verify actual vision bias A/B and native-row accurate-attention full-encode data."""
from pathlib import Path
import hashlib
import json
import statistics
ROOT=Path(__file__).resolve().parent
ranks=[]
for rank in range(8):
    p=ROOT/'results/best_text01/ranks'/('tp8_rank%d.json'%rank);r=json.loads(p.read_text())
    assert r['rank']==rank and r['completed']and r['frozen_inputs_unchanged']
    assert r['row_adapter_not_installed']and 'row_adapter_final'not in r
    assert r['native_matches_round81_sha256']and r['native_after_row_AB']['raw_bits_equal']
    assert r['stock_encode_ids_timed']and r['timing']['warm_outputs_bit_equal']
    assert r['cases']['stock_encode_ids_probe']['matches_equivalent_path']
    for case in r['row_full_cases'].values():
        assert len(case['raw_layers'])==50 and case['repeat_vs_first']['raw_bits_equal']
    ranks.append((r,hashlib.sha256(p.read_bytes()).hexdigest()))
case_names=('native_rows_CPU_condition','accurate_attention_native_rows_CPU_condition','accurate_attention_native_rows_native_condition')
text_cases={}
for name in case_names:
    assert len({r['row_full_cases'][name]['selected50_vs_CPU']['npu']['sha256']for r,_ in ranks})==1
    c=ranks[0][0]['row_full_cases'][name]
    text_cases[name]={'relative_RMS':c['selected50_vs_CPU']['relative_rms'],'conditioning':c['conditioning'],
                      'max_abs_error':c['selected50_vs_CPU']['max_abs_error'],'repeat_bit_identical':True}
clocks=[max(r['timing']['samples_synchronized_wall_s_including_d2h'][i]for r,_ in ranks)for i in range(3)]
vpath=ROOT/'results/vision01/vision87.json';v=json.loads(vpath.read_text())
assert v['completed']and v['original_parameters_unchanged']and v['original_pixels_unchanged']and v['CPU_reference_SHA_unchanged']
assert v['native_recovery']['merged']['raw_bits_equal']and all(x['raw_bits_equal']for x in v['native_recovery']['deepstack'])
for c in v['cases'].values():
    assert len(c['raw27_layers'])==27
    assert all(x['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in x['deepstack'])for x in c['repeat_bits'])
vision_cases={name:{'merged_relative_RMS':c['merged_vs_CPU']['relative_rms'],
                    'DeepStack_relative_RMS':[x['relative_rms']for x in c['DeepStack_vs_CPU']],
                    'resident_vision_median_s':c['median_resident_vision_s']}for name,c in v['cases'].items()}
leaf={name:{'native_CPU_Half_relative_RMS':c['native_vs_CPU_Half']['relative_rms'],
            'custom_CPU_Half_relative_RMS':c['custom_vs_CPU_Half']['relative_rms'],
            'padding_controls':c['padding_controls']}for name,c in v['actual_biased_leaf'].items()}
result={'scope':__doc__,'all8_best_text_repeat_and_native_recovery_verified':True,
        'best_text_uses_native_row_without_extra_row_weight_cache':True,'best_text_cases':text_cases,
        'actual_best_text_full_encode_rank_max_s':clocks,'actual_best_text_full_encode_rank_max_median_s':statistics.median(clocks),
        'vision_cases':vision_cases,'actual_vision_biased_leafs':leaf,'vision_adapter':v['adapter_final'],
        'original_params_CPU_oracle_and_inputs_unchanged':True,
        'full_encoder_precision_accepted':False,'whole_goal_complete':False,
        'not_combined_vision_adapter_and_text_full_pipeline_result':True,
        'best_text_receipts':[{'rank':r['rank'],'sha256':sha}for r,sha in ranks],
        'vision_receipt_sha256':hashlib.sha256(vpath.read_bytes()).hexdigest()}
with(ROOT/'summary87.json').open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k]for k in('best_text_cases','actual_best_text_full_encode_rank_max_median_s','vision_cases','actual_vision_biased_leafs')},indent=2))
