#!/usr/bin/env python3
"""Verify paired actual eight-rank full encoder cases,cache identity and clocks."""
from pathlib import Path
import json
import hashlib
import statistics
ROOT=Path(__file__).resolve().parent
rows=[];receipts=[]
for rank in range(8):
    p=ROOT/'results/full01/ranks'/('tp8_rank%d.json'%rank);r=json.loads(p.read_text())
    assert r['rank']==rank and r['world_size']==8 and r['completed']
    assert r['native_matches_R82_output_SHA']and r['native_recovery']['raw_bits_equal']
    assert r['native_geometry_cache_all77_boundaries_bit_equal']and r['corrected_geometry_cache_all77_boundaries_bit_equal']
    assert r['parameter_and_fixture_SHA_unchanged']and r['CPU_reference_SHA_unchanged']
    assert r['FP32_row_adapter_not_installed']and r['real_full_encoder_warm_not_CPU_condition_time']
    assert all(x['sha256_unchanged_since_prime']for x in r['geometry_final']['cached'].values())
    rows.append(r);receipts.append({'rank':rank,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
cases={}
for name in rows[0]['cases']:
    hashes={r['cases'][name]['selected50_vs_CPU']['npu']['sha256']for r in rows};assert len(hashes)==1
    vis_hashes={r['cases'][name]['vision_merged_vs_CPU']['npu']['sha256']for r in rows};assert len(vis_hashes)==1
    for r in rows:
        c=r['cases'][name]
        assert len(c['raw50_layers'])==50 and len(c['raw27_vision_layers'])==27
        assert c['repeat_vs_first']['raw_bits_equal']and c['stock_vs_equivalent']['raw_bits_equal']
        assert len(c['warm_stock_s'])==5 and len(set(c['warm_output_SHA256']))==1
        assert c['position_ids_vs_CPU']['raw_bits_equal']and c['visual_mask_vs_CPU']['raw_bits_equal']
    clocks=[max(r['cases'][name]['warm_stock_s'][i]for r in rows)for i in range(5)]
    c=rows[0]['cases'][name]
    cases[name]={'vision_fixes':c['vision_fixes'],'precise_text_attention':c['precise_text_attention'],
                 'geometry_cache':c['geometry_cache'],'selected50_relative_RMS':c['selected50_vs_CPU']['relative_rms'],
                 'selected50_max_abs_error':c['selected50_vs_CPU']['max_abs_error'],
                 'vision_merged_relative_RMS':c['vision_merged_vs_CPU']['relative_rms'],
                 'DeepStack_relative_RMS':[a['relative_rms']for a in c['vision_DeepStack_vs_CPU']],
                 'text_inputs_relative_RMS':c['text_inputs_vs_CPU']['relative_rms'],
                 'paired8rank_max_real_encode_s':clocks,'paired8rank_max_median_s':statistics.median(clocks),
                 'all5_paired8rank_max_under2s':all(x<2 for x in clocks),
                 'whole_encoder_output_SHA256':next(iter(hashes)),'all8_repeat_native_recovery_and_warm_proofs':True}
controls={}
for name in rows[0]['CPU_condition_controls']:
    c=rows[0]['CPU_condition_controls'][name]
    assert all(r['CPU_condition_controls'][name]['repeat']['raw_bits_equal']for r in rows)
    controls[name]={'selected50_relative_RMS':c['selected50_vs_CPU']['relative_rms'],
                    'output_SHA256':c['selected50_vs_CPU']['npu']['sha256'],'not_full_encoder_timing':True}
s={'scope':__doc__,'actual_full_cases':cases,'CPU_condition_only_accuracy_controls':controls,
   'all8_native_geometry_cache_77boundaries_bit_equal':True,'all8_corrected_geometry_cache_77boundaries_bit_equal':True,
   'all_original_params_fixture_CPUgold_and_cached_tensors_unchanged':True,
   'geometry_extra_bytes_per_rank':rows[0]['geometry_final']['extra_bytes'],
   'geometry_stats_rank0':rows[0]['geometry_final']['stats'],
   'no_CPU_precomputed_image_condition_in_actual_warm':True,'no_FP32_row_adapter_or_weight_shadow':True,
   'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,
   'timing_scope':'realstockprivateencode_ids realpixels/27vision/50text/layout/HCCL/D2H;rankbarrieroutsideclock;paired8rankmax5samples',
   'maximum_rank_peak_allocated_bytes':max(r['memory']['max_allocated']for r in rows),'receipts':receipts}
with(ROOT/'summary91.json').open('x')as f:f.write(json.dumps(s,indent=2)+'\n')
print(json.dumps({'actual_full_cases':cases,'CPU_condition_accuracy':controls,'cache_bytes':s['geometry_extra_bytes_per_rank']},indent=2))
