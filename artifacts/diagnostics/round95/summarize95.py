#!/usr/bin/env python3
"""Verify real eight-rank full encoder CPU-exact RoPE-table A/B proof and clocks."""
from pathlib import Path
import json
import hashlib
import statistics
ROOT=Path(__file__).resolve().parent
rows=[];receipts=[]
for rank in range(8):
    p=ROOT/'results/full01/ranks'/('tp8_rank%d.json'%rank);r=json.loads(p.read_text())
    assert r['rank']==rank and r['completed']and r['native_matches_R82_output_SHA']
    assert r['native_recovery']['raw_bits_equal']and r['parameter_and_fixture_SHA_unchanged']and r['CPU_reference_SHA_unchanged']
    assert all(r['vision_boundaries_unchanged_by_CPU_rope'].values())
    assert r['CPU_rope_prime']['cpu_table']['matches_R94_measured_CPU_table']and r['CPU_rope_prime']['native_probe']['matches_R94_observed_native_SHA']
    assert r['CPU_rope_final']['extra_bytes']==1046016 and r['CPU_rope_final']['teacher_weights']==0
    assert all(c['sha256_unchanged_since_prime']for c in r['CPU_rope_final']['cached'].values())
    rows.append(r);receipts.append({'rank':rank,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
cases={}
for name in rows[0]['cases']:
    for r in rows:
        c=r['cases'][name]
        assert len(c['raw50_layers'])==50 and len(c['raw27_vision_layers'])==27
        assert c['repeat_vs_first']['raw_bits_equal']and c['stock_vs_equivalent']['raw_bits_equal']
        assert len(c['warm_stock_s'])==5 and len(set(c['warm_output_SHA256']))==1
    hashes={r['cases'][name]['selected50_vs_CPU']['npu']['sha256']for r in rows};assert len(hashes)==1
    c=rows[0]['cases'][name];clocks=[max(r['cases'][name]['warm_stock_s'][i]for r in rows)for i in range(5)]
    cases[name]={'CPU_exact_rope':c['CPU_exact_text_rope_cache'],'corrected_vision':c['vision_fixes'],'precise_attention':c['precise_text_attention'],
                 'selected50_CPU_relative_RMS':c['selected50_vs_CPU']['relative_rms'],
                 'vision_CPU_relative_RMS':c['vision_merged_vs_CPU']['relative_rms'],
                 'paired8rankmax_warm_s':clocks,'paired8rankmax_median_s':statistics.median(clocks),
                 'all5paired_max_under2s':all(x<2 for x in clocks),'output_SHA256':next(iter(hashes)),
                 'all8_repeat_stock_equivalence_and_77boundaries_verified':True}
controls={}
for name,c in rows[0]['CPU_condition_controls'].items():
    assert all(r['CPU_condition_controls'][name]['repeat']['raw_bits_equal']for r in rows)
    controls[name]={'relative_RMS':c['selected50_vs_CPU']['relative_rms'],'output_SHA256':c['selected50_vs_CPU']['npu']['sha256'],
                    'not_full_encoder_warmtiming':True}
s={'scope':__doc__,'cases':cases,'CPU_condition_accuracy_only':controls,
   'all8_CPURope_table_matches_original_CPUbits_and_unchanged':True,
   'all8_CPURope_changes_no_vision_boundary_bits':True,'all_params_fixture_CPUgold_native_recovery_unchanged':True,
   'geometry_cache_native_bit_lossless_always_enabled':True,'CPU_rope_cache_extra_bytes_per_rank':1046016,
   'CPU_rope_cache_not_image_conditioning_or_hidden_value_cache':True,'no_Float_Linear_teacher':True,
   'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,
   'maximum_rank_peak_allocated_bytes':max(r['memory']['max_allocated']for r in rows),
   'timing_scope':'actualprivateencode_ids includesrealpixels/all27vision/all50text/HCCL/layout/packing/D2H;5pairedrankmaxbarrieroutofclock;CPU93endedbeforewarm',
   'receipts':receipts}
with(ROOT/'summary95.json').open('x')as f:f.write(json.dumps(s,indent=2)+'\n')
print(json.dumps({'cases':cases,'CPU_conditions':controls},indent=2))
