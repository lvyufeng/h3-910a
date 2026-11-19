#!/usr/bin/env python3
"""Aggregate completed eight-rank full-stock encoder clocks without changing acceptance."""
from pathlib import Path
import json
import statistics


def collect(root):
    root=Path(root)
    ranks=[json.loads((root/('tp8_rank%d.json'%i)).read_text())for i in range(8)]
    assert all(r['completed']and r['rank']==i and r['world_size']==8 for i,r in enumerate(ranks))
    assert json.loads((root/'run_finished.json').read_text())['exit_code']==0
    names=('native','corrected_vision_native_text','corrected_bc128_vision_native_text','corrected_bc128_vision_precise_text')
    assert all(tuple(r['cases'])==names for r in ranks)
    result={'scope':__doc__,'world_size':8,'completed':True,'full_encoder_precision_accepted':False,
            'whole_goal_complete':False,'numerical_tolerance':None,'cases':{},
            'no_default_promotion':True,'no_FloatLinear_teacher_BFgoldcast_or_hidden_condition_cache':True}
    for name in names:
        cases=[r['cases'][name]for r in ranks];base=cases[0]
        clocks=[max(c['warm_stock_s'][j]for c in cases)for j in range(5)]
        rms=[c['selected50_vs_CPU']['relative_rms']for c in cases]
        hashes=[c['selected50_vs_CPU']['npu']['sha256']for c in cases]
        boundaries=[[x['output']['sha256']for x in c['raw50_layers']]for c in cases]
        assert len(set(hashes))==1 and all(v==boundaries[0]for v in boundaries)
        assert all(c['repeat_vs_first']['raw_bits_equal']and c['stock_vs_equivalent']['raw_bits_equal']
                   and c['matches_actual_R101_vision_merged_and_all27_boundary_SHA']for c in cases)
        result['cases'][name]={'selected50_relativeRMS_rank0':rms[0],'selected50_relativeRMS_all8_min_max':[min(rms),max(rms)],
            'selected50_SHA256':hashes[0],'all8_selected50_and_all50_text_boundary_SHAs_equal':True,
            'vision_relativeRMS':base['vision_merged_vs_CPU']['relative_rms'],
            'DeepStack_relativeRMS':[x['relative_rms']for x in base['vision_DeepStack_vs_CPU']],
            'raw_selected_layers_relativeRMS':{str(x['layer']):x['vs_CPU']['relative_rms']for x in base['raw50_layers']if 'vs_CPU'in x},
            'paired_8rankmax_clock_s':clocks,'median_full_stock_s':statistics.median(clocks),
            'all8_repeat_stock_equivalent_and_R101_visual_SHA_verified':True}
    assert all(r['native_recovery']['raw_bits_equal']and r['parameter_and_fixture_SHA_unchanged']and r['CPU_reference_SHA_unchanged']for r in ranks)
    result['all8_native_recovery_parameters_fixture_originalCPUgold_unchanged']=True
    result['fastest_full_stock_median_s']=min(c['median_full_stock_s']for c in result['cases'].values())
    result['all_full_stock_samples_at_least2s']=all(x>=2 for c in result['cases'].values()for x in c['paired_8rankmax_clock_s'])
    result['CPU_condition_only_accuracy_controls_not_warm_benchmark']={name:c['selected50_vs_CPU']['relative_rms']for name,c in ranks[0]['CPU_condition_controls'].items()}
    return result


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    result=collect(root/'results/full01')
    with(root/'summary102.json').open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
