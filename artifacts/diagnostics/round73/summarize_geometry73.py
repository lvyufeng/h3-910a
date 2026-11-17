#!/usr/bin/env python3
"""Same-fixture geometry results; passing bits alone does not imply speedup."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
MODES=('native','qk80','qk128','pv80','pv128','both128','chunk1008','chunk2016','no_contiguous')


def summarize(root=ROOT):
    raw=json.loads((root/'geometry01.json').read_text())
    if not raw.get('completed')or raw.get('failure')or {row['call']for row in raw['calls']}!={0,17,33,53}:raise ValueError('Failed actual geometry probe')
    reports={mode:[]for mode in MODES}
    for row in raw['calls']:
        if set(row['cases'])!=set(MODES)or not row['input_immutable']:raise ValueError('Missing actual geometry/inputimmutable controls')
        for role,p in row['native_repeat_full_stages'].items():
            if not p['raw_bits_equal']or not p['finite']or p['max_abs_error']!=0 or p['different_words']!=0:raise ValueError('Native attention reference not stable')
        for mode,case in row['cases'].items():
            if case.get('failure'):
                reports[mode].append({'call':row['call'],'failure':case['failure']});continue
            if set(case['full_stage_proofs'])!={'unscaled_qk','scores','probabilities','partial_pv','output'}:raise ValueError('Missing fullmathematicalstage proof')
            exact=True
            for role,p in case['full_stage_proofs'].items():
                shape=[1,16,4032,72]if role in('partial_pv','output')else[1,16,4032,4032]
                dtype='torch.bfloat16'if role=='output'else'torch.float32'
                count=4644864 if role in('partial_pv','output')else 260112384
                if p['shape']!=shape or p['dtype']!=dtype or p['elements']!=count:raise ValueError('Truncated matrix proof')
                exact=exact and p['raw_bits_equal']and p['finite']and p['different_words']==0 and p['max_abs_error']==0 and p['relative_rms']==0
            if len(case['warm'])!=3:raise ValueError('Missing warm geometry calls')
            timings=[x['seconds']for x in case['warm']]
            median=statistics.median(timings)
            if median!=case['median_s']:raise ValueError('Wrong warm median')
            reports[mode].append({'call':row['call'],'all_full_stages_bits_equal':exact,'native_output_layout_equal':case['output_native_layout_equal'],
                                  'median_s':median,'same_call_native_s':row['cases']['native']['median_s'],
                                  'speedup_vs_same_fixture_native':row['cases']['native']['median_s']/median})
    result={'scope':'Four actual real attention fixtures; full QK/scale/softmax/PV/BF16rawmatrix proof and3warm same-input standalone calls. No fullpipeline/encoderintegration or officialquality acceptance. Querychunk implementation repeats native qkv casts per chunk; its timing does not reject optimized shared-cast chunk implementations.',
            'modes':reports,'full_stage_matrix_proofs':4*9*5,'production_deployed':False,'whole_service_text_under2s_proven':False,'official_goal_complete':False}
    return result


if __name__=='__main__':
    r=summarize();(ROOT/'geometry_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
