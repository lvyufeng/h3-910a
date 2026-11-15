#!/usr/bin/env python3
"""Summarize actual synchronized native steps, without extrapolating a student."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'native49_e2e03/result.json').read_text())
assert r.get('response',{}).get('status')=='completed' and 'failure' not in r
assert r['request_fields']['num_inference_steps']=='49'
assert r['verified_768p_5s_frame_alignment']
assert len(r['complete_step_rows'])==49*8
steps=[]
for i in range(1,50):
    rows=[x for x in r['complete_step_rows'] if x['step']==i]
    assert len(rows)==8
    steps.append({'step':i,'ranks':8,'min_s':min(x['total_s'] for x in rows),'max_s':max(x['total_s'] for x in rows),
                  'mean_s':statistics.mean(x['total_s'] for x in rows),
                  'mean_callback_and_sync_s':statistics.mean(x['callback_and_sync_s'] for x in rows)})
# Rank-max is a conservative critical path proxy; separately reported because
# ranks are not globally barrier-synchronized at every timer start.
after=[x['max_s'] for x in steps[1:]]
late=[x['max_s'] for x in steps[5:]]
after_two=[x['max_s'] for x in steps[2:]]
summary={'completed':True,'actual_intervals':49,'rank_count':8,'request_wall_s':r['request_wall_seconds'],
         'startup_s':r['startup_seconds'],'stages_s':{k:{'reported_ranks':len(v),'min':min(v),'max':max(v),'mean':statistics.mean(v)} for k,v in r['stage_times'].items()},
         'first_complete_step_rank_max_s':steps[0]['max_s'],
         'second_complete_step_rank_max_s':steps[1]['max_s'],
         'per_step_rank_max_steps3_49_mean_s':statistics.mean(after_two),
         'steps3_49_rank_max_below15_count':sum(t<15 for t in after_two),
         'steps3_49_rank_max_min_s':min(after_two),'steps3_49_rank_max_max_s':max(after_two),
         'per_step_rank_max_after_first_mean_s':statistics.mean(after),
         'per_step_rank_max_late_steps6_49_mean_s':statistics.mean(late),
         'per_step_rank_max_late_median_s':statistics.median(late),
         'per_step_rank_max_last_s':steps[-1]['max_s'],
         'rank_max_observed_step_sum_s':sum(x['max_s'] for x in steps),
         'reported_complete_denoise_rows':r['complete_denoise_rows'],'step_rows':steps,
         'mp4_metadata':r['mp4_metadata'],
         'qualification':'Actual native 49-interval same-input request with explicit PERF-only device drains before timers; per-rank maxima are not a per-step global barrier measurement. Native weight storage matches existing FP16 cast, not BF16-bit-exact or official lossless quality proof.'}
out=ROOT/'native49_summary.json'
if out.exists(): raise RuntimeError('Refuse to overwrite an existing summary')
out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('step_rows','reported_complete_denoise_rows')},indent=2))
