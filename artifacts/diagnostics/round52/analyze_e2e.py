#!/usr/bin/env python3
"""Report complete video/stage evidence with explicit incomplete-step timing caveats."""
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
r=json.loads((root/'e2e01/result.json').read_text())
assert r['response']['status']=='completed' and 'failure' not in r
assert r['verified_768p_5s_frame_alignment']
video=next(x for x in r['mp4_metadata']['tracks'] if x['handler']=='vide')
assert (video['width'],video['height'],video['sample_count'])==(1344.,768.,124)
assert len(r['fa_padding_plans'])==8 and all(x==[41792,43776,41792,23] for x in r['fa_padding_plans'])
steps=[]
for i in (1,2,3,4):
    rows=[x for x in r['step_rows'] if x['step']==i]
    assert len(rows)==8
    steps.append({'step':i,'ranks':len(rows),'mean_model_timer_s':statistics.mean(x['model_s'] for x in rows),
                  'mean_prep_model_update_sum_s':statistics.mean(x['prep_s']+x['model_s']+x['update_s'] for x in rows),
                  'qualification':'These timers end before callback/cancellation synchronization; not full synchronized step latency.'})
s={'request_completed':True,'video_track':video,'request_wall_s':r['request_wall_seconds'],
   'startup_excluded_from_request_s':r['startup_seconds'],
   'stages_s':{name:{'min':min(v),'max':max(v),'mean':statistics.mean(v)} for name,v in r['stage_times'].items()},
   'step_reporter_after_first_avg_s':sorted(set(x['steady_avg_s'] for x in r['steady_rows'])),
   'step_subtimers':steps,
   'reporter_qualification':'The old steady label averages steps 2-4, including warmup effects. steady_total is this average times all steps, not measured total; projected_49steps is not an official comparison.',
   'goal_qualification':'One four-step distilled checkpoint request with real first/last images; no 49-step original-checkpoint lossless comparison, pure text-encoder target, decode target or full synchronized 15s-step claim.'}
p=root/'e2e_summary.json'
if p.exists(): raise RuntimeError('Refuse to overwrite an existing summary')
p.write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
