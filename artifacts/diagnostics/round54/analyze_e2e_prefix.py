#!/usr/bin/env python3
"""Report actual prefix-corrected step records without over-attributing changes."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'e2e02_prefix/result.json').read_text())
assert r['response']['status']=='completed' and r['verified_768p_5s_frame_alignment']
assert r['request_fields']['num_inference_steps']=='49' and len(r['complete_step_rows'])==392
assert len(r['fa_padding_plans'])==8 and all(x==[41792,43776,41769,23] for x in r['fa_padding_plans'])
steps=[]
for i in range(1,50):
    rows=[x for x in r['complete_step_rows'] if x['step']==i];assert len(rows)==8
    steps.append({'step':i,'rank_max_s':max(x['total_s'] for x in rows),'rank_min_s':min(x['total_s'] for x in rows)})
late=[x['rank_max_s'] for x in steps[2:]];after=[x['rank_max_s'] for x in steps[1:]]
report={'completed':True,'request_wall_s':r['request_wall_seconds'],'startup_s':r['startup_seconds'],
        'stages_s':{k:{'rank_records':len(v),'min':min(v),'max':max(v)} for k,v in r['stage_times'].items()},
        'first_step_rank_max_s':steps[0]['rank_max_s'],'second_step_rank_max_s':steps[1]['rank_max_s'],
        'after_first_rank_max_mean_s':statistics.mean(after),'steps3_49_rank_max_mean_s':statistics.mean(late),
        'steps3_49_rank_max_min_s':min(late),'steps3_49_rank_max_max_s':max(late),
        'steps3_49_below15_count':sum(t<15 for t in late),'steps3_49_count':len(late),
        'last_step_rank_max_s':steps[-1]['rank_max_s'],'step_rows':steps,
        'scope':'PERF-only explicit local NPU drains; per-rank maxima not global barrier timing; concurrent CPU reference/cache variation prevents attributing full-request changes solely to the prefix patch',
        'quality_qualification':'Unprompted middle-frame logo/lettering remains; no official lossless/model-quality success.'}
OUT=ROOT/'e2e_prefix_summary.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite summary')
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='step_rows'},indent=2))
