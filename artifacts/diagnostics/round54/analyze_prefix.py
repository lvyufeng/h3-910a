#!/usr/bin/env python3
"""Consolidate verified real-prefix correction and outstanding quality scope."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'prefix_validation.json').read_text())
assert r['completed'] and 'failure' not in r
assert len(r['tail_tests'])==7
for item in r['tail_tests']:
    assert item['finite'] and item['max_error_to_one']<.002 and item['last_valid_key_impulse_error']<.002
cases=[]
for item in r['real_cases']:
    old=item['old']['relative_rms'];new=item['new']['relative_rms']
    cases.append({'name':item['name'],'old_relative_rms':old,'new_relative_rms':new,'error_reduction_factor':old/new,
                  'old_max_abs_error':item['old']['max_abs_error'],'new_max_abs_error':item['new']['max_abs_error']})
times={x['variant']:x['median_ms'] for x in r['timings']}
report={'prefix_tail_npu_cases':7,'real_cases':cases,'wrapper_ms':times,
        'fixed_order_wrapper_latency_reduction_percent':(1-times['new_exact_prefix']/times['old_zero_key_lse'])*100,
        'invalid_keys_not_assumed_zero':True,'arbitrary_sub16_valid_limit':41769,
        'score_probability_mask_version':3,
        'scope':'Selected real queries over all valid keys of all seven captured rank-local heads; not full-model quality or official lossless proof'}
OUT=ROOT/'prefix_summary.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite summary')
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
