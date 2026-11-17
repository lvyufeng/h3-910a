#!/usr/bin/env python3
"""Synchronized actual generation timing extractor;no hidden encoder cost injection."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent


def summarize(root=ROOT,generation='generation_result.json',log='generation_complete.log'):
    result=json.loads((root/generation).read_text())
    if result.get('failure')or not result.get('completed'):raise ValueError('Actual generatedvideo run incomplete/failed')
    if not result['conditioning_is_precomputed_for_this_request']or result['live_full_encoder_end_to_end_performance_proven']or result['native_bf16_compute_used']:raise ValueError('Wrong actual encoder/runtime scope')
    if result['engine_kwargs'].get('ulysses_degree')!=8 or result['engine_kwargs'].get('ring_degree')!=1 or result['engine_kwargs']['model_loaded']['text_encoder']:raise ValueError('Wrong actualSP8/externalencoderconfig')
    if [case['case']for case in result['cases']]!=['optimized','HF_reference_conditioning']:raise ValueError('Missing actual two-condition video controls')
    source=hashlib.sha256((root/'generate_optimized_fixed76.py').read_bytes()).hexdigest()
    if result['source_sha256']!=source:raise ValueError('Executedactualgenerationchanged')
    text=(root/log).read_text(errors='replace')
    rows=[{'steps':int(i),'first_s':float(a),'after_first_average_s':float(b),'observed_total_s':float(c),'last_s':float(d)}
          for i,a,b,c,d in re.findall(r'\[H3PERF\] denoise_complete steps=(\d+) first=([0-9.]+)s after_first_avg=([0-9.]+)s observed_total=([0-9.]+)s last=([0-9.]+)s',text)]
    stages={stage:[float(value)for value in re.findall(r'\[H3PERFP\] '+stage+r'=([0-9.]+)s',text)]for stage in('encode','denoise','decode')}
    from timing_rows76 import complete_timing_cases
    timing_cases=complete_timing_cases(text,2)
    for index,row in enumerate(timing_cases):
        row['case']=result['cases'][index]['case'];row['thermal_scope']='first_request_cold'if index==0 else'second_request_warmed'
        row['decode_reporter_within20s']=row['stage_rank_rows']['decode']['max_s']<=20
        row['decode_comparable_to_R64_complete_service_median']=False
        row['request_wall_s_authoritative']=result['cases'][index]['request_wall_s_excluding_live_encoder']
    return {'scope':'Two actual49stepFL2VAoutputswithFP16conditioning;excludesexternallycomputedencoder/startupandlocalpostreturnMP4export. Rawstage/timerlogsarecurrentrunobservations—notofficiallossless oradditivefullonlineE2E.',
            'engine_startup_s_excluded':result['engine_startup_seconds_excluded'],
            'cases':[{key:case[key]for key in('case','request_wall_s_excluding_live_encoder','metrics','stage_durations','video')}for case in result['cases']],
            'complete_denoise_rows':rows,'stage_times':stages,'same_call_rank_max_timing_cases':timing_cases,
            'different_raw_RGB_outputs':result['cases'][0]['raw_RGB_frame_sha256']!=result['cases'][1]['raw_RGB_frame_sha256'],
            'stage_rank_maxima_must_not_be_added_to_infer_request_wall':True,'FP16_consumed_hidden_comparison':result['FP16_consumed_hidden_comparison'],
            'native_BF16_compute_used':False,'live_encoder_in_request':False,'official_goal_complete':False,
            'note':'If rawperformance rowsareabsent,theactualrequestwall/outputmetricsremainreported;numbersarenotinferredfromoldruns.'}


if __name__=='__main__':
    r=summarize();(ROOT/'generation_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
