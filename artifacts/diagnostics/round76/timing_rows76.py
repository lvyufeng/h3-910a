"""Parse eight-rank synchronized full-step timers,not block-stack timings."""
import re
import statistics

STEP=re.compile(r'(?:\(DiffusionWorker_SP(\d+) pid=\d+\)\s*)?\[H3PERF\] dstep_complete (\d+) total=([0-9.]+)s callback_and_sync=([0-9.]+)s')
DENOISE=re.compile(r'\[H3PERF\] denoise_complete steps=(\d+) first=([0-9.]+)s after_first_avg=([0-9.]+)s observed_total=([0-9.]+)s last=([0-9.]+)s')


def complete_timing_cases(text,case_count):
    if case_count not in(1,2):raise ValueError('This sequential controlled run has one or two complete cases')
    matches={i:[]for i in range(1,50)}
    for match in STEP.finditer(text):
        rank,index,total,callback=match.groups();index=int(index)
        if index not in matches:raise ValueError('Non49step request in actual video timing log')
        matches[index].append({'rank':int(rank)if rank is not None else None,'seconds':float(total),
                               'callback_and_sync_s':float(callback),'log_line':text.count('\n',0,match.start())+1})
    denoises=[{'steps':int(i),'first_s':float(a),'after_first_average_s':float(b),'observed_total_s':float(c),'last_s':float(d)}for i,a,b,c,d in DENOISE.findall(text)]
    if len(denoises)<8*case_count:raise ValueError('Missing eight-rank completed-denoise records')
    if any(row['steps']!=49 for row in denoises):raise ValueError('Different-step request in denoise summary')
    results=[]
    for case in range(case_count):
        steps=[]
        for index,records in matches.items():
            selected=records[case*8:(case+1)*8]
            if len(selected)!=8:raise ValueError('Incomplete eight-rank full-step timing')
            named=[r['rank']for r in selected if r['rank']is not None]
            if len(named)!=len(set(named)):raise ValueError('Duplicate rank in actual full-step timer')
            if named==[]or not set(named)<=set(range(8)):raise ValueError('Invalid actual rank header')
            # IPC logging sometimes omits SP0's prefix. Accept exactly one bare
            # message only when the other seven named ranks are 1..7;do not
            # invent rank attribution for any other missing-prefix pattern.
            if any(r['rank']is None for r in selected):
                if len(named)!=7 or set(named)!=set(range(1,8)):raise ValueError('Ambiguous missing-rank prefix')
                for row in selected:
                    if row['rank']is None:row['rank']=0;row['rank0_inferred_from_other_seven']=True
            if {row['rank']for row in selected}!=set(range(8)):raise ValueError('Missing actual rank')
            steps.append({'step':index,'same_call_rank_max_s':max(row['seconds']for row in selected),
                          'same_call_rank_min_s':min(row['seconds']for row in selected),'rank_observations':selected})
        values=[row['same_call_rank_max_s']for row in steps]
        denoise=denoises[case*8:(case+1)*8]
        results.append({'steps':steps,'full_step_measurement_scope':'Prep+complete model+update+callback+device sync;max of8 ranks for SAME step. Not DiT block stack and not FP32 theoretical estimate.',
                        'first_step_s':values[0],'second_step_s':values[1],
                        'after_first_same_call_rank_max_mean_s':statistics.mean(values[1:]),
                        'after_first_two_same_call_rank_max_mean_s':statistics.mean(values[2:]),
                        'after_first_two_max_s':max(values[2:]),
                        'all49_step_rank_max_sum_s':sum(values),'median_full_step_rank_max_s':statistics.median(values),
                        'all49_count_strictly_below15s':sum(x<15 for x in values),'after_first_two_count_strictly_below15s':sum(x<15 for x in values[2:]),
                        'all49_strictly_below15s':all(x<15 for x in values),'denoise_complete_rank_rows':denoise,
                        'denoise_observed_total_rank_max_s':max(row['observed_total_s']for row in denoise)})
    stages={stage:[float(value)for value in re.findall(r'\[H3PERFP\] '+stage+r'=([0-9.]+)s',text)]for stage in('encode','denoise','decode')}
    for case,row in enumerate(results):
        row['stage_rank_rows']={}
        for stage,records in stages.items():
            selected=records[case*8:(case+1)*8]
            if len(selected)!=8:raise ValueError('Incomplete live stage timings:'+stage)
            row['stage_rank_rows'][stage]={'rows':selected,'max_s':max(selected),'min_s':min(selected)}
        row['encode_stage_scope']='External text hidden/tags broadcast+LIVE VAE media encode/conditioning preparation;does NOT include fresh encoder compute.'
        row['decode_stage_scope']='Denoise cache release/residency+live videoVAE+audioVAE+audioD2H;excludes uint8 frame quantization,videoD2H/worker transfer and MP4 export.'
    return results
