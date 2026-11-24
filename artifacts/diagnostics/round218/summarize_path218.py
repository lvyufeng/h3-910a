#!/usr/bin/env python3
"""Summarize same-process R218 diagnostics; inclusive region/copy spans are NOT additive."""
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ranks=[json.loads((HERE/'rank_receipts'/f'candidate_rank{rank}.json').read_text()) for rank in range(8)]
assert all(r['completed'] for r in ranks)
assert all(r['tp_actual']==2 and r['parallel_config']['ulysses_degree']==4 for r in ranks)
assert all(len(r['checks'])==6 for r in ranks)
for r in ranks:
    assert all(row['safety']['matrix_checks']==155 and row['safety']['mode']=='strict' and row['safety']['nonfinite_count']==0 and row['safety']['boundary_count']==0 for row in r['checks'])
    assert len(r['warm_event_profiles'])==2
cases=[]
for index,label in [(0,'clean_pre'),(1,'instrumented'),(2,'restored')]:
    values=[row['rank_max_elapsed_s'] for row in ranks[0]['checks'] if row['case_index']==index]
    per_rank={str(r['rank']):[row['local_elapsed_s'] for row in r['checks'] if row['case_index']==index] for r in ranks}
    cases.append({'case':label,'rank_max_seconds':values,'rank_max_median_s':statistics.median(values),'local_by_rank_s':per_rank})
regions={}
for region in ranks[0]['warm_event_profiles'][0]['regions']:
    units='kernel_total_ms' if region=='FA.ACL' else 'inclusive_stream_total_ms'
    values={str(r['rank']):[prof['regions'][region][units] for prof in r['warm_event_profiles']] for r in ranks}
    medians={rank:statistics.median(items) for rank,items in values.items()}
    regions[region]={'metric':units,'per_rank_repeat_ms':values,'per_rank_median_ms':medians,
                     'max_of_rank_medians_ms':max(medians.values()),
                     'calls_per_measured_step':ranks[0]['warm_event_profiles'][0]['regions'][region]['calls'],
                     'inclusive_overlap_cannot_sum':region!='FA.ACL'}
bytes_rows=[{'rank':r['rank'],'bytes_per_step':[p['H2D_bytes'] for p in r['warm_event_profiles']],
             'prefetch_calls_per_step':[p['prefetch_calls'] for p in r['warm_event_profiles']]} for r in ranks]
fa_rows=[]
for r in ranks:
    for i,prof in enumerate(r['warm_event_profiles']):
        assert len(prof['actual_ACL_launches'])==50
        assert all(s['shape']==[1,7,41792,128] and s['n_pad']==23 for s in prof['runtime_FA_shapes'])
        fa_rows.append({'rank':r['rank'],'repeat':i,'actual_launches':50,
                        'kernel_binaries':prof['actual_kernel_binaries']})
# Same-run hashes distinguish deterministic leaf copy candidate from intrinsic model repeat scatter.
outputs=[]
for r in ranks:
    outputs.append({'rank':r['rank'],'video_hashes':[c['video']['raw_sha256'] for c in r['checks']],
                    'audio_hashes':[c['audio']['raw_sha256'] for c in r['checks']]})
all_pre_profile_restore_identical=all(len(set(r['video_hashes']))==1 and len(set(r['audio_hashes']))==1 for r in outputs)
summary={'round':218,'scope':'current pinned 50-block eight-rank DiT replay at step25, NOT live encoder or full49step pipeline',
         'candidate_installed_in_model':False,'nothing_adopted':True,'whole_goal_complete':False,
         'tp':2,'usp':4,'all_155_strict_checks_per_step_per_rank':True,
         'cases':cases,'profile_overhead_ratio_to_clean_pre':cases[1]['rank_max_median_s']/cases[0]['rank_max_median_s'],
         'regions_instrumented_run_only':regions,'H2D_byte_counts':bytes_rows,
         'H2D_span_is_queue_inclusive_not_pure_transfer':True,
         'do_not_add_parent_child_or_copy_spans':True,
         'actual_FA_launch_and_binary':fa_rows,'outputs':outputs,
         'all_six_same_rank_outputs_bit_identical':all_pre_profile_restore_identical,
         'no_automatic_numeric_promotion':True,
         'next_scope':'Use same-run quantities to rank isolated candidates; a current critical-path trace is still needed for exact additive decomposition'}
(HERE/'path_summary218.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'cases':[{k:v for k,v in c.items() if k!='local_by_rank_s'} for c in cases],
                  'regions':{k:v['max_of_rank_medians_ms'] for k,v in regions.items()},
                  'exact_self_repeat':all_pre_profile_restore_identical},indent=2))
