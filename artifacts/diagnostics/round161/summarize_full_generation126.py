#!/usr/bin/env python3
"""Finalize successful R126 full-generation timing and media evidence, no model work."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def main():
    generated=ROOT/'NPU_generated126';r=json.loads((generated/'generation_result126.json').read_text())
    assert r['completed'] and r['actual_video_generated']
    events=[]
    for path in generated.glob('worker_alignment_pid*126.jsonl'):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    ends=[e for e in events if e['event']=='dit_step_end'];assert len(ends)==392
    grouped=defaultdict(list)
    for e in ends:grouped[e['step_index']].append(e)
    assert sorted(grouped)==list(range(49)) and all(len(v)==8 for v in grouped.values())
    step_max=[max(e['elapsed_s_includes_offload'] for e in grouped[s]) for s in range(49)]
    baseline=json.loads((ROOT/'R125_FA_same_scope_timing126.json').read_text())
    cache=[e for e in events if e['event']=='R126_VAE_operand_cache_end'];assert len(cache)==8
    begins=[e for e in events if e['event']=='R126_VAE_operand_cache_begin'];assert len(begins)==8
    decodes=[e for e in events if e['event']=='native_NPU_VAE_decode_end'];assert len(decodes)==8
    topology=json.loads((ROOT/'topology_query126.json').read_text());assert topology['exit_code']==0
    med=statistics.median(step_max[1:])
    summary={'scope':'Real live-encoder 49-step combined R126 regression with original parameters, exact safety checks and proven VAE operand cache; not automatic quality acceptance',
        'completed':True,'generation_wall_s':r['generation_wall_s'],'engine_initialization_s':r['engine_initialization_s'],
        'baseline_FA_generation_wall_s':1329.960543639958,
        'generation_wall_speedup_vs_previous_run':1329.960543639958/r['generation_wall_s'],
        'whole_forward_hook_warm_rank_max_median_s':med,
        'baseline_FA_same_whole_forward_hook_warm_rank_max_median_s':baseline['warm_rank_max_median_s'],
        'same_scope_hook_ratio':baseline['warm_rank_max_median_s']/med,
        'whole_forward_hook_first_rank_max_s':step_max[0],
        'whole_forward_hook_warm_rank_max_min_s':min(step_max[1:]),
        'whole_forward_hook_warm_rank_max_max_s':max(step_max[1:]),
        'whole_forward_hook_each_step_rank_max_s':step_max,
        'strict_Half_checks_passed':sum(e['actual_Half_matrix_modules_this_step'] for e in ends),
        'actual_DiT_evaluations_per_worker':r['actual_DiT_counts_by_worker_pid'],
        'main_original_DiT_blocks':50,'encoder_live_not_reused':True,
        'DiT_TP':2,'USP':4,'encoder_TP':8,'participating_NPUs':8,
        'TP_enabled_and_weight_shards_verified':True,'interconnect_two_HCCS_islands':[[0,1,2,3],[4,5,6,7]],
        'native_VAE_combined_decode_rank_max_s':max(e['elapsed_s'] for e in decodes),
        'VAE_operand_prime_rank_max_s':max(e['prime_s'] for e in begins),
        'VAE_original_FP32_parameters_unchanged':all(e['cache']['original_parameters_unchanged'] for e in cache),
        'VAE_actual_cached_operand_calls_each_rank':[e['cache']['cached_operand_call_count'] for e in cache],
        'decoded_frames':r['media_verification']['decoded_video_frames'],
        'fps':r['media_verification']['fps'],'audio_sample_rate':r['media_verification']['sample_rate'],
        'media_video_sha256':r['video_sha256'],'media_audio_sha256':r['audio_sha256'],
        'warm_isolated_step_timer_must_not_replace_whole_step_timer':True,
        'timing_difference_not_pure_causal_operator_comparison':True,
        'new_precision_policy_accepted':False,'all_long_term_latency_targets_met':False,'A2_A3_hardware_comparison_measured':False}
    target=ROOT/'full_generation_summary126.json';assert not target.exists()
    target.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='whole_forward_hook_each_step_rank_max_s'},indent=2))

if __name__=='__main__':main()
