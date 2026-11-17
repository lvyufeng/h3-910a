#!/usr/bin/env python3
"""Verify all eight KV-padded encoder receipts before summarizing timing/error.

Bit stability does not by itself mean CPU precision agreement or full encoder2s;
no numerical pass threshold is selected here. Rank maxima are matched by call.
"""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
rows=[]
for rank in range(8):
    path=ROOT/'ranks'/('tp8_rank%d.json'%rank)
    r=json.loads(path.read_text());assert r['rank']==rank and r['world_size']==8
    assert r['completed']and r['cases']['final_checks']['repeat_verified']
    assert r['cases']['repeat_vs_first']['raw_bits_equal']
    assert r['cases']['cpu_fp16_comparison']['npu_hidden_finite']
    assert not r['full_encoder_precision_accepted']and not r['whole_goal_complete']
    pad=r['kv_padding_final'];assert pad['cache_entries']==100
    assert pad['max_abs_prefix_mismatch']==0 and pad['tail_nonzero_count']==0
    assert pad['all_prefix_exact']and pad['all_cached_padded_are_zero_offset']
    assert r['frozen_inputs_unchanged']
    timing=r['timing']['samples_synchronized_wall_s_including_d2h']
    assert len(timing)==3
    rows.append({'rank':rank,'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'record':r,'timing':timing})
call_max=[max(item['timing'][call]for item in rows)for call in range(3)]
result={'scope':__doc__,'all8_ranks_repeat_bits_equal':True,'rank_max_complete_encode_s':call_max,'rank_max_complete_encode_median_s':statistics.median(call_max),
        'all_ranks_stock_encode_ids_timed':all(item['record']['stock_encode_ids_timed']for item in rows),
        'CPU_selected50_relative_RMS_by_rank':[item['record']['cases']['cpu_fp16_comparison']['selected50_pre_final_norm']['relative_rms']for item in rows],
        'embedding_numeric_relative_RMS_by_rank':[item['record']['cases']['cpu_fp16_comparison']['embedding_full_numeric']['relative_rms']for item in rows],
        'selected50_token_subsets_rank0':rows[0]['record']['cases']['cpu_fp16_comparison']['selected50_token_subsets'],
        'kv_padding_rank0':rows[0]['record']['kv_padding_final'],
        'source_and_input_hashes_unchanged':True,'production_modified':False,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
        'receipts':[{'rank':item['rank'],'sha256':item['receipt_sha256']}for item in rows]}
with(ROOT/'summary81.json').open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
