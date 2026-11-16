#!/usr/bin/env python3
"""Verify all receipts and summarize the controlled block experiment, not E2E."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
ROWS=[json.loads(p.read_text()) for p in sorted((ROOT/'block_replay').glob('block_replay02_rank*.json'))]
assert len(ROWS)==8 and {r['rank'] for r in ROWS}==set(range(8)) and all(r['completed'] for r in ROWS)
MODES=('baseline','point_gate','lossless_split','norm_fused_exploratory')
summary={'rank_count':8,'fixture_layers':[0,24,49],'fixture_shape':[5224,5376],'per_layer':[],
         'scope':'Complete teacher-forced resident one-block USP8/TP1. These latencies exclude offload, full denoise steps, encoder, VAE and stage/request overhead.',
         'measurement_limits':'Three warmed synchronized calls per mode, fixed mode order; small sub-1ms differences are not a sustained end-to-end acceleration guarantee.'}
for layer in (0,24,49):
    cases=[next(x for x in r['layers'] if x['layer']==layer) for r in ROWS]
    assert all(x['loaded_parameter_tensors']==10 and x['parallel_strategy']=='ulysses' and x['attention_backend']=='H3_910A' for x in cases)
    row={'layer':layer,'cases':{}}
    for mode in MODES:
        receipts=[x['cases'][mode] for x in cases]
        groups=[max(x['warm_groups_ms'][i] for x in receipts) for i in range(3)]
        row['cases'][mode]={'max_rank_warm_groups_ms':groups,'median_max_rank_ms':statistics.median(groups),
                            'all_captures_bits_equal':all(x['vs_capture']['raw_bits_equal'] for x in receipts),
                            'all_baselines_bits_equal':all(x['vs_baseline']['raw_bits_equal'] for x in receipts),
                            'all_first_second_third_fourth_repeat_exact':all(x[k]['raw_bits_equal'] for x in receipts for k in ('first_second','second_third','third_fourth')),
                            'relative_rms_range_vs_baseline':[min(x['vs_baseline']['relative_rms'] for x in receipts),max(x['vs_baseline']['relative_rms'] for x in receipts)],
                            'max_abs_error_vs_baseline':max(x['vs_baseline']['max_abs_error'] for x in receipts)}
        assert row['cases'][mode]['all_first_second_third_fourth_repeat_exact']
        if mode!='norm_fused_exploratory':assert row['cases'][mode]['all_captures_bits_equal'] and row['cases'][mode]['all_baselines_bits_equal']
        else:assert not row['cases'][mode]['all_baselines_bits_equal']
    summary['per_layer'].append(row)
summary['exact_outputs_per_accepted_mode']=sum(x['cases']['lossless_split']['vs_capture']['elements'] for r in ROWS for x in r['layers'])
summary['all_accepted_modes_match_actual_capture']=True
summary['norm_fusion_lossless_rejected']=True
summary['production_fusion_installed']=False
proof=json.loads((ROOT/'final_source_verification.json').read_text())
assert proof['all_production_unchanged'] and not proof['production_fusion_installed']
local_sources={'/tmp/h3_round61_adaln01/adaln_vector61.py':ROOT.parent/'round61'/'adaln_vector61.py',
               '/tmp/h3_round61_adaln01/point_vector61.py':ROOT.parent/'round61'/'point_vector61.py',
               '/tmp/h3_round61_adaln01/vec61_launcher.c':ROOT.parent/'round61'/'vec61_launcher.c',
               '/tmp/h3_round61_adaln01/point_guard_validation.py':ROOT.parent/'round61'/'point_guard_validation.py',
               '/tmp/h3_round62_blocks01/block_replay_tp8.py':ROOT/'block_replay_tp8.py'}
for remote,local in local_sources.items():assert hashlib.sha256(local.read_bytes()).hexdigest()==proof['diagnostic_sources'][remote],remote
snapshots={'diffusion/layers/indexed_modulation.py':ROOT.parent/'round61'/'source'/'indexed_modulation.py',
           'diffusion/layers/fused_qk_norm_rope.py':ROOT/'source'/'fused_qk_norm_rope.py',
           'diffusion/attention/layer.py':ROOT/'source'/'attention_layer.py',
           'diffusion/attention/parallel/factory.py':ROOT/'source'/'attention_parallel_factory.py'}
for remote,local in snapshots.items():assert hashlib.sha256(local.read_bytes()).hexdigest()==proof['production'][remote]['sha256'],remote
summary['local_source_matches_executed_source']=True
summary['additional_dispatch_sources_match_original_snapshots']=True
summary['goal_completed']=False
summary['official_matched_quality_benchmark_completed']=False
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
