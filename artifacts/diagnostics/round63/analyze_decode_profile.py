#!/usr/bin/env python3
"""Reconcile first-two-tile H3 VAE stream traces without double-counted totals."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
profiles=[json.loads(p.read_text()) for p in sorted((ROOT/'profiles').glob('decode_profile_rank*.json'))]
assert len(profiles)==8 and {p['rank'] for p in profiles}==set(range(8))
summary={'rank_count':8,'ranks':[],
         'scope':'Observed first-two decoder tile forwards in an instrumented one-step real FL2VA request; first tile includes tensor capture effects. Not clean E2E/steady-state attribution or a hardware ceiling.',
         'classification_note':'Linear, Norm, SiLU and perform_attention rows are disjoint module calls; parent decoder/attention/FF/block rows are inclusive and must not be added to them. Float casts/reshapes/RoPE/residuals are not separately timed here.'}
for p in profiles:
    row={key:p[key] for key in ('rank','latent','decoder_class','decoder_config','autocast_enabled','autocast_dtype','decoder_parameter_bytes','decoder_parameter_dtypes','linear_weight_dtypes','parameter_inference_tensor_counts','parameter_requires_grad_counts','optimized_operator_installed','total_tile_decoder_calls','profiled_tiles','output')}
    row['first_two_same_shape']=p['profiled_tiles'][0]['input']['shape']==p['profiled_tiles'][1]['input']['shape']
    row['tiles']=[]
    for tile in (0,1):
        events=[e for e in p['events'] if e['tile']==tile]
        roots=[e for e in events if e['name']=='decoder'];assert len(roots)==1
        groups={'linear':[],'norm':[],'silu':[],'attention_core':[]}
        for e in events:
            name=e['name']
            if name.endswith(('to_qkv','to_out','w1','w2')) or name in ('x_embedder','proj_out'):groups['linear'].append(e)
            elif name.endswith(('norm1','norm2','norm_q','norm_k')) or name=='norm_out':groups['norm'].append(e)
            elif name.endswith('act_fn'):groups['silu'].append(e)
            elif name.endswith('perform_attention'):groups['attention_core'].append(e)
        t={'tile':tile,'decoder_inclusive_stream_ms':roots[0]['inclusive_stream_ms'],'decoder_host_enqueue_ms':roots[0]['host_enqueue_ms'],
           'leaf_groups':{key:{'calls':len(values),'sum_observed_stream_ms':sum(e['inclusive_stream_ms'] for e in values),'sum_host_enqueue_ms':sum(e['host_enqueue_ms'] for e in values)} for key,values in groups.items()},
           'largest_observed_leaf_calls':sorted([e for values in groups.values() for e in values],key=lambda x:x['inclusive_stream_ms'],reverse=True)[:12]}
        row['tiles'].append(t)
    summary['ranks'].append(row)
summary['total_decoder_calls_per_rank']=[p['total_tile_decoder_calls'] for p in profiles]
summary['parameter_bytes_per_rank']=[p['decoder_parameter_bytes'] for p in profiles]
summary['all_decode_autocast_fp16']=all(p['autocast_enabled'] and p['autocast_dtype']=='torch.float16' for p in profiles)
summary['any_official_vae_operator_installed']=any(p['optimized_operator_installed'] for p in profiles)
(ROOT/'profile_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'all_decode_autocast_fp16':summary['all_decode_autocast_fp16'],'official_ops_installed':summary['any_official_vae_operator_installed'],
                  'decoder_calls':summary['total_decoder_calls_per_rank'],'parameter_bytes':summary['parameter_bytes_per_rank'],
                  'rank0':summary['ranks'][0]},indent=2))
