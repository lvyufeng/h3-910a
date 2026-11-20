#!/usr/bin/env python3
"""Compare actual full50-layer first-step GPU/910A activations and predictions.

Exact shared noise/conditioning/sigma bundle, original verified checkpoint, TP2/USP4.
All actual layer states have whole-array finite/range checks. Error metrics use only
stored complete-channel row samples except audio, whose 414 rows are all stored.
No whole-video quality, multi-step agreement, model-tolerance pass, or speed claim.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from torch_pt_numpy123 import load_tensor_archive
from compare_refiner_numpy123 import metric

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'full_first_step_GPU_vs_NPU123.json'
    assert not output.exists()
    bundle = load_tensor_archive(ROOT / 'same_initial_sampling123.pt')
    roots = {'GPU': ROOT/'GPU_full_first_step', 'NPU': ROOT/'NPU_full_first_step'}
    receipts = {host:{rank:json.loads((path/f'full_first_step_rank{rank}.json').read_text()) for rank in range(8)} for host,path in roots.items()}
    for host in roots:
        for rank, data in receipts[host].items():
            assert data['completed'] and data['DiT_forward_count']==1 and data['backbone_layers']==50
            assert len(data['blocks'])==50 and all(x['completed'] and x['finite'] for x in data['blocks'].values())
    for rank in range(8):
        g,n=receipts['GPU'][rank],receipts['NPU'][rank]
        assert g['exact_sampling_payload_sha256']==n['exact_sampling_payload_sha256']
        assert g['first_step_unique_timesteps']==n['first_step_unique_timesteps']
        assert g['SP_rank']==n['SP_rank']
        assert g['all_parameter_names_loaded_count']==n['all_parameter_names_loaded_count']
    weights=json.loads((ROOT/'NPU_full_checkpoint_verified123.json').read_text())
    assert weights['completed'] and weights['all_13_original_shards_and_config_index_identical']
    result={'scope':__doc__,'completed':False,'real_backbone_layers':50,'real_DiT_forward_count':1,
            'GPU_encoder_teacher_forced':True,'exact_initial_sampling_bundle_bits_identical':True,
            'all13_NPU_original_checkpoint_shards_match_pinned_GPU_reference':True,
            'all50_blocks_all8_ranks_finite_both_devices':True,
            'full49_generation_or_VAE_decode_executed':False,'multi_step_alignment_accepted':False,
            'full_model_quality_accepted':False,'numeric_tolerance':None,'GPU_reference_denominator':True,
            'metric_scope':'64 selected complete-channel local rows for layer states,512 selected target video latent rows,all414 audio latent rows',
            'layers_whole_tensor_range':{},'comparisons':{},'short_vs_full_first_two_consistency':{}}
    for i in range(50):
        result['layers_whole_tensor_range'][str(i)]={
            host:{'max_abs_across_ranks':max(receipts[host][rank]['blocks'][str(i)]['max_abs'] for rank in range(8)),
                  'finite_all_ranks':all(receipts[host][rank]['blocks'][str(i)]['finite'] for rank in range(8))}
            for host in roots}
    stages=[f'block_{i:02d}_{phase}_local_rows' for i in [0,1,24,49] for phase in ['input','output']]
    stages+=['video_velocity_all_rows','audio_velocity_all_rows','video_velocity_target_rows']
    for rank in [0,2,4,6]:
        for name in stages:
            if rank!=0 and 'velocity' in name:continue
            paths={host:path/f'{name}_sample_rank{rank}.pt' for host,path in roots.items()}
            if not all(p.is_file() for p in paths.values()):continue
            data={host:load_tensor_archive(path) for host,path in paths.items()}
            g,n=data['GPU'],data['NPU']
            assert g['full_shape']==n['full_shape'] and np.array_equal(g['row_indices'],n['row_indices'])
            for host,path in paths.items():
                expected=receipts[host][rank]['captures'][name]['sample_file_sha256']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
            row={'sample_shape':list(g['value'].shape),'full_shape':g['full_shape'],'dtype':str(g['value'].dtype),
                 'all_sample_rows':metric(n['value'],g['value']),
                 'all_rows_error_measured':len(g['row_indices'])==g['full_shape'][0],
                 'whole_tensor_max_abs':{host:receipts[host][rank]['captures'][name]['max_abs'] for host in roots},
                 'by_sample_token_tag':{}}
            if 'local_rows' in name:
                indices=g['row_indices']+receipts['GPU'][rank]['SP_rank']*g['full_shape'][0]
                tags=bundle['token_tags'][indices]
                for tag in np.unique(tags):
                    mask=tags==tag
                    row['by_sample_token_tag'][str(int(tag))]=metric(n['value'][mask],g['value'][mask])
            if row['all_rows_error_measured']:
                for host in roots:
                    value=data[host]['value']
                    assert hashlib.sha256(value.tobytes()).hexdigest()==receipts[host][rank]['captures'][name]['full_tensor_raw_sha256']
            result['comparisons'][f'{name}_rank{rank}']=row
            print(json.dumps({'stage':name,'rank':rank,'whole_error':row['all_rows_error_measured'],**row['all_sample_rows']}),flush=True)
    for host,short_directory in [('GPU','GPU_first_blocks_reference'),('NPU','NPU_first_blocks_after_patch_fix')]:
        short=json.loads((ROOT/short_directory/'first_blocks_rank0.json').read_text())
        checks={}
        for name in [f'block_{i:02d}_{phase}_local_rows' for i in [0,1] for phase in ['input','output']]:
            full=receipts[host][0]['captures'][name]
            a=load_tensor_archive(ROOT/short_directory/f'{name}_sample_rank0.pt')
            b=load_tensor_archive(roots[host]/f'{name}_sample_rank0.pt')
            assert np.array_equal(a['row_indices'],b['row_indices'])
            checks[name]={'whole_tensor_digest_identical':short['captures'][name]['sha256_raw_tensor_bytes']==full['full_tensor_raw_sha256'],
                          'sample_error':metric(b['value'],a['value'])}
        result['short_vs_full_first_two_consistency'][host]=checks
    result['all_rank0_layer_prediction_samples_compared']=all(f'{name}_rank0' in result['comparisons'] for name in stages)
    assert result['all_rank0_layer_prediction_samples_compared']
    result['completed']=True
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
