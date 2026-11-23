#!/usr/bin/env python3
"""Verify R126 actual execution evidence, not rounded documentation claims.

Passing this verifier does NOT accept a candidate, prove all-workload losslessness,
prove a new 49-step generation, or declare any latency target complete.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT/path).read_text())


def cohort(name,prefix='candidate_rank',base='rank_receipts126'):
    rows=[load(f'{base}/{name}/{prefix}{i}.json') for i in range(8)]
    assert all(r['completed'] and r['rank']==i for i,r in enumerate(rows)),name
    return rows


def check(predicate,label):
    if not predicate:raise AssertionError(label)
    return label


def verify(require_final=False,require_combined=False):
    passed=[]
    for tp in (2,4,8):
        rows=[load(f'tp{tp}_meta126/rank{i}.json') for i in range(8)]
        passed.append(check(all(r['completed'] and r['tp_actual']==tp and r['local_q_heads']==56//tp for r in rows),f'live TP{tp} groups and all local heads'))
        expected=[3*56//tp*128,5376]
        passed.append(check(all(r['selected_parameter_shapes']['blocks.0.attn.qkv_proj.weight']['shape']==expected for r in rows),f'TP{tp} actual QKV shapes'))
    sizes=[load(f'tp{tp}_meta126/rank0.json')['per_rank_parameter_bytes'] for tp in (2,4,8)]
    passed.append(check(sizes==[33168334080,16584755584,8292966336],'real TP parameter bytes are sharded, not replicated'))
    extensions=load('runtime_extensions126.json')
    passed.append(check(extensions['completed'] and not extensions['weights_loaded'] and not extensions['NPU_dispatch_returns_operators'],'actual NPU VAE dispatcher has no operator set'))
    vae=[load(f'vae_observe_native126/rank{i}.json') for i in range(8)]
    for i,r in enumerate(vae):
        assert r['completed'] and r['rank']==i
        assert r['installer']['native_called'] and r['installer']['native_return'] is False
        assert r['installer']['all_float_parameters_fp32_after']
        assert r['parameter_contract']['all_float_parameters_fp32']
        assert r['latents_sha256']=='5ccba4c9293651bcc57756a0fb2acb39f4e5bd04439c5df6470901bc4c6759b3'
        for name in ('video','audio'):
            rows=r['cases'][name+'_clean'];assert len(rows)==4
            assert not any(c['instrumented_not_clean'] for c in rows)
    passed.append('eight-rank same-final-latent FP32 VAE baseline, native False observed')
    video_warm=[max(r['cases']['video_clean'][i]['wall_s'] for r in vae) for i in (1,2,3)]
    audio_warm=[max(r['cases']['audio_clean'][i]['wall_s'] for r in vae) for i in (1,2,3)]
    all_dit={}
    for name in ('resident_TP8_FA126b','resident_TP4_FA126','sync_native_TP2_FA126','event_profile_TP2_FA126','actual_FA_profile126','weight_pool_TP2_FA126','weight_pool_profile126'):
        rows=cohort(name);all_dit[name]=rows
        for r in rows:
            for c in r['checks']:
                s=c['safety'];assert s['matrix_checks']==155 and s['nonfinite_count']==s['boundary_count']==0
                assert s['checked_before_sampler_update'] and s['all_checks_preserved']
                assert c['rank_max_elapsed_s']+1e-4>=c['local_elapsed_s']
        passed.append(name+': actual completed forwards and all 155 safety counts')
    for name in ('sync_native_TP2_FA126','resident_TP4_FA126'):
        for r in all_dit[name]:
            proof=r['device_scale_leaf_proof'];assert len(proof)==104
            assert all(p['bit_identical'] and p['scale_identical'] and p['different_words']==0 for p in proof)
        passed.append(name+': all 8x104 actual wide-GEMM leaf outputs and scales equal')
    unit=load('device_range_unit126.json')
    passed.append(check(unit['verification_passed'] and unit['positive_finite_peak_words_tested']==1371 and all(r['bit_identical'] for r in unit['synthetic_gemm_leaves']),'device range-selector and synthetic GEMM unit proof'))
    for r in all_dit['actual_FA_profile126']:
        for p in r['warm_event_profiles']:
            assert p['all_50_main_attention_calls_hit_real_FA']
            assert len(p['actual_ACL_launches'])==50 and all(x['return_code']==0 for x in p['actual_ACL_launches'])
            assert p['regions']['FA.ACL_launch']['calls']==50
            assert all(s['q_shape_BHND']==[1,7,41792,128] and s['n_pad']==23 for s in p['FA_runtime_shapes'])
            assert p['actual_kernel_binaries'][0]['object_sha256']=='aec5f81addb2063d7be34731034916e1901e5ea42816202e648a2e85fbccc6c8'
    passed.append('actual FA kernel, 50 ACL launches per measured forward on every rank')
    for r in all_dit['weight_pool_profile126']:
        for case in (1,2,3):
            p=r['weight_pool_profile_'+str(case)]
            assert len(p['block_device_spans'])==50
            assert [x['block'] for x in p['block_device_spans']]==list(range(50))
            assert p['streams_independent']==(case!=1)
            if case==1:assert p['copy_ms_overlapping_block_spans']<=1.0
            assert -1e-4<=p['overlap_fraction']<=1.0001
            assert r['weight_pool_case_'+str(case)]['byte_proof']['all_held_storage_bytes_equal']
    passed.append('real serial-zero control, independent copy streams, full 50-block body spans, held pool byte proofs')
    cache=[load(f'VAE_operand_cache126c/rank{i}.json') for i in range(8)]
    for i,r in enumerate(cache):
        assert r['completed'] and not r['original_parameters_rewritten']
        assert r['globally_aligned_tile_proof_rounds']==28 and r['local_tile_count']==(28 if i<4 else 21)
        assert r['cache_after_trial']['original_parameters_unchanged'] and r['cache_after_trial']['original_all_parameters_still_FP32']
        proof=r['first_tile_all_Linear_leaf_proof'];assert proof['leaf_comparisons']==144
        assert proof['all_same_input_leaf_words_equal'] and proof['all_eligible_modules_executed'] and proof['all_leaf_outputs_finite']
        assert len(r['all_local_prequantized_tile_proofs'])==r['local_tile_count']
        assert all(p['cached_words_equal'] and p['native_self_repeat_words_equal'] and p['finite_all'] and p['max_abs_error']==0 for p in r['all_local_prequantized_tile_proofs'])
        assert all(c['complete_Uint8_words_equal'] for case in r['cases'] for c in case['checks'])
    passed.append('VAE cache: unchanged FP32 parameters, 1152 leaf comparisons, all 196 full prequantized tile pairs equal')
    for r in cache:
        archived=ROOT/'vae/executed_cache126c'
        assert hashlib.sha256((archived/'autocast_operand_cache126.py').read_bytes()).hexdigest()==r['operand_cache_sha256']
        assert hashlib.sha256((archived/'benchmark_operand_cache126.py').read_bytes()).hexdigest()==r['driver_sha256']
    passed.append('exact executed VAE cache sources retained under their recorded SHA')
    nested=load('topology_encoder/cache_nested_inference_unit126.json')
    assert nested['completed'] and nested['outer_inference_mode_stayed_enabled'] and nested['private_copies_track_version']
    assert nested['same_input_Linear_comparisons']==144 and nested['all_words_equal'] and nested['all_original_FP32_parameters_unchanged']
    assert hashlib.sha256((ROOT/'vae/autocast_operand_cache126.py').read_bytes()).hexdigest()==nested['cache_source_sha256']
    passed.append('actual NPU nested-inference metadata regression: outer mode preserved, private version counters and 144 synthetic word comparisons')
    physical_topology=load('topology_query126.json')
    assert physical_topology['exit_code']==0 and 'HCCS' in physical_topology['stdout'] and 'PHB' in physical_topology['stdout'] and 'SYS' in physical_topology['stdout']
    mesh_hashes=set()
    for name,expected in [('encoder_mesh0_TP4_126',[0,1,2,3]),('encoder_mesh1_TP4_126',[4,5,6,7])]:
        rows=[load(f'topology_encoder/{name}/rank{i}.json') for i in range(4)]
        for rank,r in enumerate(rows):
            assert r['completed'] and r['rank']==rank and r['actual_world_size']==r['actual_visible_device_count']==r['encoder_TP_size']==4
            assert r['physical_HCCS_island']==expected and not r['new_precision_policy_accepted']
            assert r['eight_rank_hidden_reference_used_only_for_metrics_not_input']
            for c in r['cases'][0]['checks']:
                assert c['different_words_vs_first_baseline']==0
                mesh_hashes.add(c['hidden_sha256'])
    assert len(mesh_hashes)==1
    passed.append('physical 2x4 HCCS topology, live TP4 island trials: repeated and cross-island outputs equal; not precision-promoted')
    encoder=cohort('encoder_geometry126',prefix='rank')
    assert all(len(r['same_input_geometry_leaf_proof'])==2 and all(p['bit_identical'] for p in r['same_input_geometry_leaf_proof']) for r in encoder)
    passed.append('current FP16 encoder geometry leaves: two real cached methods per rank, exact outputs')
    pending=[]
    for name in ('equivalent_safety_unit126.json','equivalent_safety_bench126_rank0.json','encoder_native_diagnostic126_rank0.json'):
        if not (ROOT/name).is_file():pending.append(name)
    if require_final:assert not pending,('Missing final execution receipts',pending)
    if not pending:
        unit=load('equivalent_safety_unit126.json')
        assert unit['verification_passed'] and unit['every_65536_Half_bit_pattern_tested_on_NPU']
        assert unit['results']['native_counts']==unit['results']['candidate_counts']==[2048,2]
        passed.append('all 65536 Half words: safety predicates exactly equivalent on the actual NPU')
        diagnostic=load('encoder_native_diagnostic126_rank0.json')
        assert diagnostic['completed'] and diagnostic['all_27_vision_and_50_text_modules_executed_each_pass']
        for c in diagnostic['cases']:
            assert all(v==1 for v in c['forward_call_counts'].values())
            assert sum(k.startswith('vision.block.') for k in c['forward_call_counts'])==27
            assert sum(k.startswith('text.layer.') for k in c['forward_call_counts'])==50
        passed.append('actual full current encoder graph: 27 vision + 3 DeepStack + 50 text, bounded raw captures')
    combined_path=ROOT/'NPU_generated126/generation_result126.json'
    combined_verified=False
    if combined_path.is_file():
        result=json.loads(combined_path.read_text())
        assert result['completed'] and result['actual_video_generated']
        assert result['own_NPU_encoder_used'] and not result['GPU_encoder_or_keyframe_latent_cache_used']
        assert not result['CPU_model_inference'] and result['no_BF16_roundtrip_fallback_or_clipping']
        counts=result['actual_DiT_counts_by_worker_pid']
        assert len(counts)==8 and set(counts.values())=={49}
        assert result['equivalent_strict_safety_all_8x49x155_checks_passed']
        assert not result['new_combined_configuration_quality_accepted']
        assert result['engine_kwargs']['tensor_parallel_size']==2 and result['engine_kwargs']['ulysses_degree']==4
        assert result['engine_kwargs']['text_encoder_tp_size']==8 and result['engine_kwargs']['enable_layerwise_offload']
        events=result['VAE_operand_cache_runtime_events']
        assert len(events)==8 and all(e['cache']['original_parameters_unchanged'] and e['cache']['original_all_parameters_still_FP32'] and e['cache']['cached_operand_call_count']>0 for e in events)
        audit=[]
        for path in (ROOT/'NPU_generated126').glob('worker_alignment_pid*126.jsonl'):
            audit.extend(json.loads(line) for line in path.read_text().splitlines())
        ends=[e for e in audit if e['event']=='dit_step_end']
        constructed=[e for e in audit if e['event']=='matched_transformer_constructed']
        assert len(constructed)==8
        for e in constructed:
            assert e['main_layers']==50 and e['source_sha256']=='5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b'
            assert e['parameter_dtype_counts']['torch.float16']*2+e['parameter_dtype_counts']['torch.float32']*4==33168334080
        assert len(ends)==8*49
        assert all(e['actual_Half_matrix_modules_this_step']==155 and e['video']['finite'] and e['audio']['finite'] and e['no_observed_Half_saturation_boundary'] for e in ends)
        for pid in counts:
            assert sorted(e['step_index'] for e in ends if str(e['pid'])==pid)==list(range(49))
        assert any(e['event']=='live_NPU_encoder_hidden_saved' for e in audit)
        assert any(e['event']=='denoise_inputs_ready' and e['sigma_points']==50 and e['own_NPU_conditioning'] for e in audit)
        assert all(e['model_arithmetic_modified'] is False and e['strict_stop_placement_preserved'] for e in audit if e['event']=='R126_equivalent_safety_enabled')
        process=json.loads((ROOT/'NPU_generated126/NPU_full_generation126b_process.json').read_text())
        assert process['exit_code']==0 and process['nested_inference_lifecycle_unit_passed']
        for remote,digest in process['source_sha256'].items():
            relative=Path(remote).relative_to('/tmp/h3_guard148')
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, ('Executed full-run source drift',remote)
        for filename,key in [('h3_910a_vllm_omni_fp16.mp4','video_sha256'),('h3_910a_vllm_omni_audio.wav','audio_sha256')]:
            media=ROOT/'NPU_generated126'/filename
            assert hashlib.sha256(media.read_bytes()).hexdigest()==result[key]
        combined_verified=True
        passed.append('real combined regression: live encoder, original 50 blocks x49, all 60760 strict checks, actual VAE operand hits, media SHA verified')
    if require_combined:assert combined_verified,'Combined 49-step regression evidence has not completed'
    return {'scope':__doc__,'evidence_verification_passed':True,'checks_passed':passed,'pending_execution_evidence':pending,
            'new_candidate_automatically_accepted':False,'full_49_step_combined_generation_verified126':combined_verified,
            'warm_baseline_video_decode_rank_max_median_s':statistics.median(video_warm),
            'warm_baseline_audio_decode_rank_max_median_s':statistics.median(audio_warm),
            'true_TP_bytes':sizes,'no_local_weight_or_tensor_transfer_required':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-final',action='store_true');p.add_argument('--require-combined',action='store_true');p.add_argument('--output')
    args=p.parse_args();result=verify(args.require_final,args.require_combined)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        target=Path(args.output).resolve();assert target.parent==ROOT
        if target.exists():raise FileExistsError('Choose a fresh verification output name')
        target.write_text(text)
    print(text,end='')

if __name__=='__main__':main()
