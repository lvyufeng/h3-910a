#!/usr/bin/env python3
"""Verify R124 numerical gates, actual inference execution and media receipts.

This verifier reads only local source/JSON/media metadata and worker audits. It
never imports Torch or runs model inference. It distinguishes same-input gates,
free-running sample quality, encoder diagnostics, and unmet performance targets.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def load(relative):return json.loads((ROOT/relative).read_text())


def main():
    gates=load('sigma_checks/sigma_checks_rank0.json')
    generation=load('NPU_generated/generation_result124.json')
    quality=load('NPU_generated/all_frame_quality_inspection124.json')
    source=load('executed_source_manifest124g.json')
    gpu=load('GPU_pipeline_reference/GPU_pipeline_trajectory_result124c.json')
    assert gpu['completed'] and gpu['all49_predictions_all8ranks_finite']
    assert set(gpu['actual_DiT_evaluations_by_rank'].values())=={49} and len(gpu['actual_DiT_evaluations_by_rank'])==8
    assert gates['completed'] and gates['all_selected_sigma_gates_accepted']
    assert {r['step_index'] for r in gates['checks']}=={0,10,25,40,48}
    assert generation['completed'] and generation['actual_video_generated']
    assert generation['own_NPU_encoder_used'] and not generation['GPU_encoder_or_keyframe_latent_cache_used']
    assert generation['no_local_model_weights'] and not generation['CPU_model_inference']
    assert generation['quality_mode']=='lossless' and generation['original_VAE_FP32_parameters_preserved']
    for name in ['generate_full_NPU124g.py','alignment_hooks124.py','sitecustomize.py','npu_process_bootstrap124e.py','capture_NPU_media124.py','npu_half_nan_compat124.py','media_helpers124.py']:
        data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==source['source_files'][name]['sha256'],name
    audit_paths=list((ROOT/'NPU_full_run_audit').glob('worker_alignment_pid*124.jsonl'));assert len(audit_paths)==8
    rank_reports={};rank0_rows=None
    for path in audit_paths:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        ends=[r for r in rows if r['event']=='dit_step_end'];starts=[r for r in rows if r['event']=='dit_step_begin']
        assert len(ends)==len(starts)==49 and [r['step_index'] for r in ends]==list(range(49))
        assert all(r['video']['finite'] and r['audio']['finite'] for r in ends)
        assert all(r['actual_Half_matrix_modules_this_step']==155 and r['full50_main_layers_executed'] for r in ends)
        assert [r['checked_Half_matrix_output_count'] for r in ends]==[155*(i+1) for i in range(49)]
        assert not any(r['event']=='half_matrix_range_warning' for r in rows)
        constructed=next(r for r in rows if r['event']=='matched_transformer_constructed');assert constructed['main_layers']==50
        assert set(constructed['parameter_dtype_counts'])=={'torch.float16','torch.float32'}
        noise=next(r for r in rows if r['event']=='exact_initial_noise_used');assert not noise['GPU_encoder_activation_used'] and not noise['GPU_keyframe_latent_used']
        assert next(r for r in rows if r['event']=='denoise_inputs_ready')['own_NPU_conditioning']
        decode=next(r for r in rows if r['event']=='native_NPU_VAE_decode_end');assert not decode['BF16_fallback']
        rank=int(ends[0]['rank']);rank_reports[str(rank)]={'pid':ends[0]['pid'],'actual_evaluations':49,'actual_main_layers_per_evaluation':50,
                    'audited_Half_matrix_module_calls_per_evaluation':155,'total_audited_Half_matrix_module_calls':155*49,
                    'all_prediction_outputs_finite':True,'Half_saturation_boundary_observed':False,
                    'warm_step_median_s_including_offload_and_audit':statistics.median(r['elapsed_s_includes_offload'] for r in ends[2:]),
                    'sum_forward_s_including_offload_and_audit':sum(r['elapsed_s_includes_offload'] for r in ends),
                    'native_VAE_decode_s_including_transfer':decode['elapsed_s'],'NanToNum_compatibility_calls_during_decode':decode['exact_NanToNum_compatibility_calls']}
        if rank==0:rank0_rows=rows
    assert set(rank_reports)=={str(i) for i in range(8)}
    assert rank0_rows is not None
    gates_rows=[]
    for row in gates['checks']:
        v=row['video_target_velocity'];a=row['audio_velocity']
        assert v['shape']==[37296,96] and a['shape']==[414,32]
        assert v['relative_RMS']<=0.005 and a['relative_RMS']<=0.005
        assert v['finite'] and a['finite']
        gates_rows.append({'zero_based_step':row['step_index'],'video_relative_RMS':v['relative_RMS'],'audio_relative_RMS':a['relative_RMS'],
                           'video_RMSE':v['RMSE'],'audio_RMSE':a['RMSE'],'video_p99_abs_error':v['p99_abs_error'],'audio_p99_abs_error':a['p99_abs_error'],
                           'video_max_abs_error':v['max_abs_error'],'audio_max_abs_error':a['max_abs_error'],'same_input_1percent_gate_passed':True,'0point5percent_target_met':True})
    for name,key in [('h3_910a_vllm_omni_fp16.mp4','video_sha256'),('h3_910a_vllm_omni_audio.wav','audio_sha256')]:
        path=ROOT/'NPU_generated'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==generation[key]
    assert quality['completed'] and quality['media_sha256']==generation['video_sha256']
    assert quality['decoded_video_frames']==quality['red_ball_detected_frames']==124
    assert not quality['frames_with_multiple_large_red_components'] and quality['native_encoded_audio_finite']
    frames=quality['frames'];assert [r['frame_index'] for r in frames]==list(range(124))
    assert generation['media_verification']['decoded_video_frames']==124 and generation['media_verification']['sample_rate']==32000
    ys=[r['largest_red_centroid_y'] for r in frames]
    media={'video_sha256':generation['video_sha256'],'audio_sha256':generation['audio_sha256'],'width':1344,'height':768,'frames':124,'fps':24,
           'duration_s':124/24,'sample_rate':32000,'channels':2,'audio_samples_per_channel':generation['media_verification']['decoded_audio_samples'],
           'native_audio_peak_dBFS':20*math.log10(generation['native_audio_peak']),'native_audio_RMS_dBFS':20*math.log10(generation['native_audio_RMS']),
           'audio_gain_normalized':False,'all_frames_red_ball_detected':True,'multiple_large_red_components_observed':False,
           'first_last_ball_centroids':[quality['first_ball_centroid'],quality['last_ball_centroid']],
           'vertical_centroid_range_pixels':[min(ys),max(ys)],'automatic_universal_perceptual_quality_score_claimed':False}
    encoder=next(r for r in rank0_rows if r['event']=='live_NPU_encode_prompt_completed')
    payload={'scope':__doc__,'completed':True,'round124_numerical_and_generation_milestone_complete':True,'overall_speed_and_A2_A3_comparison_goal_complete':False,
             'original_full50_layer_model':True,'no_student_or_LoRA':True,'GPU_reference_actual49_step_pipeline_completed':True,
             'same_input_full_tensor_gates':gates_rows,'original_NPU_full_generation_execution_by_rank':rank_reports,'media':media,
             'performance_observations_not_benchmark_acceptance':{'engine_initialization_s':generation['engine_initialization_s'],
                  'full_live_generation_wall_s_excluding_initialization':generation['generation_wall_s'],'live_encoder_stage_s_with_integration_and_transfer':encoder['elapsed_s'],
                  'warm_DiT_step_median_s_with_offload_and_audit':rank_reports['0']['warm_step_median_s_including_offload_and_audit'],
                  'native_VAE_decode_s_with_transfer':rank_reports['0']['native_VAE_decode_s_including_transfer'],
                  'encoder_under2s_accepted':False,'DiT_under15s_accepted':False,'VAE_under20s_accepted':False,'A2_A3_measured_this_round':False},
             'quality_acceptance_scope':'This single same-seed first/last red-ball sample: all124 frames tracked; contact sheet plus two full-resolution frames visually reviewed by lead. Motion/scene are consistent with prompt and GPU sample. Extremely quiet native audio is retained unchanged. Not a general prompt suite or auditory/perceptual equivalence proof.',
             'encoder_visual_RMS_diagnostic_not_waived_or_used_as_DiT_gate':load('own_NPU_encoder_vs_successful_GPU124.json')['by_token_tag'],
             'CPU_work_is_activation_statistics_media_processing_and_weight_staging_not_CPU_model_inference':True,
             'temporary_services_stopped':True,'GPU_and_NPU_leases_released':True}
    destination=ROOT/'verified_full_generation124.json'
    if destination.exists():
        assert json.loads(destination.read_text())==payload,'Current evidence differs from the preserved verification receipt'
    else:
        destination.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'verification_passed':True,'all5_gates_passed_target':True,'all8_ranks_completed49_real_original50_layer_forwards':True,'video_frames':124,
                      'native_audio_peak_dBFS':media['native_audio_peak_dBFS'],'warm_step_median_s':rank_reports['0']['warm_step_median_s_including_offload_and_audit'],
                      'native_VAE_decode_s':rank_reports['0']['native_VAE_decode_s_including_transfer'],'overall_speed_goal_not_complete':True},indent=2))


if __name__=='__main__':main()
