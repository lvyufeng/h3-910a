#!/usr/bin/env python3
"""Validate received scalar quality metrics without model data or adoption policy."""
import argparse
import json
import math
from pathlib import Path

SOURCE=Path(__file__).resolve().parent


def verify_float(metric):
    assert metric['word_bits'] in (16,32) and metric['total_words']>0
    assert 0<=metric['different_words']<=metric['total_words']
    assert metric['all_words_equal']==(metric['different_words']==0)
    assert metric['complete_logical_word_range_compared'] and metric['finite_inputs']
    assert metric['no_alignment_gain_normalization_or_clipping']
    assert metric['relative_RMS_percent']==100*metric['relative_RMS_ratio']
    assert metric['relative_RMS_ratio']>=0 and math.isfinite(metric['relative_RMS_ratio'])
    assert metric['max_abs_difference']>=0 and metric['root_mean_square_error']>=0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quality',type=Path,required=True);args=parser.parse_args()
    report=json.loads(args.quality.read_text());assert report['completed']
    assert report['model_process_exit_code']==0 and report['model_private_children_exited_before_postprocessing']
    assert report['generation_requests']==3 and report['strict_checks_allranks']==182280
    assert not report['numeric_baseline_adopted'] and not report['new_numeric_tolerance'] and not report['quality_pass_or_acceptance_decided']
    assert not report['full49_candidate_repeatability_tested'] and report['global_deterministic_flag_can_affect_encoder_and_VAE_too']
    assert not report['arrays_normalized_resampled_aligned_clipped_or_gain_fitted'] and not report['encoded_media_PSNR_measured']
    assert [p['pair_id'] for p in report['comparisons']]==['candidate_vs_original','restored_vs_original','candidate_vs_restored']
    summary=[]
    for pair in report['comparisons']:
        assert pair['comparison_direction']=='actual_minus_reference' and pair['original_restored_noise_is_not_a_candidate_tolerance']
        for key in ('own_live_encoder_conditioning','video_latent','audio_latent','native_decoded_audio'):verify_float(pair[key])
        assert pair['own_live_encoder_conditioning']['word_bits']==16
        audio=pair['native_decoded_audio'];assert audio['shape']==[2,165600] and audio['word_bits']==32 and audio['total_words']==331200
        for channel in audio['channels']:verify_float(channel)
        video=pair['native_decoded_video'];assert video['shape']==[124,768,1344,3] and video['dtype']=='uint8'
        assert video['total_channel_values']==124*768*1344*3 and video['total_RGB_pixels']==124*768*1344
        assert len(video['per_frame'])==124 and [p['frame_index'] for p in video['per_frame']]==list(range(124))
        assert sum(p['different_channel_values'] for p in video['per_frame'])==video['different_channel_values']
        assert sum(p['different_RGB_pixels'] for p in video['per_frame'])==video['different_RGB_pixels']
        assert video['all_channel_values_equal']==(video['different_channel_values']==0)
        assert video['complete_logical_channel_and_RGB_pixel_range_compared'] and video['no_alignment_gain_normalization_or_clipping']
        assert 0<=video['max_abs_difference']<=255 and video['data_range_for_PSNR_only']==255
        if video['psnr_perfect']:assert video['psnr_db'] is None
        else:assert math.isfinite(video['psnr_db'])
        summary.append({'pair_id':pair['pair_id'],'actual_label':pair['actual_request_label'],'reference_label':pair['reference_request_label'],
            'own_live_encoder_relative_RMS_percent':pair['own_live_encoder_conditioning']['relative_RMS_percent'],
            'video_latent_relative_RMS_percent':pair['video_latent']['relative_RMS_percent'],
            'audio_latent_relative_RMS_percent':pair['audio_latent']['relative_RMS_percent'],
            'native_decoded_RGB_PSNR_dB':video['psnr_db'],'native_decoded_RGB_MAE_0_255':video['mean_absolute_error'],
            'native_decoded_RGB_max_abs_0_255':video['max_abs_difference'],'native_decoded_RGB_changed_pixels':video['different_RGB_pixels'],
            'native_decoded_audio_relative_RMS_percent':audio['relative_RMS_percent'],'native_decoded_audio_max_abs_difference':audio['max_abs_difference'],
            'native_decoded_audio_SNR_dB':audio['snr_db'],'native_decoded_audio_correlation':audio['pearson_correlation']})
    result={'quality_scalar_contracts_verified':True,'comparisons':summary,'not_an_adoption_or_model_tolerance_decision':True,
        'original_restored_noise_not_candidate_allowance':True,'only_one_candidate_complete49_request':True,
        'combined_global_flag_and_final_AdaLN_test_not_isolated_operator_causality':True}
    target=SOURCE/'verified_quality136.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
