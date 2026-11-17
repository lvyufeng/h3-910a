#!/usr/bin/env python3
"""Validate actual generated media and preserve failed visual/independent-hidden gates."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
MANUAL_INDICES=[0,12,24,36,48,62,74,86,98,110,123]


def summarize(root=ROOT):
    receipt=json.loads((root/'generation_result.json').read_text());scan=json.loads((root/'video_scan.json').read_text());verified=json.loads((root/'final_verification.json').read_text())
    if not receipt['completed']or not scan['completed']or not verified['all_match']or not verified['original_generation_processes_gone']:raise ValueError('Incomplete actual generation/decoder/source verification')
    if scan['source_sha256']!=hashlib.sha256((root/'scan_video76.py').read_bytes()).hexdigest():raise ValueError('Executed full-frame scan source changed')
    videos=[]
    for case,row in zip(receipt['cases'],scan['videos']):
        label=case['case'];video=root/(label+'.mp4');sheet=root/(label+'_contactsheet.png')
        if hashlib.sha256(video.read_bytes()).hexdigest()!=case['video']['sha256']or row['mp4_sha256']!=case['video']['sha256']:raise ValueError('Actual local video or remote decoder provenance changed')
        proof=verified['contact_sheets'][sheet.name]
        if sheet.stat().st_size!=proof['bytes']or hashlib.sha256(sheet.read_bytes()).hexdigest()!=proof['sha256']:raise ValueError('Delivered contact sheet changed')
        if len(row['frames'])!=124 or [f['index']for f in row['frames']]!=list(range(124)):raise ValueError('Missing actual decoded frame')
        xs=[f['red_centroid_x']for f in row['frames']];ys=[f['red_centroid_y']for f in row['frames']]
        if any(x is None for x in xs):raise ValueError('Red-color diagnostic has absent observations')
        diffs=[f['consecutive_frame_MAE']for f in row['frames'][1:]]
        videos.append({'case':label,'MP4_sha256':case['video']['sha256'],'contactsheet_sha256':proof['sha256'],
                       'full124_frames_decoded':True,'manual_review_time_points':MANUAL_INDICES,
                       'width':row['stream']['width'],'height':row['stream']['height'],'fps':row['stream']['average_rate'],'duration_s':row['stream']['duration_s'],
                       'video_only_export_audio_tracks':row['stream']['audio_tracks'],
                       'original_RGB_endpoint_reconstruction_only':row['original_exported_RGB_endpoint_metrics_before_MP4_codec'],
                       'red_colour_centroid_diagnostic':{'first_x':xs[0],'last_x':xs[-1],'min_x':min(xs),'min_x_frame':xs.index(min(xs)),
                                                         'max_x':max(xs),'max_x_frame':xs.index(max(xs)),'min_y':min(ys),'max_y':max(ys)},
                       'frame_MAE_diagnostic_not_temporal_quality_score':{'mean':statistics.mean(diffs),'max':max(diffs),'max_transition_to_frame':diffs.index(max(diffs))+1},
                       'visual_review':{'unexpected_middle_logo_and_lettering':True,'unexpected_middle_radial_glow':True,'visible_ball_ghosting_colour_fringing':True,
                                        'first_last_red_ball_anchors_approximately_preserved':True,'prompt_left_to_right_motion_has_initial_leftward_and_overshoot':True,
                                        'quality_pass':False}})
    opt,hf=scan['videos'];xerrors=[abs(a['red_centroid_x']-b['red_centroid_x'])for a,b in zip(opt['frames'],hf['frames'])]
    return {'scope':'Actual two49stepFL2VA1344x768/5s videos. Quantitative full-frame/endpoint/color metrics are diagnostics,not perceptual quality scores. Visual review covers11 sampled time points per video+originaloptimizedPNG0/12/24.',
            'videos':videos,'different_raw_RGB_outputs':receipt['cases'][0]['raw_RGB_frame_sha256']!=receipt['cases'][1]['raw_RGB_frame_sha256'],
            'different_conditions_red_centroid_diagnostic':{'mean_abs_x_difference_pixels':statistics.mean(xerrors),'max_abs_x_difference_pixels':max(xerrors),'max_difference_frame':xerrors.index(max(xerrors))},
            'HF_encoder_conditioning_alone_did_not_remove_observed_visual_artifacts':True,'sole_cause_established':False,
            'independent_HF_hidden_quality_gate_still_unpassed_relative_RMS':receipt['FP16_consumed_hidden_comparison']['relative_rms'],
            'native_BF16_compute_used':False,'full_live_encoder_E2E_measured':False,'audio_waveform_exported_or_reviewed':False,'official_lossless_accepted':False,'official_goal_complete':False,
            'limits':'Red segmentation is not verified object tracking. Endpoint PSNR is reconstruction only;originalPNG endpoints are separate from lossyMP4 pixels. No OCR/CLIP/optical-flow score;manual review did not cover every single frame. Both conditions share sameFP16DiT/VAE so HF control is not an independentDiT or originalBF16generated-video oracle.'}


if __name__=='__main__':
    r=summarize();(ROOT/'quality_summary.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'quality_pass':[c['visual_review']['quality_pass']for c in r['videos']],
                      'HF_conditioning_alone_removes_artifacts':not r['HF_encoder_conditioning_alone_did_not_remove_observed_visual_artifacts'],
                      'video_diagnostic':r['different_conditions_red_centroid_diagnostic'],'sole_cause_established':r['sole_cause_established']},indent=2))
