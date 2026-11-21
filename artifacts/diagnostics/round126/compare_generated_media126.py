#!/usr/bin/env python3
"""Compare actual generated media without pretending live trajectories are paired tensors.

Only small encoded video and native Float WAV are read. This is not model
inference, an operator losslessness gate, a quality score, or acceptance of a
numerical policy. No audio gain, alignment search, denoising or retouching.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024**2),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();baseline=Path(args.baseline).resolve();candidate=Path(args.candidate).resolve();out=Path(args.output).resolve()
    assert baseline==Path('/tmp/h3_align12401/NPU_full_generation124j')
    assert candidate in (Path('/tmp/h3_perf12601/NPU_full_generation126'),Path('/tmp/h3_perf12601/NPU_full_generation126b'))
    assert out==Path('/tmp/h3_perf12601/R125_FA_vs_R126_media.json') and not out.exists()
    import numpy as np
    import av
    import soundfile as sf
    base_json=json.loads((baseline/'generation_result124.json').read_text())
    cand_json=json.loads((candidate/'generation_result126.json').read_text())
    assert base_json['completed'] and cand_json['completed'] and cand_json['actual_video_generated']
    video_name='h3_910a_vllm_omni_fp16.mp4';audio_name='h3_910a_vllm_omni_audio.wav'
    for root,record in [(baseline,base_json),(candidate,cand_json)]:
        assert sha(root/video_name)==record['video_sha256']
        assert sha(root/audio_name)==record['audio_sha256']
    frames=[];total_square=total_abs=elements=0
    with av.open(str(baseline/video_name)) as a,av.open(str(candidate/video_name)) as b:
        assert a.streams.video[0].average_rate==b.streams.video[0].average_rate
        for index,(fa,fb) in enumerate(itertools.zip_longest(a.decode(video=0),b.decode(video=0))):
            assert fa is not None and fb is not None,'Generated videos have different frame counts'
            x=fa.to_ndarray(format='rgb24');y=fb.to_ndarray(format='rgb24')
            assert x.shape==y.shape==(768,1344,3)
            delta=x.astype(np.float64)-y.astype(np.float64)
            square=float(np.square(delta).sum());absolute=float(np.abs(delta).sum());n=x.size
            mse=square/n;total_square+=square;total_abs+=absolute;elements+=n
            frames.append({'index':index,'MSE_encoded_pixels':mse,'PSNR_dB':None if mse==0 else 10*math.log10(255**2/mse),
                           'mean_abs_error_0_255':absolute/n,'max_abs_error_0_255':float(np.abs(delta).max())})
    assert len(frames)==124
    x_raw,rate_a=sf.read(str(baseline/audio_name),dtype='float32',always_2d=True)
    y_raw,rate_b=sf.read(str(candidate/audio_name),dtype='float32',always_2d=True)
    assert rate_a==rate_b==32000 and x_raw.shape==y_raw.shape==(165600,2)
    raw_audio_bits_equal=bool(np.array_equal(x_raw.view(np.uint32),y_raw.view(np.uint32)))
    x=x_raw.astype(np.float64);y=y_raw.astype(np.float64)
    delta=y-x;norm=np.linalg.norm(x);rms_x=float(np.sqrt(np.mean(x*x)));rms_y=float(np.sqrt(np.mean(y*y)))
    nonzero_psnr=[f['PSNR_dB'] for f in frames if f['PSNR_dB'] is not None]
    result={'scope':__doc__,'completed':True,'baseline':str(baseline),'candidate':str(candidate),
      'baseline_video_sha256':base_json['video_sha256'],'candidate_video_sha256':cand_json['video_sha256'],
      'media_frame_count':124,'video':{'per_frame':frames,'encoded_pixel_PSNR_min_dB':min(nonzero_psnr) if nonzero_psnr else None,
        'encoded_pixel_PSNR_median_dB':float(np.median(nonzero_psnr)) if nonzero_psnr else None,
        'encoded_pixel_overall_PSNR_dB':None if total_square==0 else 10*math.log10(255**2/(total_square/elements)),
        'encoded_pixel_mean_absolute_error_0_255':total_abs/elements},
      'audio':{'samples_per_channel':165600,'channels':2,'sample_rate':32000,
        'raw_sample_bits_equal':raw_audio_bits_equal,'relative_RMS_error':float(np.linalg.norm(delta)/max(norm,1e-30)),
        'absolute_RMS_error':float(np.sqrt(np.mean(delta*delta))),'max_absolute_error':float(np.abs(delta).max()),
        'reference_RMS':rms_x,'candidate_RMS':rms_y,'reference_RMS_dBFS':20*math.log10(max(rms_x,1e-30)),
        'candidate_RMS_dBFS':20*math.log10(max(rms_y,1e-30)),
        'correlation':float(np.corrcoef(x.reshape(-1),y.reshape(-1))[0,1])},
      'live_encoder_recomputed_each_run':True,'same_initial_noise_does_not_make_live_hidden_conditioning_equal':True,
      'not_an_isolated_causal_operator_comparison':True,'quality_automatically_accepted':False,
      'no_audio_gain_or_pixel_modifications':True,'model_weights_read_or_transferred':False}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('video',)},indent=2),flush=True)
    print(json.dumps({'video_summary':{k:v for k,v in result['video'].items() if k!='per_frame'}},indent=2),flush=True)

if __name__=='__main__':main()
