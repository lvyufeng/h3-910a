#!/usr/bin/env python3
"""Native repeated-request media comparison, not a candidate losslessness gate.

Runs after the existing model process finishes. No tensor/model weight reads,
normalization, timing alignment search, audio gain, denoising or pixel edits.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

SESSION=Path('/tmp/h3_runtime12701/engine_lifecycle127c')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference,candidate):
    import av
    import numpy as np
    import soundfile as sf
    squares=absolute=elements=0;perframe=[]
    with av.open(str(reference/'h3_910a_original127.mp4')) as a,av.open(str(candidate/'h3_910a_original127.mp4')) as b:
        assert a.streams.video[0].average_rate==b.streams.video[0].average_rate
        for index,(fa,fb) in enumerate(itertools.zip_longest(a.decode(video=0),b.decode(video=0))):
            assert fa is not None and fb is not None
            x=fa.to_ndarray(format='rgb24');y=fb.to_ndarray(format='rgb24');assert x.shape==y.shape==(768,1344,3)
            delta=x.astype(np.float64)-y.astype(np.float64)
            value=float(np.square(delta).sum());mse=value/x.size;squares+=value;elements+=x.size
            absolute+=float(np.abs(delta).sum());perframe.append(None if mse==0 else 10*math.log10(255**2/mse))
    assert len(perframe)==124
    x,ra=sf.read(str(reference/'h3_910a_native127.wav'),dtype='float32',always_2d=True)
    y,rb=sf.read(str(candidate/'h3_910a_native127.wav'),dtype='float32',always_2d=True)
    assert ra==rb==32000 and x.shape==y.shape==(165600,2)
    equal=bool(np.array_equal(x.view(np.uint32),y.view(np.uint32)))
    a=x.astype(np.float64);b=y.astype(np.float64);delta=b-a
    psnr=[v for v in perframe if v is not None]
    return {'reference':reference.name,'candidate':candidate.name,'frames':124,
        'video_encoded_PSNR_min_dB':min(psnr) if psnr else None,
        'video_encoded_PSNR_median_dB':float(np.median(psnr)) if psnr else None,
        'video_encoded_overall_PSNR_dB':10*math.log10(255**2/(squares/elements)) if squares else None,
        'video_encoded_MAE_0_255':absolute/elements,'video_encoded_pixels_all_equal':squares==0,
        'audio_raw_bits_equal':equal,'audio_relative_RMS_error':float(np.linalg.norm(delta)/max(np.linalg.norm(a),1e-30)),
        'audio_absolute_RMS_error':float(np.sqrt(np.mean(delta*delta))),'audio_max_abs_error':float(np.abs(delta).max()),
        'audio_reference_RMS':float(np.sqrt(np.mean(a*a))),'audio_candidate_RMS':float(np.sqrt(np.mean(b*b))),
        'audio_correlation':float(np.corrcoef(a.reshape(-1),b.reshape(-1))[0,1]),
        'not_causal_candidate_operator_pair':True,'native_repeat_noise_is_not_a_candidate_tolerance':True}


def main():
    result=json.loads((SESSION/'lifecycle_result127.json').read_text());assert result['completed']
    for row in result['requests']:
        path=SESSION/row['label']
        assert sha(path/'h3_910a_original127.mp4')==row['video_sha256']
        assert sha(path/'h3_910a_native127.wav')==row['audio_sha256']
    report={'scope':__doc__,'completed':True,'same_engine_live_encoder_recomputed':True,
        'model_weights_read_or_transferred':False,'quality_automatically_accepted':False,
        'pairs':[compare(SESSION/'r01_cold',SESSION/'r02_warm'),
                 compare(SESSION/'r02_warm',SESSION/'r03_native_reload')]}
    out=SESSION.parent/'native_repeat_media127.json';assert not out.exists()
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
