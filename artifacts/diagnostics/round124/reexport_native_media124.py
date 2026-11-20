#!/usr/bin/env python3
"""Re-export preserved native generated frames/audio without any model replay.

This handles driver-side mux/export failures after actual49-step generation.
Only a completed native-media receipt is accepted. No model weights are loaded,
no latent/encoder cache is substituted, and no pixel editing/audio gain is used.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);assert root.resolve().parent==Path('/tmp/h3_align12401')
    import numpy as np
    import soundfile as sf
    from media_helpers124 import export_av,validate_av
    capture=json.loads((root/'native_NPU_media_capture124.json').read_text())
    assert capture['completed'] and not capture['contains_model_weights'] and not capture['native_preencoded_output']
    source=Path(capture['file']);assert source.resolve().parent==root.resolve() and digest(source)==capture['sha256']
    with np.load(source,allow_pickle=False) as data:
        frames=data['frames'];audio=data['audio'];sample_rate=int(data['sample_rate']);fps=int(data['fps'])
        if audio.ndim==3 and audio.shape[0]==1:audio=audio[0]
        assert frames.shape==(124,768,1344,3) and frames.dtype==np.uint8
        assert audio.ndim==2 and audio.shape[0]==2 and audio.dtype==np.float32 and np.isfinite(audio).all()
        assert sample_rate==32000 and fps==24
        video=root/'h3_910a_vllm_omni_fp16_reexport.mp4';wave=root/'h3_910a_native_audio_reexport.wav'
        assert not video.exists() and not wave.exists()
        export_av(frames,audio,video);sf.write(str(wave),audio.T,sample_rate,subtype='FLOAT')
    receipt={'scope':__doc__,'completed':True,'no_model_replay':True,'model_weights_loaded':False,'CPU_model_inference':False,
             'video_file':str(video),'audio_file':str(wave),'video_sha256':digest(video),'audio_sha256':digest(wave),
             'media_verification':validate_av(video),'audio_peak':float(np.abs(audio).max()),'audio_RMS':float(np.sqrt(np.mean(audio.astype(np.float64)**2)))}
    path=root/'native_media_reexport124.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
