"""Persist native H3 media before driver-side muxing without changing model output.

The post-process callable stays at module scope so multiprocessing can pickle it.
Captures only generated frames/audio (or native pre-encoded MP4 bytes), never
model parameters or weights. This lets a codec/UI/export issue be repaired
without re-running49 original DiT evaluations. No audio gain or pixel retouching.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

_NATIVE_POST_PROCESS=None


def _digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def capture_NPU_post_process126(output,output_type='np'):
    native=_NATIVE_POST_PROCESS
    if native is None:
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline
        native=pipeline._minimax_h3_post_process
        assert native is not capture_NPU_post_process126
    result=native(output,output_type=output_type)
    root=Path(os.environ['H3_NPU_RAW_OUTPUT_DIR124'])
    assert root.resolve().parent==Path('/tmp/h3_guard148') and root.name.startswith('NPU_full_generation126')
    assert isinstance(result,dict) and output_type=='np'
    videos=result['video'];assert isinstance(videos,list) and len(videos)==1
    if isinstance(videos[0],(bytes,bytearray,memoryview)):
        path=root/'native_NPU_encoded124.mp4'
        with path.open('xb') as f:f.write(bytes(videos[0]))
        metadata={'native_preencoded_output':True,'file':str(path),'bytes':path.stat().st_size,'sha256':_digest(path)}
    else:
        import numpy as np
        frames=np.asarray(videos[0]);audio=np.asarray(result['audio'])
        assert frames.dtype==np.uint8 and frames.ndim==4 and frames.shape[-1]==3
        assert audio.dtype==np.float32 and np.isfinite(audio).all()
        path=root/'native_NPU_frames_audio124.npz'
        with path.open('xb') as f:np.savez(f,frames=frames,audio=audio,sample_rate=np.array(result['audio_sample_rate']),fps=np.array(result['fps']))
        metadata={'native_preencoded_output':False,'file':str(path),'bytes':path.stat().st_size,'sha256':_digest(path),
                  'frame_shape':list(frames.shape),'audio_shape':list(audio.shape),'sample_rate':int(result['audio_sample_rate']),'fps':int(result['fps'])}
    metadata.update({'completed':True,'contains_model_weights':False,'no_audio_gain_or_pixel_editing':True})
    (root/'native_NPU_media_capture124.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return result


def install_capture(pipeline):
    global _NATIVE_POST_PROCESS
    if pipeline._minimax_h3_post_process is capture_NPU_post_process126:return
    _NATIVE_POST_PROCESS=pipeline._minimax_h3_post_process
    pipeline._minimax_h3_post_process=capture_NPU_post_process126
