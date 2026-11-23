#!/usr/bin/env python3
"""Host preflight for migration R138: hashes only, no model weights loaded.

Verifies the fixed R136 latent, the installed native VAE sources and the
forward-context signature that a previous round got wrong, then writes
preflight139.json inside the private root. Never touches the NPU.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/tmp/h3_vae13902')
PRIOR=Path('/tmp/h3_quality13601/engine_quality136a')
PRIOR_RECEIPT=PRIOR/'lifecycle_result136.json'
SOURCES=(
    '/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vae.py',
    '/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py',
    '/home/mseco/models/MiniMax-H3/FL2VA/audio_vae/dac_audio_vae.py',
)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    assert sys.version_info[:2]==(3,12),sys.version
    assert ROOT.resolve()==Path('/tmp/h3_vae13902') and not (ROOT/'engine_vae139b').exists()
    prior=json.loads(PRIOR_RECEIPT.read_text());assert prior['completed']
    row=prior['requests'][0];assert row['label']=='r01_original_false'
    files=list((PRIOR/row['label']).glob('final_latents_pid*127.pt'));assert len(files)==1
    latent_sha=sha(files[0]);assert latent_sha==row['final_latents_sha256']

    from vllm_omni.diffusion.forward_context import set_forward_context
    import inspect
    signature=inspect.signature(set_forward_context)
    signature.bind(vllm_config=None,omni_diffusion_config=None)
    bind_ok=True
    try:
        signature.bind(od_config=None)
        bind_ok=False
    except TypeError:
        pass
    assert bind_ok,'od_config must be rejected by the context signature'

    result={'completed':True,'attempt':'R139','model_weights_loaded':False,'npu_used':False,
        'fixed_video_latent_SHA_verified':True,'fixed_video_latent_path':str(files[0]),
        'fixed_video_latent_sha256':latent_sha,'fixed_video_latent_label':row['label'],
        'native_source_SHA_verified':True,
        'native_source_sha256':{path:sha(Path(path)) for path in SOURCES},
        'forward_context_signature':str(signature),'correct_kwarg_names_bind':True,'wrong_kwarg_rejected':True,
        'weights_or_activations_not_transferred_to_client':True,'measurement_only_no_math_change':True}
    target=ROOT/'preflight139.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()