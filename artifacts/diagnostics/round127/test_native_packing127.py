#!/usr/bin/env python3
"""Small CPU-only native weight-packing contract probe, no model inference or weights."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/tmp/h3_runtime12701')


def main():
    import torch
    import torch_npu
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    native=LayerwiseOffloadHook._to_cpu
    source=Path(__import__('vllm_omni.diffusion.offloader.layerwise_backend',fromlist=['x']).__file__)
    os.environ['H3_R127_SESSION_OUT']=str(ROOT/'engine_lifecycle127_unit')
    session=Path(os.environ['H3_R127_SESSION_OUT']);session.mkdir(exist_ok=False)
    from init_diagnostics127 import install
    install()
    wrapped=LayerwiseOffloadHook._to_cpu
    words=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(256,256)
    shapes=[('contiguous',words),('transpose',words.t()),('offset',words[3:249,2:254])]
    checks=[]
    for label,value in shapes:
        for method,func in [('native',native),('observed',wrapped)]:
            base=value.contiguous() if label=='contiguous' else value
            target=torch.nn.Parameter(base.detach(),requires_grad=False)
            original=target.detach().contiguous().clone()
            pool,metadata=func({'test_weight':target},{},True)
            meta=metadata[torch.float16][0]
            flat=pool[torch.float16][meta['offset']:meta['offset']+meta['numel']]
            reconstructed=torch.as_strided(flat,size=meta['shape'],stride=meta['stride'])
            equal=torch.equal(reconstructed.contiguous().view(torch.uint8),original.view(torch.uint8))
            assert equal and target.numel()==0 and pool[torch.float16].is_pinned()
            checks.append({'label':label,'method':method,'all_words_equal':equal,'stored_stride':list(meta['stride']),
                           'source_shape':list(original.shape),'target_offload_placeholder_numel':target.numel()})
    result={'scope':__doc__,'completed':True,'model_weights_read_or_downloaded':False,'device_model_inference':False,
        'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'diagnostic_source_sha256':hashlib.sha256((ROOT/'init_diagnostics127.py').read_bytes()).hexdigest(),
        'checks':checks,'all_native_and_observed_synthetic_words_equal':True,
        'does_not_establish_native_segfault_root_cause':True}
    path=ROOT/'init_packing_unit127.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
