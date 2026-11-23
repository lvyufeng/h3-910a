#!/usr/bin/env python3
"""Complete R126 native dispatcher/residency source facts; do not load weights."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path

ROOT=Path('/tmp/h3_silu152')

def main():
    import torch
    import torch_npu
    modules={
        'vae_actual_dispatch':'vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch',
        'module_residency':'vllm_omni.diffusion.offloader.module_residency',
        'tensor_utils':'vllm_omni.diffusion.offloader.tensor_utils',
        'component_utils':'vllm_omni.diffusion.offloader.component_utils',
        'vae_temporal':'vllm_omni.diffusion.models.minimax_h3.vae_temporal',
        'chunked_decode':'vllm_omni.diffusion.models.minimax_h3.chunked_decode',
        'vae_executor':'vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor',
    }
    r={'scope':__doc__,'completed':False,'weights_loaded':False,'modules':{},'component_config':{}}
    imported={}
    for label,name in modules.items():
        module=importlib.import_module(name);path=Path(inspect.getfile(module)).resolve()
        assert path.suffix=='.py' and str(path).startswith('/home/mseco/')
        data=path.read_bytes();saved=ROOT/'native_sources'/f'{label}.py';assert not saved.exists()
        saved.write_bytes(data);imported[label]=module
        r['modules'][label]={'import_name':name,'path':str(path),'snapshot':str(saved),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    disp=imported['vae_actual_dispatch']
    result=disp.resolve_h3_vae_operators(torch.device('npu:0'))
    r['NPU_dispatch_returns_operators']=result is not None
    r['NPU_dispatch_result_type']=str(type(result))
    checkpoint=Path('/home/mseco/models/MiniMax-H3/FL2VA')
    for label in ['video_vae','audio_vae']:
        cfg=checkpoint/label/'config.json';data=cfg.read_bytes()
        r['component_config'][label]={'path':str(cfg),'sha256':hashlib.sha256(data).hexdigest(),'config':json.loads(data)}
        r['component_config'][label]['source_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint/label).iterdir() if p.is_file() and p.suffix=='.py'}
    r['completed']=True
    (ROOT/'runtime_extensions126.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
