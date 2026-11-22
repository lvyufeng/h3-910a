"""Authorized full50 same-input diagnostic: native->deterministic->restored only.

No sampler/generation or copy candidate. Process-wide deterministic flag changes
are diagnostic, not adopted precision policy. Original FP32 exceptions/155checks
and actual TP2+USP4 remain. All tensor snapshots/weights stay on the owned host.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from fingerprints129 import fingerprint,normalized_fingerprint,compare_outputs

ROOT=Path('/tmp/h3_audio13201')
OLD=Path('/tmp/h3_overlap12801')


def clone(value,torch,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:clone(v,torch,device) for k,v in value.items()}
    if isinstance(value,list):return [clone(v,torch,device) for v in value]
    if isinstance(value,tuple):return tuple(clone(v,torch,device) for v in value)
    return value


def model_master_state(model,torch):
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    rows={};offloaded=[]
    for name,param in model.named_parameters():
        local=param.to_local() if hasattr(param,'to_local') else param
        if local.numel():rows[name]=fingerprint(local,torch,include_pointers=False)
        else:offloaded.append(name)
    for name,buffer in model.named_buffers():rows['buffer:'+name]=fingerprint(buffer,torch,include_pointers=False)
    masters=[]
    for index,block in enumerate(model.blocks):
        hook=block._hook_registry._hooks[LayerwiseOffloadHook._HOOK_NAME]
        masters.append({'owner_block':index,'next_block':(index+1)%50,'cpu_master':{
            str(dtype):fingerprint(t,torch,include_pointers=True) for dtype,t in hook.dtype_cpu_flattened_weights.items()},
            'metadata_sha':hashlib.sha256(json.dumps({str(k):v for k,v in hook.dtype_metadata.items()},sort_keys=True,default=str).encode()).hexdigest(),
            'copy_stream':int(hook.copy_stream.npu_stream)})
    return {'materialized_parameters_and_buffers':rows,'empty_parameter_names':offloaded,'native_masters':masters}


def runtime_state(torch):
    return {'pid':os.getpid(),'native_thread_id':threading.get_native_id(),'device':torch.npu.current_device(),
        'current_stream':int(torch.npu.current_stream().npu_stream),
        'deterministic':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'HF32_matmul':torch.npu.matmul.allow_hf32,'HF32_conv':torch.npu.conv.allow_hf32}


def compare_time(actual,reference,torch):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype==torch.float32
    changed=int((actual.reshape(-1).view(torch.int32)!=reference.reshape(-1).view(torch.int32)).sum())
    delta=actual.double()-reference.double();den=float(reference.double().square().sum().sqrt().clamp_min(1e-30))
    return {'shape':list(actual.shape),'dtype':str(actual.dtype),'different_FP32_words':changed,'all_words_equal':changed==0,
        'max_abs_difference':float(delta.abs().max()),'relative_RMS_error':float(delta.square().sum().sqrt())/den}
