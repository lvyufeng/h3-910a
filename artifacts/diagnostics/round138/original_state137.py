"""Immutable logical-word state of original components; no computed weight getter."""
import hashlib
import json
from fingerprints129 import fingerprint,normalized_fingerprint


def qualified(value):return type(value).__module__+'.'+type(value).__qualname__


def component_state(component,torch):
    parameters={};empty=[]
    for name,p in component.named_parameters():
        local=p.to_local() if hasattr(p,'to_local') else p
        if local.numel():parameters[name]=fingerprint(local,torch)
        else:empty.append(name)
    buffers={name:fingerprint(b,torch) for name,b in component.named_buffers()}
    hooks=[]
    for index,hook in enumerate(getattr(component,'_omni_layerwise_hooks',())):
        hooks.append({'index':index,'class':qualified(hook),
            'cpu_masters':{str(k):fingerprint(v,torch,include_pointers=True) for k,v in hook.dtype_cpu_flattened_weights.items()},
            'metadata_SHA':hashlib.sha256(json.dumps({str(k):v for k,v in hook.dtype_metadata.items()},sort_keys=True,default=str).encode()).hexdigest(),
            'copy_stream':int(hook.copy_stream.npu_stream)})
    stagers={}
    for name in ('_omni_non_block_stager','_stager','_encoder_stager','_decoder_stager'):
        stager=getattr(component,name,None)
        if stager is not None:
            stagers[name]={'class':qualified(stager),'loaded':bool(stager.loaded),
                'cpu_masters':[fingerprint(g.master,torch,include_pointers=True) for g in stager._groups]}
    return {'parameters':parameters,'empty_parameter_names':empty,'buffers':buffers,'layerwise_hooks':hooks,'stagers':stagers}


def same_original_state(before,after):
    return normalized_fingerprint(before)==normalized_fingerprint(after) and before['layerwise_hooks']==after['layerwise_hooks'] and before['stagers']==after['stagers']
