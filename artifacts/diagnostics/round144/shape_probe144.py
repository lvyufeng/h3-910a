"""R144 shape probe: what token count do the DiT linears and attention actually see?

Measurement only. Records tensor shapes once per module; no values, no math.
"""
from __future__ import annotations

_STATE={'shapes':{},'fa':[],'installed':False}


def install(torch,model):
    targets={}
    for index,block in enumerate(model.blocks):
        for name in ('qkv_proj','out_proj'):
            module=getattr(block.attn,name,None)
            if module is not None:targets[f'block{index}.attn.{name}']=module
        for name in ('fc1','fc2'):
            module=getattr(block.mlp,name,None)
            if module is not None:targets[f'block{index}.mlp.{name}']=module
    for label,module in targets.items():
        def hook(mod,inputs,output,_label=label):
            if _label in _STATE['shapes']:return
            x=inputs[0] if inputs else None
            _STATE['shapes'][_label]={'input_shape':list(x.shape) if hasattr(x,'shape') else None,
                                      'input_dtype':str(getattr(x,'dtype',None)),
                                      'weight_shape':list(mod.weight.shape) if hasattr(mod,'weight') else None,
                                      'output_shape':list(output.shape) if hasattr(output,'shape') else None,
                                      'output_dtype':str(getattr(output,'dtype',None))}
        module.register_forward_hook(hook)
    try:
        from vllm_omni.diffusion.attention.backends import fa910a
        original=fa910a.flash_attention_910a
        def wrapped(q,k,v,*args,**kwargs):
            if len(_STATE['fa'])<4:
                _STATE['fa'].append({'q':list(q.shape),'k':list(k.shape),'v':list(v.shape),
                                     'dtype':str(q.dtype),'n_pad':kwargs.get('n_pad')})
            return original(q,k,v,*args,**kwargs)
        fa910a.flash_attention_910a=wrapped
    except BaseException as exc:
        _STATE['fa_error']=str(exc)
    _STATE['installed']=True
    return True


def collect():
    return {'installed':_STATE['installed'],'shapes':dict(_STATE['shapes']),'fa_calls':list(_STATE['fa']),
            'fa_error':_STATE.get('fa_error'),'measurement_only_no_math_change':True}
