"""R153b: per-shape SiLU attribution inside the MLP, via hooks only.

Replacing mlp.forward breaks the harness dispatcher (it stages the layer weights), so
this probe keeps the native call path and only registers module hooks plus a wrapper
around torch.nn.functional.silu. Install is idempotent because the hook group is
installed more than once. Statistics use the median and drop the first call.
"""
from __future__ import annotations
import statistics

_STATE={'installed':False,'silu':{},'mlp':[],'fc1':[],'fc2':[]}


def _pair_hook(torch,module,store):
    def pre(mod,inputs):
        event=torch.npu.Event(enable_timing=True);event.record()
        store.append([event,None])
    def post(mod,inputs,output):
        event=torch.npu.Event(enable_timing=True);event.record()
        if store and store[-1][1] is None:
            store[-1][1]=event
        return None
    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post)


def install(torch,blocks):
    if _STATE['installed']:
        return True
    import torch.nn.functional as F
    original=F.silu

    def silu(input,*args,**kwargs):
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record()
        result=original(input,*args,**kwargs)
        end.record()
        key=tuple(input.shape)
        _STATE['silu'].setdefault(key,[]).append([start,end])
        return result

    F.silu=silu
    for block in blocks:
        _pair_hook(torch,block.mlp,_STATE['mlp'])
        _pair_hook(torch,block.mlp.fc1,_STATE['fc1'])
        _pair_hook(torch,block.mlp.fc2,_STATE['fc2'])
    _STATE['installed']=True
    return True


def collect(torch):
    if not _STATE['installed']:return {'installed':False}
    torch.npu.synchronize()
    def spans(store):
        out=[]
        for pair in store:
            if pair[1] is None:continue
            try:out.append(pair[0].elapsed_time(pair[1]))
            except BaseException:pass
        return out
    def stat(values):
        if not values:return None
        trimmed=values[1:] if len(values)>1 else values
        return {'count':len(values),'median_ms':round(statistics.median(trimmed),3),
                'dropped_first_call':len(values)>1}
    out={'installed':True,'measurement_only_no_math_change':True,'no_synchronize_inside_forward':True,
         'uses_the_native_forward_path':True}
    out['mlp']=stat(spans(_STATE['mlp']))
    out['fc1']=stat(spans(_STATE['fc1']))
    out['fc2']=stat(spans(_STATE['fc2']))
    per_shape={}
    for key,pairs in _STATE['silu'].items():
        values=spans(pairs)
        if values:per_shape[str(key)]=stat(values)
    out['silu_by_shape']=dict(sorted(per_shape.items(),key=lambda kv:-(kv[1]['median_ms']*kv[1]['count'])))
    if out['mlp'] and out['fc1'] and out['fc2']:
        resid=out['mlp']['median_ms']-out['fc1']['median_ms']-out['fc2']['median_ms']
        out['mlp_residual_ms']=round(resid,3)
        mlp_shape=next((k for k in per_shape if k.startswith('(10448, 7168')),None)
        if mlp_shape:
            out['siLU_for_the_mlp_shape_ms']=per_shape[mlp_shape]['median_ms']
            out['siLU_share_of_residual_percent']=round(100*per_shape[mlp_shape]['median_ms']/resid,1)
            out['mul_plus_chunk_ms']=round(resid-per_shape[mlp_shape]['median_ms'],3)
    _STATE['silu'].clear();_STATE['mlp'].clear();_STATE['fc1'].clear();_STATE['fc2'].clear()
    return out
