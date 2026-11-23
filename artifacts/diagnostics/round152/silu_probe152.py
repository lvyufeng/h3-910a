"""R152: time the MLP SiLU activation with device events, without touching math.

MiniMaxH3MLP.forward is fc1 -> chunk -> silu(gate)*up -> fc2, so everything that is
not fc1 or fc2 must be the SiLU and the elementwise multiply. This probe wraps
torch.nn.functional.silu to measure the SiLU span on device; the multiply and any
launch overhead stay in the remainder.
"""
from __future__ import annotations
import statistics

_STATE={'installed':False,'silu':[],'mlp':[],'pending':{}}
_KEEP=400


def install(torch,blocks):
    import torch.nn.functional as F
    original=F.silu

    def silu(input,*args,**kwargs):
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record()
        result=original(input,*args,**kwargs)
        end.record()
        if len(_STATE['silu'])<_KEEP:_STATE['silu'].append((start,end,tuple(input.shape)))
        return result

    F.silu=silu

    def wrap_mlp(module,index):
        def pre(mod,inputs):
            event=torch.npu.Event(enable_timing=True);event.record()
            _STATE['pending'][index]={'start':event}
        def post(mod,inputs,output):
            event=torch.npu.Event(enable_timing=True);event.record()
            pair=_STATE['pending'].pop(index,None)
            if pair is not None:_STATE['mlp'].append((pair['start'],event))
            return None
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for index,block in enumerate(blocks):
        wrap_mlp(block.mlp,index)
    _STATE['installed']=True
    return True


def collect(torch):
    if not _STATE['installed']:return {'installed':False}
    torch.npu.synchronize()
    def spans(rows):
        out=[]
        for start,end,*rest in rows:
            try:out.append(start.elapsed_time(end))
            except BaseException:pass
        return out
    silu=spans(_STATE['silu']);mlp=spans(_STATE['mlp'])
    result={'installed':True,'measurement_only_no_math_change':True,
            'no_synchronize_inside_forward':True,
            'silu_calls':len(silu),'mlp_calls':len(mlp)}
    if silu:
        result['silu']={'count':len(silu),'median_ms':round(statistics.median(silu),3),
                        'sum_ms':round(sum(silu),1),'min_ms':round(min(silu),3),'max_ms':round(max(silu),3)}
        result['silu_shapes']=sorted({s[2] for s in _STATE['silu']})[:4]
    if mlp:
        result['mlp']={'count':len(mlp),'median_ms':round(statistics.median(mlp),3),'sum_ms':round(sum(mlp),1)}
        if silu and len(silu)==len(mlp):
            pair=[m-s for m,s in zip(mlp,silu)]
            result['mlp_minus_silu_per_call_ms']={'median':round(statistics.median(pair),3),
                'sum':round(sum(pair),1),'note':'the elementwise multiply plus launch overhead'}
    _STATE['silu'].clear();_STATE['mlp'].clear()
    return result
