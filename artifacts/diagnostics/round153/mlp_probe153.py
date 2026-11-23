"""R153: per-op attribution inside MiniMaxH3MLP.forward.

The R152 module-level probe could not say where the mlp residual goes. This replaces
each block's mlp.forward with a copy that runs the exact same operators in the exact
same order and only inserts device-event records between them:

    fc1 -> chunk -> silu -> mul -> fc2

so the per-block spans are directly comparable, and the recorded result is returned
unchanged. Nothing about the arithmetic is modified.
"""
from __future__ import annotations
import statistics

_STATE={'installed':False,'records':[]}
_MAX_BLOCKS=6          # enough blocks for a median, small enough to stay cheap


def _record(torch,label,block_index,rows):
    event=torch.npu.Event(enable_timing=True)
    event.record()
    rows.append((label,block_index,event))


def install(torch,blocks):
    if _STATE['installed']:
        return True                      # idempotent: the hook group is installed more than once
    import torch.nn.functional as F

    for block in blocks[:_MAX_BLOCKS]:
        original=block.mlp.forward
        index=len(_STATE['records'])

        def make(original,index):
            def timed_forward(x,*args,**kwargs):
                rows=[]
                _record(torch,'start',index,rows)
                hidden,_=block.mlp.fc1(x)
                _record(torch,'after_fc1',index,rows)
                gate,up=hidden.chunk(2,dim=-1)
                _record(torch,'after_chunk',index,rows)
                activated=F.silu(gate)
                _record(torch,'after_silu',index,rows)
                prod=activated*up
                _record(torch,'after_mul',index,rows)
                out,_=block.mlp.fc2(prod)
                _record(torch,'after_fc2',index,rows)
                _STATE['records'].append(rows)
                return out
            return timed_forward

        block.mlp.forward=make(original,index)

    _STATE['installed']=True
    return True


def collect(torch):
    if not _STATE['installed']:return {'installed':False}
    torch.npu.synchronize()
    spans={}
    for rows in _STATE['records']:
        events={label:event for label,_,event in rows}
        order=['start','after_fc1','after_chunk','after_silu','after_mul','after_fc2']
        for a,b,naming in zip(order,order[1:],('fc1','chunk','silu','mul','fc2')):
            try:ms=events[a].elapsed_time(events[b])
            except BaseException:continue
            spans.setdefault(naming,[]).append(ms)
    out={'installed':True,'measurement_only_no_math_change':True,
         'ops_run_in_the_native_order':['fc1','chunk','silu','mul','fc2'],
         'no_synchronize_inside_forward':True,
         'blocks_profiled':len(_STATE['records'])}
    looks={}
    for naming,values in spans.items():
        if not values:continue
        trimmed=values[1:] if len(values)>1 else values      # drop the first call
        looks[naming]={'count':len(values),'median_ms':round(statistics.median(values),3),
                       'median_ms_without_first':round(statistics.median(trimmed),3)}
    total=sum(v['median_ms_without_first'] for v in looks.values()) if looks else None
    if looks:
        looks['attributed_total_ms']=round(total,2)
        # scale the per-block attribution to the whole model
        looks['per_layer_ms']={k:v['median_ms_without_first'] for k,v in looks.items() if k!='attributed_total_ms'}
        looks['per_step_ms']={k:round(v*50,1) for k,v in looks['per_layer_ms'].items()}
        looks['per_step_total_ms']=round(sum(looks['per_step_ms'].values()),1)
    out['spans']=looks
    _STATE['records'].clear()
    return out
