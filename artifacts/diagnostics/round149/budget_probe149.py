"""R149 complete event-based step budget: no synchronize inside the forward.

Extends the R147 probe to every block-level module plus the FA entry point, so the
step can be split into FA, the four linears, attention/mlp/adaln/norm sub-totals and
the residual that belongs to collectives, residual adds and staging waits.
"""
from __future__ import annotations
import statistics

_STATE={'installed':False,'pending':{},'fa':[]}

# label -> dotted attribute path (None means the block itself)
TARGETS=(('block',None),('attn','attn'),('mlp','mlp'),('adaln','adaln_proj'),
         ('norm1','norm1'),('norm2','norm2'),
         ('qkv','attn.qkv_proj'),('out_proj','attn.out_proj'),
         ('fc1','mlp.fc1'),('fc2','mlp.fc2'))


def install(torch,blocks):
    def wrap(label,module,index):
        def pre(mod,inputs):
            event=torch.npu.Event(enable_timing=True);event.record()
            _STATE['pending'].setdefault((label,index),{})['start']=event
        def post(mod,inputs,output):
            event=torch.npu.Event(enable_timing=True);event.record()
            _STATE['pending'].setdefault((label,index),{})['end']=event
            return None
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)
    for index,block in enumerate(blocks):
        for label,attr in TARGETS:
            module=block
            if attr is not None:
                for part in attr.split('.'):
                    module=getattr(module,part,None)
                    if module is None:break
            if module is None:raise RuntimeError(f'missing module {attr} for {label}')
            wrap(label,module,index)
    from vllm_omni.diffusion.attention.backends import fa910a
    original=fa910a.flash_attention_910a
    def wrapped(q,k,v,*args,**kwargs):
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record()
        result=original(q,k,v,*args,**kwargs)
        end.record()
        _STATE['fa'].append((start,end))
        return result
    fa910a.flash_attention_910a=wrapped
    _STATE['installed']=True
    return True


def collect(torch):
    if not _STATE['installed']:return {'installed':False}
    torch.npu.synchronize()
    by_label={}
    for (label,index),pair in _STATE['pending'].items():
        if 'start' not in pair or 'end' not in pair:continue
        try:ms=pair['start'].elapsed_time(pair['end'])
        except BaseException:continue
        by_label.setdefault(label,[]).append(ms)
    fa=[]
    for start,end in _STATE['fa']:
        try:fa.append(start.elapsed_time(end))
        except BaseException:pass
    spans={label:{'count':len(v),'median_ms':round(statistics.median(v),3),'sum_ms':round(sum(v),1)}
           for label,v in by_label.items()}
    if fa:spans['fa']={'count':len(fa),'median_ms':round(statistics.median(fa),3),'sum_ms':round(sum(fa),1)}
    _STATE['pending'].clear();_STATE['fa']=[]
    return {'installed':True,'spans':spans,'blocks':len(by_label.get('block',[])),
            'measurement_only_no_math_change':True,'no_synchronize_inside_forward':True}
