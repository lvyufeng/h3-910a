"""R147 device-event attribution for the DiT linears.

The R141 region numbers came from hooks that call torch.npu.synchronize() at both
edges, which drains every stream (including the weight copy stream) and therefore
folds cross-stream waiting into the region. This probe records torch events only,
never synchronizes inside the forward, and computes the spans after the step.
"""
from __future__ import annotations
import statistics

_STATE={'installed':False,'spans':{},'pending':{}}

TARGETS=(('block',None),('qkv','attn.qkv_proj'),('out_proj','attn.out_proj'),
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
    _STATE['installed']=True
    return True


def collect(torch):
    if not _STATE['installed']:return {'installed':False}
    torch.npu.synchronize()          # one sync, only after the step
    by_label={}
    for (label,index),pair in _STATE['pending'].items():
        if 'start' not in pair or 'end' not in pair:continue
        try:ms=pair['start'].elapsed_time(pair['end'])
        except BaseException:continue
        by_label.setdefault(label,[]).append(ms)
    rows={}
    for label,values in by_label.items():
        rows[label]={'count':len(values),'median_ms':round(statistics.median(values),3),
                     'sum_ms':round(sum(values),1),'min_ms':round(min(values),3),'max_ms':round(max(values),3)}
    _STATE['pending'].clear()
    return {'installed':True,'spans':rows,'blocks':len(by_label.get('block',[])),
            'measurement_only_no_math_change':True,
            'no_synchronize_inside_forward':True,
            'note':'device events only; the single synchronize happens after the measured step'}
