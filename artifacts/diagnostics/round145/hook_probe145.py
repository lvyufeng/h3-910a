"""R145 hook overhead: how much of each block is the offloader itself?

Measurement only. Times the host side of the native layerwise hook (prefetch issue,
offload/clear, pre_forward) with perf_counter. No device synchronize is added inside
the forward, no storage/stream/value is changed.
"""
from __future__ import annotations

_STATE={'installed':False,'host':{},'calls':{}}


def _bump(kind,ms):
    _STATE['host'][kind]=_STATE['host'].get(kind,0.0)+ms
    _STATE['calls'][kind]=_STATE['calls'].get(kind,0)+1


def install(torch,blocks,platform_module):
    import time
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    for name in ('prefetch_layer','offload_layer','pre_forward','post_forward'):
        original=getattr(LayerwiseOffloadHook,name)
        if getattr(original,'_hook145',False):continue
        def wrapper(self,*args,_orig=original,_name=name,**kwargs):
            started=time.perf_counter()
            result=_orig(self,*args,**kwargs)
            _bump(_name,(time.perf_counter()-started)*1000.0)
            return result
        wrapper._hook145=True
        setattr(LayerwiseOffloadHook,name,wrapper)
    # block wall time so the hook share can be expressed against it
    import time as _t
    for i,b in enumerate(blocks):
        def pre(module,inputs,_i=i):
            _STATE.setdefault('block_start',{})[_i]=_t.perf_counter()
        def post(module,inputs,output,_i=i):
            start=_STATE.get('block_start',{}).pop(_i,None)
            if start is not None:
                _bump('block',(_t.perf_counter()-start)*1000.0)
            return None
        b.register_forward_pre_hook(pre)
        b.register_forward_hook(post)
    _STATE['installed']=True
    return True


def collect():
    host=dict(_STATE['host']);calls=dict(_STATE['calls'])
    block_ms=host.get('block')
    result={'installed':_STATE['installed'],'host_ms_total':{k:round(v,1) for k,v in sorted(host.items())},
            'calls':calls,
            'per_call_ms':{k:round(host[k]/calls[k],2) for k in host if calls.get(k)},
            'hook_share_of_block_percent':(round(100*(host.get('pre_forward',0.0)+host.get('post_forward',0.0))/block_ms,2)
                                           if block_ms else None),
            'note':'host wall time only; prefetch_layer issues the async copy and also rebinds the next block tensors on the host',
            'measurement_only_no_math_change':True,'no_synchronize_inside_forward':True}
    _STATE['host'].clear();_STATE['calls'].clear()
    return result
