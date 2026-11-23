"""R141 roofline side channel: TP all-reduce, weight H2D and pure copy bandwidth.

Measurement only: every wrapper calls the original object, never changes tensor
values, plans or streams, and never inserts a synchronize inside the model forward.
"""
from __future__ import annotations
import time

_STATE={'tp_reduce_calls':0,'tp_reduce_events':[],'installed_tp':False,
        'prefetch_calls':0,'prefetch_events':[]}


def install_tp_reduce_timing(torch):
    """Time the TP all-reduce used by RowParallelLinear (out_proj / fc2)."""
    import vllm.model_executor.layers.linear as linear_module
    original=linear_module.tensor_model_parallel_all_reduce
    if getattr(original,'_roofline141',False):return False
    def wrapped(*args,**kwargs):
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record()
        result=original(*args,**kwargs)
        end.record()
        _STATE['tp_reduce_events'].append((start,end))
        _STATE['tp_reduce_calls']+=1
        return result
    wrapped._roofline141=True
    linear_module.tensor_model_parallel_all_reduce=wrapped
    _STATE['installed_tp']=True
    return True


def install_prefetch_timing(torch):
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    original=LayerwiseOffloadHook.prefetch_layer
    if getattr(original,'_roofline141',False):return False
    def prefetch(self,*args,**kwargs):
        stream=getattr(self,'copy_stream',None)
        start=end=None
        if stream is not None:
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record(stream)
        result=original(self,*args,**kwargs)
        if start is not None:
            end.record(stream);_STATE['prefetch_events'].append((start,end));_STATE['prefetch_calls']+=1
        return result
    prefetch._roofline141=True
    LayerwiseOffloadHook.prefetch_layer=prefetch
    return True


def _sum(torch,events):
    total=0.0
    for start,end in events:
        try:total+=start.elapsed_time(end)
        except BaseException:pass
    return total


def h2d_bandwidth(torch,device,chunk_mb=256,iterations=8):
    bytes_per=chunk_mb*1024*1024
    host=torch.empty(bytes_per//2,dtype=torch.float16,pin_memory=True)
    target=torch.empty_like(host,device=device)
    torch.npu.synchronize();started=time.perf_counter()
    for _ in range(iterations):target.copy_(host,non_blocking=True)
    torch.npu.synchronize();elapsed=time.perf_counter()-started
    moved=bytes_per*iterations
    return {'chunk_mib':chunk_mb,'iterations':iterations,'elapsed_s':round(elapsed,4),
            'bandwidth_GB_per_s':round(moved/elapsed/1e9,2),'bytes_moved':moved,'pinned':True}


def collect(torch):
    result={'tp_all_reduce':{'calls':_STATE['tp_reduce_calls'],'device_ms_total':round(_sum(torch,_STATE['tp_reduce_events']),3),
                             'installed':_STATE['installed_tp'],
                             'note':'wraps vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce'},
            'prefetch':{'calls':_STATE['prefetch_calls'],'copy_stream_device_ms_total':round(_sum(torch,_STATE['prefetch_events']),3)},
            'measurement_only_no_math_change':True,
            'parent_child_regions_inclusive_do_not_sum':True,
            'region_timing_uses_synchronize_coarse_attribution':True}
    _STATE['tp_reduce_events']=[];_STATE['prefetch_events']=[]
    return result
