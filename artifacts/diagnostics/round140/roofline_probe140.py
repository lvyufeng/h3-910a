"""R140 roofline side-channel: H2D weight staging + pure PCIe/HCCS copy bandwidth.

Measurement only. No math, no candidate, no tolerance. The prefetch wrapper only
records device events that the native hook already implies; it never adds a
synchronize inside the model forward and never changes staged bytes.
"""
from __future__ import annotations
from pathlib import Path
import statistics
import time

_STATE={'prefetch_calls':0,'prefetch_ms_total':0.0,'prefetch_bytes':0,'events':[],'stream':None,'installed':False}


def install_prefetch_timing(torch):
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    original=LayerwiseOffloadHook.prefetch_layer
    if getattr(original,'_roofline140',False):return False
    def prefetch(self,*args,**kwargs):
        stream=getattr(self,'copy_stream',None)
        start=end=None
        if stream is not None:
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record(stream)
        result=original(self,*args,**kwargs)
        if start is not None:
            end.record(stream)
            _STATE['events'].append((start,end))
            _STATE['prefetch_calls']+=1
        return result
    prefetch._roofline140=True
    LayerwiseOffloadHook.prefetch_layer=prefetch
    _STATE['installed']=True
    return True


def collect_prefetch_timing(torch):
    total=0.0
    for start,end in _STATE['events']:
        try:total+=start.elapsed_time(end)
        except BaseException:pass
    _STATE['events']=[]
    _STATE['prefetch_ms_total']=total
    return {'prefetch_calls':_STATE['prefetch_calls'],'prefetch_device_ms_total':round(total,3),
            'note':'copy-stream device time; parent of any H2D copies the hook issues'}


def h2d_bandwidth(torch,device,chunk_mb=256,iterations=8):
    """Pure pinned host->device copy rate. No model math, no collectives."""
    bytes_per=chunk_mb*1024*1024
    host=torch.empty(bytes_per//2,dtype=torch.float16,pin_memory=True)
    device_tensor=torch.empty_like(host,device=device)
    torch.npu.synchronize();started=time.perf_counter()
    for _ in range(iterations):
        device_tensor.copy_(host,non_blocking=True)
    torch.npu.synchronize();elapsed=time.perf_counter()-started
    moved=bytes_per*iterations
    return {'chunk_mib':chunk_mb,'iterations':iterations,'elapsed_s':round(elapsed,4),
            'bandwidth_GB_per_s':round(moved/elapsed/1e9,2) if elapsed else None,
            'bytes_moved':moved,'pinned':True}


def receipt():
    return {'prefetch':{'calls':_STATE['prefetch_calls'],'device_ms_total':round(_STATE['prefetch_ms_total'],3)},
            'installed':_STATE['installed']}
