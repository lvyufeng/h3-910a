"""Read-only CPU packing diagnostic; native copies and model values are untouched."""
from __future__ import annotations
import faulthandler
import functools
import json
import os
from pathlib import Path
import threading
import time

_FILES=[]


def install():
    import torch
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    from vllm_omni.diffusion.offloader.tensor_utils import physical_storage_numel
    from runtime_context127 import SESSION
    path=SESSION/f'initialization_pack_pid{os.getpid()}127.jsonl'
    crash=SESSION/f'initialization_crash_pid{os.getpid()}127.txt'
    file=crash.open('x');_FILES.append(file)
    faulthandler.enable(file=file,all_threads=True)
    native=LayerwiseOffloadHook._to_cpu
    calls=0
    def record(name,**data):
        with path.open('a') as f:f.write(json.dumps({'event':name,'pid':os.getpid(),'thread_id':threading.get_native_id(),
            'unix':time.time(),**data})+'\n')
    @functools.wraps(native)
    def to_cpu(params,bufs,pin_memory=True):
        nonlocal calls
        calls+=1;sequence=calls;description=[]
        for name,target in [*params.items(),*bufs.items()]:
            value=target.to_local() if hasattr(target,'to_local') else target
            span=physical_storage_numel(value)
            bytes_required=(value.storage_offset()+span)*value.element_size() if value.numel() else 0
            storage_bytes=value.untyped_storage().nbytes()
            row={'name':name,'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),
                'device':str(value.device),'numel':value.numel(),'storage_span_numel':span,
                'storage_offset':value.storage_offset(),'storage_bytes':storage_bytes,
                'required_storage_bytes':bytes_required,'data_pointer':value.data_ptr(),'requires_grad':value.requires_grad}
            description.append(row)
            if value.device.type=='cpu' and bytes_required>storage_bytes:
                record('pack_invalid_storage_span',sequence=sequence,tensor=row)
                raise ValueError('CPU storage span invalid before native Half packing: '+name)
        record('native_pack_begin',sequence=sequence,pin_memory=pin_memory,tensors=description)
        start=time.perf_counter()
        try:
            result=native(params,bufs,pin_memory)
        except BaseException as exc:
            record('native_pack_exception',sequence=sequence,type=type(exc).__name__,message=str(exc));raise
        record('native_pack_end',sequence=sequence,host_elapsed_s=time.perf_counter()-start,
            CPU_master_bytes=sum(t.numel()*t.element_size() for t in result[0].values()),
            all_pinned=all(t.is_pinned() for t in result[0].values()),values_or_copy_implementation_changed=False)
        return result
    LayerwiseOffloadHook._to_cpu=staticmethod(to_cpu)
    record('native_packing_diagnostics_installed',CPU_copy_implementation_changed=False,model_precision_changed=False)
