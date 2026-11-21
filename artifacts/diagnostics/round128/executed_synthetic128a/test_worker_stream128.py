#!/usr/bin/env python3
"""Synthetic execution-thread/allocator/ready/consumer tests, not a model speed claim."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

ROOT=Path('/tmp/h3_overlap12801')


def intersections(copies,bodies):
    overlap=0.0
    for copy in copies:
        spans=[]
        for body in bodies:
            lo=max(copy['start_ms'],body['start_ms']);hi=min(copy['end_ms'],body['end_ms'])
            if hi>lo:spans.append((lo,hi))
        spans.sort();merged=[]
        for lo,hi in spans:
            if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
            else:merged.append((lo,hi))
        overlap+=sum(hi-lo for lo,hi in merged)
    return overlap


def main():
    assert ROOT.is_dir() and not (ROOT/'worker_stream_unit128.json').exists()
    import torch
    import torch_npu
    from vllm_omni.platforms import current_omni_platform as platform
    from owner_stream128 import OwnedCopyStream128,StreamOwnershipError
    torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    original_factory=platform.Stream;torch.set_num_threads(4)
    result={'scope':__doc__,'completed':False,'weights_loaded_or_transferred':False,'CPU_or_NPU_model_inference':False,
        'torch_version':torch.__version__,'torch_npu_version':torch_npu.__version__,
        'actual_device_name':torch.npu.get_device_name(0),'main_thread_id':threading.get_native_id(),
        'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    parent=OwnedCopyStream128(torch,device=0);result['parent_stream']=parent.facts();messages=queue.Queue()
    def wrong_thread():
        try:
            parent.require_owner()
        except StreamOwnershipError as exc:messages.put({'wrong_thread_refused_before_runtime':True,'message':str(exc)})
        else:messages.put({'wrong_thread_refused_before_runtime':False})
    thread=threading.Thread(target=wrong_thread);thread.start();thread.join(timeout=10);assert not thread.is_alive()
    result.update(messages.get_nowait());assert result['wrong_thread_refused_before_runtime'];parent.close()
    def execution():
        try:
            torch.npu.set_device(0)
            capability=OwnedCopyStream128(torch,device=0);facts=capability.facts()
            # Exhaustive Half word H2D, transposed/offset layouts stay unmodified.
            words=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(256,256)
            exact=[]
            for label,host in [('all_Half_words',words),('transpose',words.t()),('offset',words[3:249,2:254])]:
                master=torch.empty_strided(host.shape,host.stride(),dtype=host.dtype,device='cpu',pin_memory=True)
                master.copy_(host)
                with capability.copying() as copy:
                    device=torch.empty_strided(master.shape,master.stride(),dtype=master.dtype,device='npu:0')
                    device.copy_(master,non_blocking=True);ready=torch.npu.Event();ready.record(copy)
                capability.compute.wait_event(ready)
                returned=device.cpu().contiguous();capability.drain()
                equal=torch.equal(returned.view(torch.uint8),master.contiguous().view(torch.uint8));assert equal
                exact.append({'label':label,'shape':list(master.shape),'stride':list(master.stride()),'all_raw_words_equal':equal})
            # Synthetic prefetch pipeline. Buffers are owned and retained until drain.
            def trial(asynchronous):
                stream=capability.copy if asynchronous else capability.compute
                master=torch.arange(4*1024**2,dtype=torch.float32).remainder_(1024).to(torch.float16).pin_memory()
                slots=[torch.empty_like(master,device='npu:0') for _ in range(2)]
                torch.manual_seed(128)
                matrix=torch.randn((1024,1024),dtype=torch.float16,device='npu:0')
                weight=torch.randn((1024,1024),dtype=torch.float16,device='npu:0')
                a=torch.npu.Event();b=torch.npu.Event()
                ready={};consumed={};pairs=[];bodies=[];results=[]
                def copy_to(index):
                    key=index%2;stream.wait_stream(capability.compute)
                    if key in consumed:stream.wait_event(consumed[key])
                    with torch.npu.stream(stream):
                        first=torch.npu.Event(enable_timing=True);last=torch.npu.Event(enable_timing=True)
                        first.record(stream);slots[key].copy_(master,non_blocking=True);last.record(stream)
                        event=torch.npu.Event();event.record(stream);ready[index]=event
                    if index:pairs.append((first,last))
                copy_to(0);capability.drain()
                epoch=torch.npu.Event(enable_timing=True);epoch.record(capability.compute);start=time.perf_counter()
                for index in range(9):
                    capability.compute.wait_event(ready[index])
                    if index<8:copy_to(index+1)
                    first=torch.npu.Event(enable_timing=True);last=torch.npu.Event(enable_timing=True)
                    first.record(capability.compute)
                    # Consume current transfer then independent compute, both on consumer.
                    results.append(slots[index%2].clone())
                    for _ in range(8):torch.mm(matrix,weight)
                    last.record(capability.compute);bodies.append((first,last))
                    done=torch.npu.Event();done.record(capability.compute);consumed[index%2]=done
                capability.drain();wall=time.perf_counter()-start
                all_equal=all(torch.equal(x.cpu().view(torch.uint8),master.view(torch.uint8)) for x in results);assert all_equal
                copy_rows=[{'start_ms':epoch.elapsed_time(a),'end_ms':epoch.elapsed_time(b),'elapsed_ms':a.elapsed_time(b)} for a,b in pairs]
                body_rows=[{'start_ms':epoch.elapsed_time(a),'end_ms':epoch.elapsed_time(b),'elapsed_ms':a.elapsed_time(b)} for a,b in bodies]
                overlap=intersections(copy_rows,body_rows)
                if not asynchronous:assert overlap<=0.01,'Serial control unexpectedly overlaps'
                return {'mode':'async' if asynchronous else 'serial','host_wall_s':wall,'actual_copy_stream':int(stream.npu_stream),
                    'copies':copy_rows,'consumer_spans':body_rows,'copy_overlap_with_consumer_spans_ms':overlap,
                    'all_9_consumed_transfer_buffers_raw_bytes_equal':all_equal,
                    'consumer_spans_are_inclusive_not_pure_device_busy':True,'not_model_performance_evidence':True}
            trials=[trial(False),trial(True),trial(False)]
            capability.close();assert capability.closed
            messages.put({'execution_stream':facts,'exhaustive_Half_copy_checks':exact,'serial_async_restored_synthetic_trials':trials,
                          'execution_stream_drained_then_closed':True})
        except BaseException as exc:
            import traceback
            messages.put({'failure':{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}})
    worker=threading.Thread(target=execution,name='R128ExecutionThread');worker.start();worker.join(timeout=480)
    assert not worker.is_alive(),'Synthetic worker timed out; stop this process, no blind retry'
    row=messages.get_nowait();result.update(row)
    assert platform.Stream is original_factory
    result['global_platform_factory_unchanged']=True
    result['completed']='failure' not in result
    (ROOT/'worker_stream_unit128.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert result['completed'],result.get('failure')


if __name__=='__main__':main()
