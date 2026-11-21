#!/usr/bin/env python3
"""Separate post-model NUMA/affinity pinned H2D probe, not an H3 speed promotion.

Only this private process thread is rebound. Existing model masters, kernel,
driver/SDK/NUMA policy are untouched. Page queries are read-only and optional.
"""
from __future__ import annotations
import ctypes
import errno
import json
import os
from pathlib import Path
import statistics

ROOT=Path('/tmp/h3_overlap12801')


def cpus(value):
    result=[]
    for span in value.strip().split(','):
        if '-' in span:
            first,last=map(int,span.split('-'));result.extend(range(first,last+1))
        else:result.append(int(span))
    return result


def page_nodes(tensor):
    try:lib=ctypes.CDLL('libnuma.so.1',use_errno=True);call=lib.numa_move_pages
    except (OSError,AttributeError):return {'known':False,'reason':'libnuma read-only page-query unavailable'}
    call.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    call.restype=ctypes.c_int
    base=tensor.data_ptr();size=tensor.numel()*tensor.element_size();page=os.sysconf('SC_PAGE_SIZE')
    addresses=[(base+offset)//page*page for offset in range(0,size,max(page,size//64//page*page))]
    pages=(ctypes.c_void_p*len(addresses))(*addresses);status=(ctypes.c_int*len(addresses))()
    rc=call(0,len(addresses),pages,None,status,0)  # nodes=NULL: query only; no page migration.
    if rc<0:return {'known':False,'errno':ctypes.get_errno(),'reason':os.strerror(ctypes.get_errno())}
    counts={}
    for value in status:counts[str(value)]=counts.get(str(value),0)+1
    return {'known':all(value>=0 for value in status),'sampled_pages':len(addresses),'page_node_or_negative_errno_counts':counts,
        'read_only_query_no_migration':True,'not_whole_tensor_page_enumeration':True}


def main():
    path=ROOT/'numa_probe128.json';assert not path.exists()
    model=json.loads((ROOT/'engine_overlap128a_process.json').read_text())
    assert model['exit_code']==0,'Do not continue probing after an uncertain model/runtime failure'
    assert not Path('/proc',str(model['pid'])).exists() and not Path('/proc',str(model['launcher_pid'])).exists()
    import torch
    import torch_npu
    from owner_stream128 import OwnedCopyStream128
    torch.npu.set_device(0);torch.set_num_threads(1)
    original=set(os.sched_getaffinity(0));nodes={int(p.parent.name[4:]):set(cpus(p.read_text()))&original for p in Path('/sys/devices/system/node').glob('node*/cpulist')}
    nodes={node:values for node,values in nodes.items() if values};assert len(nodes)>=2
    capability=OwnedCopyStream128(torch,device=0);masters=[];rows=[]
    destination=torch.empty(16*1024**2,dtype=torch.float16,device='npu:0')
    try:
        # Hold all host sources until final drain so a previous node's pinned arena is not silently reused.
        for node,affinity in sorted(nodes.items()):
            os.sched_setaffinity(0,affinity)
            master=torch.empty(16*1024**2,dtype=torch.float16,device='cpu',pin_memory=True)
            master.view(torch.uint8).numpy().fill(7);masters.append(master)
            measured=[]
            for repeat in range(6):
                with capability.copying() as stream:
                    begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    begin.record(stream);destination.copy_(master,non_blocking=True);end.record(stream)
                capability.compute.wait_event(end);capability.drain();measured.append(begin.elapsed_time(end))
            same=bool(torch.equal(destination.cpu().view(torch.uint8),master.view(torch.uint8)));assert same
            rows.append({'requested_first_touch_CPU_node':node,'thread_affinity_CPU_count':len(affinity),'source_bytes':master.numel()*master.element_size(),
                'source_pinned':master.is_pinned(),'source_pointer':master.data_ptr(),'physical_page_query':page_nodes(master),
                'H2D_copy_stream_median_warm_repeats1to5_ms':statistics.median(measured[1:]),'all_copy_times_ms':measured,
                'copied_words_equal':same,'not_proof_of_model_master_locality_or_formal_step_speed':True})
    finally:
        os.sched_setaffinity(0,original);capability.close()
    assert set(os.sched_getaffinity(0))==original
    result={'scope':__doc__,'completed':True,'model_process_already_finished':True,'source_data_synthetic_only':True,'rows':rows,
        'original_thread_affinity_restored':True,'model_masters_not_rebound_or_reallocated':True,'NUMA_policy_or_pages_moved':False,
        'host_source_reference_lifetimes_held_until_drain':True,'no_model_latency_or_precision_promotion':True}
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
