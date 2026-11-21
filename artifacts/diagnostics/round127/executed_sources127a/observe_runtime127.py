"""Low-interference private runtime observations. Never changes streams or placement."""
from __future__ import annotations
from collections import Counter, defaultdict
import contextvars
import functools
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from runtime_context127 import event,request_index

_LABELS={}
_PREFETCH={}
_DEVICE_EVENTS=[]


def stream_facts(torch,copy_stream=None):
    current=torch.npu.current_stream()
    result={'thread_id':threading.get_native_id(),'current_stream':int(current.npu_stream),
            'device':str(current.device),'affinity':sorted(os.sched_getaffinity(0))}
    if copy_stream is not None:
        result.update({'copy_stream':int(copy_stream.npu_stream),'copy_is_current':copy_stream==current})
    return result


def memory(torch):
    try:
        free,total=torch.npu.mem_get_info()
        return {'allocated_bytes':torch.npu.memory_allocated(),'reserved_bytes':torch.npu.memory_reserved(),
            'peak_allocated_bytes':torch.npu.max_memory_allocated(),'peak_reserved_bytes':torch.npu.max_memory_reserved(),
            'free_bytes':int(free),'total_bytes':int(total)}
    except Exception as exc:return {'unavailable':type(exc).__name__+': '+str(exc)}


def numa_sample(pointers):
    try:
        mappings=[]
        for line in Path('/proc/self/maps').read_text().splitlines():
            lohi=line.split()[0];lo,hi=(int(v,16) for v in lohi.split('-'))
            mappings.append((lo,hi,line))
        nodes={int(line.split()[0],16):line for line in Path('/proc/self/numa_maps').read_text().splitlines()}
        results=[]
        for ptr in pointers:
            mapping=next((v for v in mappings if v[0]<=ptr<v[1]),None)
            results.append({'pointer':ptr,'vma':mapping[2] if mapping else None,
                'numa_map':nodes.get(mapping[0]) if mapping else None,
                'node_counts_are_whole_VMA_not_exact_tensor_pages':True})
        return results
    except Exception as exc:return [{'unavailable':type(exc).__name__+': '+str(exc)}]


def stager_snapshot(stager,*,include_numa=False):
    groups=getattr(stager,'_groups',())
    masters=[g.master for g in groups]
    metadata=[(int(t.data_ptr()),int(t.numel()*t.element_size()),str(t.dtype),bool(t.is_pinned())) for t in masters]
    sample=sorted(metadata,key=lambda v:v[1],reverse=True)[:4]
    row={'object_id':id(stager),'label':_LABELS.get(id(stager),'unmapped'),'loaded':bool(stager.loaded),
        'master_groups':len(groups),'master_bytes':sum(v[1] for v in metadata),'all_masters_pinned':all(v[3] for v in metadata),
        'master_metadata_sha256':hashlib.sha256(json.dumps(metadata).encode()).hexdigest(),
        'sample_largest_masters':sample,'device_storage_bytes':sum(t.numel()*t.element_size() for t in stager._device_storages),
        'device_storage_count':len(stager._device_storages)}
    if include_numa:row['pinned_NUMA_VMA_samples']=numa_sample([v[0] for v in sample])
    return row


def describe_pipeline(pipeline,torch,*,numa=False):
    stagers=[]
    for label in ('text_encoder','video_vae','audio_vae'):
        component=getattr(pipeline,label,None)
        if component is None:continue
        for attr in ('_stager','_encoder_stager','_decoder_stager','_omni_non_block_stager'):
            obj=getattr(component,attr,None)
            if obj is not None:
                _LABELS[id(obj)]=label+attr
                stagers.append(stager_snapshot(obj,include_numa=numa))
    transformer=pipeline.transformer
    parameter_bytes=Counter();materialized=0
    for p in transformer.parameters():
        parameter_bytes[str(p.dtype)]+=p.numel()*p.element_size()
    from vllm_omni.diffusion.hooks import HookRegistry
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    hooked=[]
    for index,block in enumerate(transformer.blocks):
        registry=getattr(block,'_hook_registry',None)
        if registry is None:continue
        hook=registry.get_hook(LayerwiseOffloadHook._HOOK_NAME)
        if hook is None:continue
        materialized+=bool(hook.is_materialized)
        hooked.append({'index':index,'hook_id':id(hook),'materialized':bool(hook.is_materialized),
            'pinned_bytes':sum(t.numel()*t.element_size() for t in hook.dtype_cpu_flattened_weights.values()),
            'all_pinned':all(t.is_pinned() for t in hook.dtype_cpu_flattened_weights.values()),
            'streams':stream_facts(torch,hook.copy_stream)})
    return {'stagers':stagers,'DiT_blocks':len(transformer.blocks),'layerwise_hooks':hooked,
        'materialized_DiT_blocks':materialized,'logical_parameter_bytes_by_dtype':dict(parameter_bytes),
        'HF32_matmul':torch.npu.matmul.allow_hf32,'HF32_conv':torch.npu.conv.allow_hf32,'memory':memory(torch)}


def drain_device_events():
    rows=[]
    for row,begin,end in _DEVICE_EVENTS:
        rows.append({**row,'current_stream_elapsed_ms':begin.elapsed_time(end),
            'scope':'native stage transition on consumer stream, includes its waits; not pure copy-engine busy'})
    _DEVICE_EVENTS.clear()
    return rows


def reset_request_observations():
    _PREFETCH.clear();_DEVICE_EVENTS.clear()


def install(torch):
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager,BoundedAllocatorCache
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    def wrap_stager(method):
        original=getattr(PinnedModuleStager,method)
        @functools.wraps(original)
        def call(self,*args,**kwargs):
            if not request_index():return original(self,*args,**kwargs)
            before=stager_snapshot(self);facts=stream_facts(torch,self.copy_stream)
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record();start=time.perf_counter()
            try:return original(self,*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-start;end.record()
                after=stager_snapshot(self)
                row={'method':method,'request_index':request_index(),'label':_LABELS.get(id(self),'unmapped'),
                    'host_elapsed_s':elapsed,'before':before,'after':after,'streams':facts}
                event('stager_transition',**row)
                _DEVICE_EVENTS.append(({'method':method,'label':row['label']},begin,end))
        setattr(PinnedModuleStager,method,call)
    for method in ('load','_load_once','offload'):wrap_stager(method)
    original_cache=BoundedAllocatorCache.release_if_needed
    @functools.wraps(original_cache)
    def cache(self,*args,**kwargs):
        if not request_index():return original_cache(self,*args,**kwargs)
        before=memory(torch);start=time.perf_counter()
        result=original_cache(self,*args,**kwargs)
        event('allocator_cache_native_release',force=bool(kwargs.get('force',False)),released=result,
            host_elapsed_s=time.perf_counter()-start,before=before,after=memory(torch))
        return result
    BoundedAllocatorCache.release_if_needed=cache
    original_prefetch=LayerwiseOffloadHook.prefetch_layer
    @functools.wraps(original_prefetch)
    def prefetch(self,*args,**kwargs):
        if request_index():
            key=(id(self),threading.get_native_id())
            if key not in _PREFETCH:
                _PREFETCH[key]={'hook_id':id(self),'calls':0,'nonblocking_calls':0,'streams':stream_facts(torch,self.copy_stream),
                    'weight_bytes_per_copy':sum(t.numel()*t.element_size() for t in self.dtype_cpu_flattened_weights.values())}
            row=_PREFETCH[key];row['calls']+=1
            row['nonblocking_calls']+=bool(kwargs.get('non_blocking',args[0] if args else True))
        return original_prefetch(self,*args,**kwargs)
    LayerwiseOffloadHook.prefetch_layer=prefetch


def prefetch_summary():return list(_PREFETCH.values())
