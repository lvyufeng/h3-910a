"""Private native-copy DiT adapter: owner thread, independent stream, bounded lifetimes.

Only the 50 existing DiT hook instances are changed. No reusable/resident pool,
model arithmetic, pinned-master layout or native copy implementation changes.
Real detached device tensors (not Parameter references) are retained until a
separate consumer-complete event. Explicit backpressure is counted, never hidden.
"""
from __future__ import annotations
from types import MethodType
import hashlib
import json
import time
from owner_stream128 import OwnedCopyStream128

_MISSING=object()


class NativeCopyAdapter128:
    METHODS=('prefetch_layer','offload_layer','pre_forward','post_forward')

    def __init__(self,torch,blocks,hooks,*,asynchronous=True,max_retired_blocks=2):
        self.torch=torch;self.blocks=list(blocks);self.hooks=list(hooks)
        assert len(self.blocks)==len(self.hooks)==50
        assert [id(h.next_block) for h in self.hooks]==[id(b) for b in self.blocks]
        assert all(h._prev_hook is self.hooks[(i-1)%50] for i,h in enumerate(self.hooks))
        self.asynchronous=asynchronous;self.max_retired=max_retired_blocks
        assert max_retired_blocks==2
        self.state='new';self.capability=None;self.saved=[];self.owned=[]
        self.live={};self.ready={};self.prepared=set();self.retired=[]
        self.active_owner=None;self.expected_owner=0;self.completed_cycles=0
        self.copy_count=0;self.copy_bytes=0;self.recovered_missing=0
        self.retirement_backpressure_count=0;self.retirement_backpressure_host_s=0.0
        self.peak_retired_blocks=0;self.peak_retired_bytes=0
        self.profile=False;self.copy_pairs=[];self.body_pairs=[]
        self.audit_each_copy=False;self.byte_proofs=[]
        self.master_signatures=[];self.metadata_signatures=[]

    @staticmethod
    def signature(tensor):
        return (id(tensor),str(tensor.device),str(tensor.dtype),tuple(tensor.shape),tuple(tensor.stride()),
            tensor.storage_offset(),tensor.data_ptr(),tensor.untyped_storage()._cdata,tensor._version,bool(tensor.is_pinned()))

    @staticmethod
    def metadata_sha(hook):
        return hashlib.sha256(json.dumps({str(k):v for k,v in hook.dtype_metadata.items()},sort_keys=True,default=str).encode()).hexdigest()

    def _require(self):
        assert self.state=='active','Adapter is not active; do not continue a failed or closed candidate'
        self.capability.require_owner()

    @staticmethod
    def held_tensors(hook):
        # .data/.detach() makes a storage-holding tensor independent of later Parameter.data rebinding.
        targets=[*hook.next_block_parameters.values(),*hook.next_block_buffers.values()]
        return [(t.to_local() if hasattr(t,'to_local') else t.data).detach() for t in targets]

    def activate(self):
        assert self.state=='new'
        device=self.torch.npu.current_device()
        assert all(h.device.index==device for h in self.hooks)
        # Owner-thread creation is lazy here, not in the control/init constructor.
        self.capability=OwnedCopyStream128(self.torch,device=device)
        self.torch.npu.synchronize()  # Boundary, before any instance is modified.
        self.master_signatures=[{dtype:self.signature(t) for dtype,t in h.dtype_cpu_flattened_weights.items()} for h in self.hooks]
        self.metadata_signatures=[self.metadata_sha(h) for h in self.hooks]
        assert all(t.device.type=='cpu' and t.is_pinned() for h in self.hooks for t in h.dtype_cpu_flattened_weights.values())
        for index,h in enumerate(self.hooks):
            if all(t.numel()>0 for t in self.held_tensors(h)):
                self.live[index]=self.held_tensors(h);self.prepared.add(index)
        assert self.prepared=={0},'Activate only at a drained complete native 50-block ring boundary'
        first_ready=self.torch.npu.Event();first_ready.record(self.capability.compute);self.ready[0]=first_ready
        self.state='active'
        try:
            for index,h in enumerate(self.hooks):
                original={name:getattr(h,name) for name in self.METHODS}
                attributes={name:h.__dict__.get(name,_MISSING) for name in self.METHODS}
                self.saved.append((h,original,attributes,h.copy_stream))
                owner=(index-1)%50
                def prefetch(hook,non_blocking=True,*,target=index,original=original['prefetch_layer']):
                    return self._prefetch(target,hook,original,non_blocking)
                def offload(hook,*,owner=owner,original=original['offload_layer']):
                    return self._offload(owner,hook,original)
                def pre(hook,module,*args,owner=owner,original=original['pre_forward'],**kwargs):
                    self._require()
                    assert self.active_owner is None and owner==self.expected_owner,'Only the entire sequential 50-block graph is supported'
                    self.active_owner=owner
                    if owner not in self.prepared or not hook.is_materialized:
                        self.recovered_missing+=1
                        hook._prev_hook.prefetch_layer(non_blocking=False)
                    # Current-layer ready is separate from hook._prefetch_done (NEXT-layer ready).
                    self.capability.compute.wait_event(self.ready[owner])
                    result=original(module,*args,**kwargs)
                    if self.profile:
                        pair=(self.torch.npu.Event(enable_timing=True),self.torch.npu.Event(enable_timing=True))
                        pair[0].record(self.capability.compute);self.body_pairs.append((owner,pair))
                    return result
                def post(hook,module,output,owner=owner,original=original['post_forward']):
                    self._require();assert self.active_owner==owner
                    if self.profile:self.body_pairs[-1][1][1].record(self.capability.compute)
                    result=original(module,output)
                    self.active_owner=None;self.expected_owner=(owner+1)%50
                    if owner==49:self.completed_cycles+=1
                    return result
                replacements={'prefetch_layer':MethodType(prefetch,h),'offload_layer':MethodType(offload,h),
                    'pre_forward':MethodType(pre,h),'post_forward':MethodType(post,h)}
                h.copy_stream=self.capability.copy if self.asynchronous else self.capability.compute
                for name,method in replacements.items():
                    setattr(h,name,method);self.owned.append((h,name,method))
            return self
        except BaseException:
            self.rollback();raise

    def _assert_masters(self,index,hook):
        assert self.metadata_sha(hook)==self.metadata_signatures[index],'Native dtype/shape/stride metadata changed'
        assert {dtype:self.signature(t) for dtype,t in hook.dtype_cpu_flattened_weights.items()}==self.master_signatures[index],'Pinned native master mutated or rebound'

    def _prefetch(self,index,hook,original,non_blocking):
        self._require();self._assert_masters(index,hook);self._reap()
        if index in self.prepared:
            assert not all(t.numel()>0 for t in self.held_tensors(hook)),'Do not silently replace already-prepared native weights'
            # A skipped consumer did not use these tensors, but their prior copy must finish before free.
            old=self.live.pop(index);self.prepared.remove(index)
            self.retired.append((self.ready.pop(index),old,index));self._reap()
        pair=None
        if self.profile:
            pair=(self.torch.npu.Event(enable_timing=True),self.torch.npu.Event(enable_timing=True));pair[0].record(hook.copy_stream)
        original(non_blocking=non_blocking)  # Keep native allocation, raw H2D, binding and NEXT-ready implementation.
        assert hook._prefetch_done is not None
        self.live[index]=self.held_tensors(hook);self.prepared.add(index);self.ready[index]=hook._prefetch_done
        assert all(t.numel()>0 for t in self.live[index])
        self.copy_count+=1;self.copy_bytes+=sum(t.numel()*t.element_size() for t in hook.dtype_cpu_flattened_weights.values())
        if self.profile:
            pair[1].record(hook.copy_stream);self.copy_pairs.append((index,pair))
        if self.audit_each_copy:self._audit_transfer(index,hook)

    def _offload(self,index,hook,original):
        self._require();assert index in self.prepared and index in self.live
        held=self.live[index]
        original()  # Native consumer wait for NEXT-ready and current Parameter placeholder rebinding.
        done=self.torch.npu.Event();done.record(self.capability.compute)
        self.live.pop(index);self.prepared.remove(index);self.ready.pop(index,None)
        self.retired.append((done,held,index));self._reap()

    def _reap(self):
        # A ready event is never reused as a consumed-storage fence.
        self.retired=[row for row in self.retired if not row[0].query()]
        while len(self.retired)>self.max_retired:
            start=time.perf_counter();self.retired[0][0].synchronize()
            self.retirement_backpressure_host_s+=time.perf_counter()-start;self.retirement_backpressure_count+=1
            self.retired.pop(0)
        self.peak_retired_blocks=max(self.peak_retired_blocks,len(self.retired))
        def bytes_held(row):
            unique={t.untyped_storage()._cdata:t.untyped_storage().nbytes() for t in row[1]}
            return sum(unique.values())
        self.peak_retired_bytes=max(self.peak_retired_bytes,sum(bytes_held(row) for row in self.retired))

    def _audit_transfer(self,index,hook):
        # Nonclean diagnostic only: all physical flat-prefix words for every dtype, not a pixel/tolerance gate.
        self.capability.compute.wait_event(self.ready[index]);self.torch.npu.synchronize()
        for dtype,master in hook.dtype_cpu_flattened_weights.items():
            view=next(t for t in self.live[index] if t.dtype==dtype)
            flat=view.as_strided((master.numel(),),(1,),storage_offset=0)
            actual=flat.cpu().contiguous()
            equal=bool(self.torch.equal(actual.view(self.torch.uint8),master.view(self.torch.uint8)))
            self.byte_proofs.append({'target_block':index,'dtype':str(dtype),'bytes':master.numel()*master.element_size(),
                'entire_native_flat_master_raw_words_equal':equal})
            assert equal,'Native independent-stream H2D changed physical master bytes'
        self._assert_masters(index,hook)

    def begin_profile(self):
        self._require();assert not self.profile and self.active_owner is None and self.expected_owner==0
        self.capability.drain();self.copy_pairs=[];self.body_pairs=[]
        self.epoch=self.torch.npu.Event(enable_timing=True);self.epoch.record(self.capability.compute);self.profile=True

    def end_profile(self):
        self._require();assert self.profile and self.active_owner is None and self.expected_owner==0
        self.capability.drain();self.profile=False
        copies=[{'target_block':i,'start_ms':self.epoch.elapsed_time(a),'end_ms':self.epoch.elapsed_time(b),'copy_span_ms':a.elapsed_time(b)} for i,(a,b) in self.copy_pairs]
        bodies=[{'block':i,'start_ms':self.epoch.elapsed_time(a),'end_ms':self.epoch.elapsed_time(b),'inclusive_span_ms':a.elapsed_time(b)} for i,(a,b) in self.body_pairs]
        assert sorted(r['target_block'] for r in copies)==list(range(50)) and [r['block'] for r in bodies]==list(range(50))
        from test_worker_stream128 import intersections
        overlap=intersections(copies,bodies)
        if not self.asynchronous:assert overlap<=0.01,'Serial same-stream control unexpectedly overlaps'
        return {'copies':copies,'block_consumer_spans':bodies,'copy_overlap_with_inclusive_block_spans_ms':overlap,
            'not_HBM_or_kernel_busy_time':True,'host_allocator_gaps_in_copy_event_span':True,**self.facts()}

    def facts(self):
        self._require();self._reap()
        return {**self.capability.facts(),'mode':'async' if self.asynchronous else 'serial',
            'only_50_DiT_hooks_modified_encoder_VAE_HCCL_unchanged':True,'native_copy_implementation_preserved':True,
            'prepared_targets':sorted(self.prepared),'complete_50_block_cycles':self.completed_cycles,
            'actual_native_copies':self.copy_count,'actual_native_copy_bytes':self.copy_bytes,
            'peak_retired_blocks':self.peak_retired_blocks,'peak_retired_storage_bytes':self.peak_retired_bytes,
            'retirement_backpressure_count':self.retirement_backpressure_count,
            'retirement_backpressure_host_s':self.retirement_backpressure_host_s,'missing_block_native_sync_recoveries':self.recovered_missing,
            'reusable_or_resident_pool_enabled':False}

    def rollback(self):
        if self.state=='closed':return
        assert self.state=='active' and self.capability is not None
        self.capability.require_owner()
        try:
            self.torch.npu.synchronize();self.capability.drain()
        except BaseException:
            # Do not free retained storages or falsely call an undrained device restored.
            self.state='rollback_failed';raise
        from vllm_omni.diffusion.offloader.tensor_utils import clear_block_storage
        for h,name,method in self.owned:assert getattr(h,name) is method,'Refuse third-party instance-hook overwrite'
        for block in self.blocks:clear_block_storage(dict(block.named_parameters()),dict(block.named_buffers()),None)
        for h,original,attributes,stream in reversed(self.saved):
            for name,value in attributes.items():
                if value is _MISSING:h.__dict__.pop(name,None)
                else:setattr(h,name,value)
            h.copy_stream=stream;h._prefetch_done=None
        # First layer was prefetched by native initialization; restore that actual native state, not just handles.
        self.hooks[0].prefetch_layer(non_blocking=False);self.torch.npu.synchronize()
        assert all(t.numel()>0 for t in self.held_tensors(self.hooks[0]))
        assert all(not h.is_materialized for h in self.hooks if h is not self.hooks[1])
        for index,h in enumerate(self.hooks):self._assert_masters(index,h)
        self.live.clear();self.retired.clear();self.prepared.clear();self.ready.clear()
        self.capability.close();self.state='closed'
