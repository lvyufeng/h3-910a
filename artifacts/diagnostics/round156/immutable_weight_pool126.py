"""R126 private immutable-weight pool, attached only to benchmark hook instances.

No global offloader patch and no model arithmetic change. Native pinned CPU
masters and exact tensor metadata are reused. Each reusable device slot has an
explicit compute-complete event; the copy stream waits on it before overwriting
that slot. Ready events keep the native next-layer dependency. Buffers live until
all streams are drained on uninstall. Resident blocks get distinct storage.
"""
from __future__ import annotations
from types import MethodType
import hashlib
import json
import torch
from vllm_omni.platforms import current_omni_platform as platform
from vllm_omni.diffusion.offloader.tensor_utils import clear_block_storage, set_tensor_storage


class ImmutableWeightPool126:
    def __init__(self, blocks, hooks, *, resident_blocks=0, asynchronous=True, slots=2):
        self.blocks = list(blocks)
        self.hooks = list(hooks)
        assert len(self.blocks) == len(self.hooks) == 50
        assert 0 <= resident_blocks <= 50 and slots == 2
        self.resident_blocks = resident_blocks
        self.asynchronous = asynchronous
        self.device = hooks[0].device
        self.compute_stream = platform.current_stream()
        self.copy_stream = torch.npu.Stream() if asynchronous else self.compute_stream
        self.saved = []
        self.buffers = {}
        self.ready = {}
        self.prepared = set()
        self.consumer_fences = {}
        self.expected_next_owner = 0
        self.active_owner = None
        self.consumed = {}
        self.slot_consumed = {}
        self.target_index = {id(h.next_block): i for i,h in enumerate(self.hooks)}
        # hooks are [owner49->target0, owner0->target1, ..., owner48->target49].
        assert [id(h.next_block) for h in self.hooks] == [id(b) for b in self.blocks]
        self.owner_index = {id(h): (i-1) % 50 for i,h in enumerate(self.hooks)}
        self.copy_count = 0
        self.copy_bytes = 0
        self.resident_skips = 0
        self.profile = False
        self.profile_pairs = {}
        self.compute_pairs = {}
        self.profile_rows = []
        self.compute_rows = []
        self.epoch = None
        self.active = False
        self.install_started = False
        self.owned_methods = []
        self.last_target = {}
        self.master_versions = [{dtype: t._version for dtype,t in h.dtype_cpu_flattened_weights.items()}
                                for h in self.hooks]
        self.master_signatures = [{dtype:self._master_signature(t) for dtype,t in h.dtype_cpu_flattened_weights.items()}
                                  for h in self.hooks]
        self.metadata_sha256 = [hashlib.sha256(json.dumps({str(k):v for k,v in h.dtype_metadata.items()},sort_keys=True,default=str).encode()).hexdigest()
                                for h in self.hooks]

    @staticmethod
    def _master_signature(t):
        return (id(t),str(t.device),str(t.dtype),tuple(t.shape),tuple(t.stride()),t.storage_offset(),
                t.data_ptr(),t.untyped_storage()._cdata,t._version,bool(t.is_pinned()))

    def _slot(self, index):
        return ('resident', index) if index < self.resident_blocks else ('ring', index % 2)

    def install(self):
        assert not self.install_started and not self.active
        self.install_started = True
        try:
            return self._install_impl()
        except BaseException:
            self.uninstall()
            raise

    def _install_impl(self):
        assert not self.active
        torch.npu.synchronize()
        for b,h in zip(self.blocks,self.hooks):
            clear_block_storage(dict(b.named_parameters()),dict(b.named_buffers()),None)
            h._prefetch_done = None
        capacities = {}
        for i,h in enumerate(self.hooks):
            key=self._slot(i)
            for dtype,master in h.dtype_cpu_flattened_weights.items():
                capacities[(key,dtype)] = max(capacities.get((key,dtype),0),master.numel())
        for key,numel in capacities.items():
            self.buffers[key] = torch.empty(numel,dtype=key[1],device=self.device)
        for i,h in enumerate(self.hooks):
            self.ready[i] = torch.npu.Event()
            self.consumer_fences[i] = torch.npu.Event()
            self.consumed[i] = None
            self.profile_pairs[i] = (torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True))
            self.compute_pairs[(i-1) % 50] = (torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True))
            old={name:getattr(h,name) for name in ('prefetch_layer','offload_layer','pre_forward','post_forward')}
            self.saved.append((h,old))
            def prefetch(hook,non_blocking=True,*,index=i):
                return self._prefetch(index,hook,non_blocking)
            def offload(hook,*,owner=(i-1) % 50):
                return self._offload(owner,hook)
            def pre(hook,module,*args,original=old['pre_forward'],owner=(i-1) % 50,**kwargs):
                assert self.active_owner is None and owner==self.expected_next_owner, 'Pool permits only complete sequential 50-block execution'
                self.active_owner=owner
                if owner not in self.prepared or not hook.is_materialized:
                    hook._prev_hook.prefetch_layer(non_blocking=False)
                # Explicit weight-ready dependency even when ring traversal is
                # interrupted; it is NOT the slot's compute-complete fence.
                self.compute_stream.wait_event(self.ready[owner])
                result=original(module,*args,**kwargs)
                # Body span starts AFTER next-weight staging dispatch. A serial
                # copy therefore cannot masquerade as overlapping this span.
                if self.profile:self.compute_pairs[owner][0].record(self.compute_stream)
                return result
            def post(hook,module,output,original=old['post_forward'],owner=(i-1) % 50):
                assert self.active_owner==owner
                if self.profile:
                    self.compute_pairs[owner][1].record(self.compute_stream)
                    self.compute_rows.append(owner)
                result=original(module,output)
                self.active_owner=None;self.expected_next_owner=(owner+1) % 50
                return result
            replacements={'prefetch_layer':MethodType(prefetch,h),'offload_layer':MethodType(offload,h),
                          'pre_forward':MethodType(pre,h),'post_forward':MethodType(post,h)}
            for name,method in replacements.items():
                setattr(h,name,method)
                self.owned_methods.append((h,name,method))
        self.active=True
        for i in range(self.resident_blocks):self.hooks[i].prefetch_layer(non_blocking=False)
        self.hooks[0].prefetch_layer(non_blocking=False)
        torch.npu.synchronize()

    def _prefetch(self,index,hook,non_blocking):
        assert self.active
        for dtype,master in hook.dtype_cpu_flattened_weights.items():
            assert self._master_signature(master)==self.master_signatures[index][dtype], 'Immutable host master rebound or mutated'
            assert master._version == self.master_versions[index][dtype], 'Immutable host master changed'
            assert master.device.type == 'cpu' and master.is_pinned()
        current=platform.current_stream()
        assert current == self.compute_stream, 'Benchmark changed compute stream ownership'
        if index < self.resident_blocks and self.consumed[index] is not None:
            assert index in self.prepared, 'Resident cache lost its generation'
            assert all(t.numel()>0 for t in hook.next_block_parameters.values()), 'External resident offload is unsupported'
            hook._prefetch_done = self.ready[index]
            self.resident_skips += 1
            return
        self.copy_stream.wait_stream(current)
        key=self._slot(index)
        # Every previous consumer of this fixed slot must have finished first.
        with platform.stream(self.copy_stream):
            event=self.slot_consumed.get(key)
            if event is not None:self.copy_stream.wait_event(event)
            if self.profile:self.profile_pairs[index][0].record(self.copy_stream)
            for dtype,master in hook.dtype_cpu_flattened_weights.items():
                self.buffers[(key,dtype)][:master.numel()].copy_(master,non_blocking=non_blocking)
                self.copy_bytes += master.numel()*master.element_size()
            if self.profile:self.profile_pairs[index][1].record(self.copy_stream)
            self.ready[index].record(self.copy_stream)
        for dtype,metadata_rows in hook.dtype_metadata.items():
            storage=self.buffers[(key,dtype)]
            for metadata in metadata_rows:
                assert 0<=metadata['offset'] and metadata['offset']+metadata['numel']<=hook.dtype_cpu_flattened_weights[dtype].numel()<=storage.numel()
                name=metadata['name']
                target=hook.next_block_parameters.get(name,hook.next_block_buffers.get(name))
                assert target is not None
                value=torch.as_strided(storage[metadata['offset']:metadata['offset']+metadata['numel']],
                                       size=metadata['shape'],stride=metadata['stride'])
                set_tensor_storage(target,value)
                assert target.dtype==dtype and target.device==self.device and target.stride()==metadata['stride']
        hook._prefetch_done=self.ready[index]
        self.last_target[key]=index
        self.prepared.add(index)
        self.copy_count += 1
        if self.profile:self.profile_rows.append(index)
        if index < self.resident_blocks:
            # Non-None also serves as the resident materialization marker. It is
            # recorded again after actual compute before any teardown.
            event=torch.npu.Event();event.record(self.copy_stream);self.consumed[index]=event

    def _offload(self,index,hook):
        # Preserve native wait for the NEXT block before scheduling its compute.
        if hook._prefetch_done is not None:self.compute_stream.wait_event(hook._prefetch_done)
        event=self.consumer_fences[index]
        event.record(self.compute_stream)
        self.consumed[index]=event
        self.slot_consumed[self._slot(index)]=event
        if index >= self.resident_blocks:
            clear_block_storage(hook.block_parameters,hook.block_buffers,None)
            self.prepared.discard(index)
        hook._prefetch_done=None

    def begin_profile(self):
        torch.npu.synchronize()
        self.profile_rows.clear();self.compute_rows.clear()
        self.epoch=torch.npu.Event(enable_timing=True);self.epoch.record(self.compute_stream)
        self.profile=True

    def end_profile(self):
        assert self.profile
        torch.npu.synchronize()
        self.profile=False
        copies=[];compute=[]
        for i in self.profile_rows:
            a,b=self.profile_pairs[i]
            copies.append({'target_block':i,'start_ms':self.epoch.elapsed_time(a),
                           'end_ms':self.epoch.elapsed_time(b),'copy_ms':a.elapsed_time(b)})
        for i in self.compute_rows:
            a,b=self.compute_pairs[i]
            compute.append({'block':i,'start_ms':self.epoch.elapsed_time(a),
                            'end_ms':self.epoch.elapsed_time(b),'span_ms':a.elapsed_time(b)})
        # Each measured forward records each target at most once. Block0's
        # initial resident contents were loaded before the profile epoch.
        assert len(self.profile_rows)==len(set(self.profile_rows))
        assert self.compute_rows==list(range(50)), 'Hook registry did not execute every measured block wrapper'
        assert self.active_owner is None and self.expected_next_owner==0
        overlap=0.0
        for c in copies:
            intersections=[]
            for k in compute:
                lo=max(c['start_ms'],k['start_ms']);hi=min(c['end_ms'],k['end_ms'])
                if hi>lo:intersections.append((lo,hi))
            intersections.sort();merged=[]
            for lo,hi in intersections:
                if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
                else:merged.append((lo,hi))
            overlap+=sum(hi-lo for lo,hi in merged)
        total=sum(c['copy_ms'] for c in copies)
        return {'copies':copies,'block_device_spans':compute,'total_H2D_device_ms':total,
                'copy_ms_overlapping_block_spans':overlap,
                'overlap_fraction':overlap/total if total else None,
                'compute_span_is_inclusive_not_kernel_busy_time':True,
                'profile_has_no_module_host_sync_or_activation_copies':True,
                'compute_stream':str(self.compute_stream.npu_stream),
                'copy_stream':str(self.copy_stream.npu_stream),
                'streams_independent':self.compute_stream!=self.copy_stream}

    def audit_pool_bytes(self):
        """Nonclean full-byte proof of every currently held resident/ring prefix."""
        torch.npu.synchronize()
        rows=[]
        for key,index in self.last_target.items():
            hook=self.hooks[index]
            for dtype,master in hook.dtype_cpu_flattened_weights.items():
                gpu=self.buffers[(key,dtype)][:master.numel()].cpu().contiguous()
                same=bool(torch.equal(gpu.view(torch.uint8),master.view(torch.uint8)))
                rows.append({'target_block':index,'dtype':str(dtype),'bytes':master.numel()*master.element_size(),
                             'whole_storage_bytes_equal':same})
                del gpu
        now=[hashlib.sha256(json.dumps({str(k):v for k,v in h.dtype_metadata.items()},sort_keys=True,default=str).encode()).hexdigest()
             for h in self.hooks]
        assert now==self.metadata_sha256,'Offloader metadata changed'
        assert all(row['whole_storage_bytes_equal'] for row in rows),'Resident buffer mutated or H2D bytes changed'
        return {'held_buffers':rows,'metadata_sha256_unchanged':True,'profile_instrumented_not_clean':True,
                'all_held_storage_bytes_equal':True,'comparison_is_not_CPU_model_inference':True}

    def snapshot(self):
        return {'resident_blocks':self.resident_blocks,'asynchronous':self.asynchronous,
                'device_pool_bytes':sum(t.numel()*t.element_size() for t in self.buffers.values()),
                'H2D_layer_copy_calls':self.copy_count,'H2D_bytes':self.copy_bytes,
                'resident_prefetch_skips':self.resident_skips,'slot_reuse_waits_for_compute_event':True,
                'original_immutable_pinned_masters_reused':True,'D2H_weight_copies':False,
                'model_arithmetic_modified':False}

    def uninstall(self):
        if not self.install_started:return
        torch.npu.synchronize()
        for h,name,method in self.owned_methods:
            assert getattr(h,name) is method, 'Refuse to overwrite a third-party hook change'
        for b in self.blocks:
            clear_block_storage(dict(b.named_parameters()),dict(b.named_buffers()),None)
        for h,old in reversed(self.saved):
            for name,value in old.items():setattr(h,name,value)
            h._prefetch_done=None
        self.saved.clear();self.owned_methods.clear();self.buffers.clear();self.consumed.clear();self.slot_consumed.clear()
        self.prepared.clear();self.consumer_fences.clear();self.last_target.clear()
        self.active=False;self.install_started=False
        self.hooks[0].prefetch_layer(non_blocking=False)
        torch.npu.synchronize()
