"""R218 same-run profiling: existing computation/checks unchanged; no sync inside hooks.

Native ACL events surround the direct ACL FA launch. Torch events describe inclusive
module stream spans; parent/child spans MUST NOT be summed. H2D spans include queue waits.
"""
from __future__ import annotations
from collections import defaultdict
import ctypes
import hashlib
import inspect
from pathlib import Path
import time
from types import MethodType
import torch

class PathProfile218:
    def __init__(self,model,audit,native_hooks,slots=180):
        from vllm_omni.diffusion.attention.backends import fa910a
        self.model=model; self.audit=audit; self.offloads=native_hooks; self.fa=fa910a
        self.slots=slots; self.events={}; self.counts=defaultdict(int)
        self.host_ms=defaultdict(float); self.pending={}; self.handles=[]; self.saved=[]
        self.shapes=[]; self.launches=[]; self.copy_bytes=[]
        self.original_audit_handles=[];self.new_audit_handles=[]
        self.expected_fa_sha='937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1'
        assert hashlib.sha256(Path(inspect.getfile(fa910a)).read_bytes()).hexdigest()==self.expected_fa_sha
        self.lib=self.fa._lib()
        self.lib.aclrtCreateEventWithFlag.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint32]
        self.lib.aclrtCreateEventWithFlag.restype=ctypes.c_int
        self.lib.aclrtRecordEvent.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        self.lib.aclrtRecordEvent.restype=ctypes.c_int
        self.lib.aclrtEventElapsedTime.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_void_p]
        self.lib.aclrtEventElapsedTime.restype=ctypes.c_int
        self.lib.aclrtDestroyEvent.argtypes=[ctypes.c_void_p]
        self.lib.aclrtDestroyEvent.restype=ctypes.c_int
        self.acl_pairs=[]
        for _ in range(60):
            a=ctypes.c_void_p();b=ctypes.c_void_p()
            self._check(self.lib.aclrtCreateEventWithFlag(ctypes.byref(a),8),'create start event')
            self._check(self.lib.aclrtCreateEventWithFlag(ctypes.byref(b),8),'create end event')
            self.acl_pairs.append((a,b))

    @staticmethod
    def _check(rc,label):
        if rc:raise RuntimeError(f'{label} rc={rc}')

    def _pair(self,label):
        if label not in self.events:
            self.events[label]=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(self.slots)]
        i=self.counts[label]
        assert i<self.slots,label
        return self.events[label][i]

    def _call(self,label,original,args,kwargs,stream=None):
        a,b=self._pair(label)
        a.record(stream) if stream is not None else a.record()
        begin=time.perf_counter();result=original(*args,**kwargs)
        self.host_ms[label]+=1000*(time.perf_counter()-begin)
        b.record(stream) if stream is not None else b.record()
        self.counts[label]+=1
        return result

    def _wrap(self,owner,name,label):
        original=getattr(owner,name)
        def wrapper(*args,**kwargs):
            if label=='FA.wrapper':
                self.shapes.append({'shape':list(args[0].shape),'n_pad':kwargs.get('n_pad',0),'dtype':str(args[0].dtype)})
            return self._call(label,original,args,kwargs)
        setattr(owner,name,wrapper);self.saved.append((owner,name,original,wrapper))

    def _method(self,owner,name,label):
        original=getattr(owner,name)
        def wrapper(instance,*args,**kwargs):return self._call(label,original,args,kwargs)
        bound=MethodType(wrapper,owner)
        setattr(owner,name,bound);self.saved.append((owner,name,original,bound))

    def _module(self,owner,label):
        stack=[]
        def pre(module,inputs):
            a,b=self._pair(label);a.record();stack.append((a,b,time.perf_counter()))
        def post(module,inputs,result):
            a,b,start=stack.pop();b.record()
            self.host_ms[label]+=1000*(time.perf_counter()-start);self.counts[label]+=1
        self.handles.extend([owner.register_forward_pre_hook(pre),owner.register_forward_hook(post)])

    def install(self):
        torch.npu.synchronize()
        # Wrap the same audit closure at its original module location. Count, masks,
        # boundary test, integer host reads and strict failure point are unchanged.
        for handle in self.audit.handles:handle.remove()
        self.audit.handles.clear()
        for name,module in self.model.named_modules():
            if not name.endswith(('.qkv_proj','.mlp.fc1','.adaln_proj.linear')):continue
            original=self.audit._hook(name)
            label='Check.qkv' if name.endswith('.qkv_proj') else ('Check.fc1' if name.endswith('.mlp.fc1') else 'Check.adaln')
            def hooked(module,inputs,result,original=original,label=label):
                return self._call(label,original,(module,inputs,result),{})
            handle=module.register_forward_hook(hooked)
            self.audit.handles.append(handle);self.new_audit_handles.append(handle)
        self._wrap(self.fa,'flash_attention_910a','FA.wrapper')
        self._wrap(self.fa,'_pad_qkv','FA.padding')
        self._wrap(self.fa,'_flash_attention_core','FA.core')
        original_launch=self.lib.fa910a_launch
        def launch(*args):
            i=self.counts['FA.ACL'];assert i<60
            a,b=self.acl_pairs[i]
            stream=args[-1]
            assert stream.value==torch.npu.current_stream().npu_stream
            self._check(self.lib.aclrtRecordEvent(a,stream),'record start')
            rc=original_launch(*args)
            self._check(self.lib.aclrtRecordEvent(b,stream),'record end')
            self._check(rc,'FA launch')
            self.counts['FA.ACL']+=1
            self.launches.append({'symbol':args[1].decode(),'blockDim':int(args[2]),'stream':stream.value,'rc':int(rc)})
            return rc
        self.lib.fa910a_launch=launch;self.saved.append((self.lib,'fa910a_launch',original_launch,launch))
        for block in self.model.blocks:
            self._module(block,'main.block')
            self._module(block.attn,'main.attention')
            self._module(block.mlp,'main.MLP')
            self._module(block.adaln_proj,'main.AdaLN')
            self._module(block.attn.qkv_proj,'linear.qkv')
            self._module(block.attn.out_proj,'linear.out')
            self._module(block.mlp.fc1,'linear.fc1')
            self._module(block.mlp.fc2,'linear.fc2')
            self._module(block.attn.q_norm,'Attention.q_norm')
            self._module(block.attn.k_norm,'Attention.k_norm')
            attention=block.attn.attention
            self._method(attention.parallel_strategy,'pre_attention','USP.pre')
            self._method(attention.parallel_strategy,'post_attention','USP.post')
        for name in ['token_refiner','time_embedder','final_layer','video_patch_proj','audio_patch_proj']:
            module=getattr(self.model,name,None)
            if module is not None:self._module(module,name)
        for offload in self.offloads:
            original=offload.prefetch_layer
            def prefetch(instance,*args,original=original,hook=offload,**kwargs):
                self.copy_bytes.append(sum(t.numel()*t.element_size() for t in hook.dtype_cpu_flattened_weights.values()))
                return self._call('H2D.prefetch_inclusive',original,args,kwargs,stream=hook.copy_stream)
            bound=MethodType(prefetch,offload)
            offload.prefetch_layer=bound;self.saved.append((offload,'prefetch_layer',original,bound))

    def reset(self):
        self.counts.clear();self.host_ms.clear();self.shapes.clear();self.launches.clear();self.copy_bytes.clear()

    def receipt(self):
        rows={}
        for label,n in self.counts.items():
            if label=='FA.ACL':continue
            times=[float(a.elapsed_time(b)) for a,b in self.events[label][:n]]
            rows[label]={'calls':n,'inclusive_stream_total_ms':sum(times),'per_call_ms':times,
                         'host_dispatch_total_ms':self.host_ms[label]}
        times=[]
        for a,b in self.acl_pairs[:self.counts['FA.ACL']]:
            ms=ctypes.c_float()
            self._check(self.lib.aclrtEventElapsedTime(ctypes.byref(ms),a,b),'elapsed kernel event')
            times.append(float(ms.value))
        rows['FA.ACL']={'calls':len(times),'kernel_total_ms':sum(times),'per_call_ms':times,
                        'event_method':'native ACL same stream, no per-hook synchronization'}
        assert len(times)==50
        assert len(self.copy_bytes)==50,len(self.copy_bytes)
        assert self.audit.calls==155
        binaries=[]
        for key,value in self.fa._CACHE.items():
            if isinstance(key,tuple) and len(key)==6 and isinstance(key[0],int):
                obj,symbol,blocks=value
                binaries.append({'path':obj,'symbol':symbol,'blockDim':blocks,
                                  'sha256':hashlib.sha256(Path(obj).read_bytes()).hexdigest()})
        return {'regions':rows,'actual_ACL_launches':self.launches,'runtime_FA_shapes':self.shapes,
                'actual_kernel_binaries':binaries,'prefetch_calls':len(self.copy_bytes),
                'H2D_bytes':sum(self.copy_bytes),'H2D_copy_inclusive_not_pure_transfer':True,
                'parent_child_and_copy_spans_overlap_do_not_sum':True,
                'unchanged_155_strict_checks':self.audit.calls,
                'live_encoder_executed':False,'scope':'isolated DiT replay with TP2+USP4',
                'no_synchronization_added_inside_hooks':True}

    def close(self):
        torch.npu.synchronize()
        for owner,name,original,wrapper in reversed(self.saved):
            assert getattr(owner,name) is wrapper
            setattr(owner,name,original)
        self.saved.clear()
        for h in self.handles:h.remove()
        self.handles.clear()
        for h in self.audit.handles:h.remove()
        self.audit.handles.clear()
        for name,module in self.model.named_modules():
            if name.endswith(('.qkv_proj','.mlp.fc1','.adaln_proj.linear')):
                self.audit.handles.append(module.register_forward_hook(self.audit._hook(name)))
        for a,b in self.acl_pairs:
            self._check(self.lib.aclrtDestroyEvent(a),'destroy start')
            self._check(self.lib.aclrtDestroyEvent(b),'destroy end')
        self.acl_pairs.clear()
