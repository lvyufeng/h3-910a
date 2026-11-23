"""R126 actual FA launch/USP profiling; no kernel, values, mask or plan changes.

Events and counters exist only in a separate instrumented forward. The actual
ACL launcher is counted, not just backend selection. Parent spans are inclusive;
never sum nested FA-wrapper/core/kernel spans. No hook synchronizes the device.
"""
from __future__ import annotations
from collections import defaultdict
from types import MethodType
import hashlib
import inspect
from pathlib import Path
import time
import torch


class ActualAttentionProfile126:
    def __init__(self,model,slots=160):
        from vllm_omni.diffusion.attention.backends import fa910a
        self.fa=fa910a
        self.model=model
        self.slots=slots
        self.counts=defaultdict(int)
        self.events={}
        self.host_ms=defaultdict(float)
        self.saved=[]
        self.handles=[]
        self.shapes=[]
        self.launches=[]
        labels=['FA.wrapper','FA.padding','FA.core','FA.ACL_launch','USP.pre_attention','USP.post_attention',
                'Attention.local_backend','Attention.q_norm','Attention.k_norm']
        for label in labels:
            self.events[label]=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(slots)]
        self.expected_fa_sha='937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1'
        assert hashlib.sha256(Path(inspect.getfile(fa910a)).read_bytes()).hexdigest()==self.expected_fa_sha
        assert not fa910a._SYNC_EACH_CALL

    def _call(self,label,original,args,kwargs):
        index=self.counts[label];assert index<self.slots
        self.events[label][index][0].record()
        started=time.perf_counter()
        result=original(*args,**kwargs)
        self.host_ms[label]+=(time.perf_counter()-started)*1000
        self.events[label][index][1].record()
        self.counts[label]+=1
        return result

    def _function(self,owner,name,label):
        original=getattr(owner,name)
        def call(*args,**kwargs):
            if label=='FA.wrapper':
                q=args[0]
                self.shapes.append({'q_shape_BHND':list(q.shape),'dtype':str(q.dtype),'device':str(q.device),
                    'q_stride':list(q.stride()),'n_pad':int(kwargs.get('n_pad',0)),
                    'actual_FA_function_called':True})
            if label=='FA.ACL_launch':
                ptr=args[-1].value
                current=torch.npu.current_stream().npu_stream
                assert ptr==current,'FA launch is not on the profiled torch compute stream'
                self.launches.append({'kernel_symbol':args[1].decode(),'blockDim':int(args[2]),
                                      'ACL_stream':str(ptr)})
            result=self._call(label,original,args,kwargs)
            if label=='FA.ACL_launch':
                assert result==0
                self.launches[-1]['return_code']=int(result)
            return result
        setattr(owner,name,call)
        self.saved.append((owner,name,original,call))

    def _method(self,owner,name,label):
        original=getattr(owner,name)
        def call(instance,*args,**kwargs):return self._call(label,original,args,kwargs)
        bound=MethodType(call,owner)
        setattr(owner,name,bound)
        self.saved.append((owner,name,original,bound))

    def _module(self,module,label):
        stack=[]
        def pre(owner,inputs):
            index=self.counts[label];assert index<self.slots
            self.events[label][index][0].record();stack.append((index,time.perf_counter()))
        def post(owner,inputs,output):
            index,t=stack.pop();self.events[label][index][1].record()
            self.host_ms[label]+=(time.perf_counter()-t)*1000;self.counts[label]+=1
        self.handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])

    def install(self):
        torch.npu.synchronize()
        self._function(self.fa,'flash_attention_910a','FA.wrapper')
        self._function(self.fa,'_pad_qkv','FA.padding')
        self._function(self.fa,'_flash_attention_core','FA.core')
        lib=self.fa._lib();self._function(lib,'fa910a_launch','FA.ACL_launch')
        for block in self.model.blocks:
            attention=block.attn.attention
            self._method(attention,'_run_local_attention','Attention.local_backend')
            strategy=attention.parallel_strategy
            self._method(strategy,'pre_attention','USP.pre_attention')
            self._method(strategy,'post_attention','USP.post_attention')
            self._module(block.attn.q_norm,'Attention.q_norm')
            self._module(block.attn.k_norm,'Attention.k_norm')

    def reset(self):
        self.counts.clear();self.host_ms.clear();self.shapes.clear();self.launches.clear()

    def receipt(self):
        rows={}
        for label,n in self.counts.items():
            times=[a.elapsed_time(b) for a,b in self.events[label][:n]]
            rows[label]={'calls':n,'inclusive_device_total_ms':sum(times),'inclusive_device_mean_ms':sum(times)/n,
                         'host_dispatch_total_ms':self.host_ms[label]}
        assert self.counts['FA.ACL_launch']==50 and self.counts['FA.wrapper']==50
        assert self.counts['USP.pre_attention']==50 and self.counts['USP.post_attention']==50
        assert all(x['q_shape_BHND']==[1,7,41792,128] and x['n_pad']==23 for x in self.shapes)
        binaries=[]
        for key,value in self.fa._CACHE.items():
            if len(key)==6 and isinstance(key[0],int):
                obj,symbol,blocks=value
                binaries.append({'cache_key':list(key),'object_path':obj,'symbol':symbol,'blockDim':blocks,
                                 'object_sha256':hashlib.sha256(Path(obj).read_bytes()).hexdigest()})
        return {'regions':rows,'actual_ACL_launches':self.launches,'FA_runtime_shapes':self.shapes,
                'actual_kernel_binaries':binaries,'fa_python_sha256':self.expected_fa_sha,
                'per_module_host_synchronization_added':False,
                'all_50_main_attention_calls_hit_real_FA':True,
                'parent_child_regions_are_inclusive_do_not_sum':True,
                'kernel_span_uses_torch_events_on_the_same_ACL_stream':True}

    def close(self):
        torch.npu.synchronize()
        for owner,name,original,wrapper in reversed(self.saved):
            assert getattr(owner,name) is wrapper
            setattr(owner,name,original)
        self.saved.clear()
        for handle in self.handles:handle.remove()
        self.handles.clear()
