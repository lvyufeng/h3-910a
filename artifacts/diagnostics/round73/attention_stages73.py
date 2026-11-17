"""Exact vision attention inner-stage wrapper; each operation delegates unchanged.

No per-operator synchronization; parent inclusive intervals are not added to child
sums. to(dtype) and contiguous are separate so copy/packing costs are visible.
Capture is bounded/rank0-only and used only in an excluded first forward.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from event_profile66 import EventProfile


class AttentionStages:
    def __init__(self, encoder, candidate, root, rank, tensor_identity):
        self.root,self.rank,self.identity=root,rank,tensor_identity
        self.profile=EventProfile(torch,capacity=1024)
        self.original=candidate._vision_sdpa_bf16_compatible
        self.original_vision=encoder.vision.forward
        self.encoder,self.candidate=encoder,candidate
        self.call_count=0
        self.capture=False
        self.saved=[]
        self.layouts={}
        def wrapped(q,k,v):
            return self.forward(q,k,v)
        candidate._vision_sdpa_bf16_compatible=wrapped
        def vision(*args,**kwargs):
            return self.profile.call('parent.vision',self.original_vision,*args,**kwargs)
        encoder.vision.forward=vision

    def begin_call(self,profile=False,capture=False):
        self.call_count=0;self.capture=capture
        if profile:self.profile.begin()

    def _to(self,value,dtype):
        return value.to(dtype)

    def _contiguous(self,value):
        return value.contiguous()

    def _inner(self,q,k,v):
        dtype=torch.float32 if q.dtype in (torch.bfloat16,torch.float16) else q.dtype
        values=[]
        for name,value in zip(('q','k','v'),(q,k,v)):
            cast=self.profile.call('child.'+name+'_cast',self._to,value,dtype)
            dense=self.profile.call('child.'+name+'_contiguous',self._contiguous,cast)
            values.append(dense)
        qf,kf,vf=values
        unscaled=self.profile.call('child.qk',torch.matmul,qf,kf.transpose(-2,-1))
        scores=self.profile.call('child.scale',lambda value,scale:value*scale,unscaled,q.shape[-1]**-0.5)
        # Match the original expression's temporary lifetime: do not retain an
        # extra1GiB unscaled score matrix throughout softmax/PV profiling.
        del unscaled
        probabilities=self.profile.call('child.softmax',torch.softmax,scores,dim=-1)
        output=self.profile.call('child.pv',torch.matmul,probabilities,vf)
        return self.profile.call('child.output_cast',self._to,output,q.dtype)

    @staticmethod
    def layout(value):
        return {'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),
                'storage_offset':value.storage_offset(),'storage_bytes':value.untyped_storage().nbytes(),
                'physical_format':torch_npu.get_npu_format(value),'pointer_aligned32':value.data_ptr()%32==0}

    def forward(self,q,k,v):
        if q.device.type not in ('npu','privateuseone'):
            raise ValueError('Vision profile only supports measured NPU path')
        index=self.call_count;self.call_count+=1
        if index in (0,17,33,53) and index not in self.layouts:
            self.layouts[index]={'query':self.layout(q),'key':self.layout(k),'value':self.layout(v)}
        output=self.profile.call('parent.attention',self._inner,q,k,v)
        if self.capture and self.rank==0 and index in (0,17,33,53):
            assert q.shape==k.shape==v.shape==(1,16,4032,72) and q.dtype==k.dtype==v.dtype==torch.bfloat16
            reference=self.original(q,k,v)
            original_equal=torch.equal(output.detach().cpu().contiguous().view(torch.uint8),reference.detach().cpu().contiguous().view(torch.uint8))
            assert original_equal
            payload={name:value.detach().cpu().contiguous().clone() for name,value in zip(('query','key','value','output'),(q,k,v,output))}
            path=self.root/('attention_call%02d.pt'%index)
            with path.open('xb') as f:torch.save(payload,f)
            meta={'call':index,'layer':index//2,'image_segment':index%2,
                  'input_layouts':{name:self.layout(value) for name,value in zip(('query','key','value'),(q,k,v))},
                  'output_layout':self.layout(output),'same_input_original_bits_equal':original_equal,'tensor_identities':self.identity(payload),
                  'file_bytes':path.stat().st_size,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'scope':'Bounded excluded first-call actual attention inputs/native output; saved copies are dense and original strided layout metadata must be reconstructed explicitly for wrapper timing.'}
            assert meta['file_bytes']<64*1024*1024
            assert all(item['finite'] for item in meta['tensor_identities'].values())
            self.saved.append(meta);assert sum(item['file_bytes'] for item in self.saved)<256*1024*1024
            with path.with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
        return output

    def finish(self):
        return self.profile.finish()

    def restore(self):
        self.profile.active=False
        self.candidate._vision_sdpa_bf16_compatible=self.original
        self.encoder.vision.forward=self.original_vision
