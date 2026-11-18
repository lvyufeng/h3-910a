#!/usr/bin/env python3
"""Private gated precise text SDPA using the actually executed R85 batch kernel."""
from pathlib import Path
import json
import hashlib


class AccurateAttention85:
    def __init__(self,candidate,runtime,kernel_dir):
        self.candidate=candidate;self.runtime=runtime;self.kernel_dir=Path(kernel_dir)
        self.original=candidate._scaled_dot_product_attention;self.enabled=False
        self.metas={};self.calls=0;self.kernel_calls=0;self.half_outputs=0
        self.before_half_float_checks=0
        candidate._scaled_dot_product_attention=self.forward
    def prepare(self):
        import batch_gemm85 as kernel
        import fcntl
        assert hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest()=='546cd211ddb3938c3d8dac9d5207a79646c1458a3887df7e8c8562426069854a'
        self.kernel_dir.mkdir(parents=True,exist_ok=True)
        with(self.kernel_dir/'.engine85.lock').open('a')as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            for m in(512,507):
                for k,n in((128,2048),(2048,128)):
                    self.metas[(m,k,n)]=kernel.build_kernel(batch=8,m=m,k=k,n=n,out_dir=self.kernel_dir)
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
        return self.snapshot()
    def set_enabled(self,value):
        if value and len(self.metas)!=4:raise RuntimeError('precise attention used before all4 kernels prepared')
        self.enabled=bool(value)
    def gemm(self,a,packed_b,n):
        import torch
        import batch_gemm85 as kernel
        h,m,k=a.shape;meta=self.metas[(m,k,n)]
        pa=kernel.pack_a(a)
        c=torch.empty(meta['gm_shapes']['c'],dtype=torch.float32,device=a.device)
        aux=[torch.zeros(16,dtype=torch.float16,device=a.device)for _ in range(2)]
        self.runtime.launch(meta,[pa,packed_b,*aux,c],synchronize=True)
        self.kernel_calls+=1
        out=kernel.unpack_c(c,m=m);assert out.dtype==torch.float32
        return out
    def forward(self,q,k,v):
        if not self.enabled:return self.original(q,k,v)
        import torch
        import batch_gemm85 as kernel
        assert q.dtype==k.dtype==v.dtype==torch.float16
        assert q.shape==(1,8,2043,128)and k.shape==(1,1,2043,128)and v.shape==k.shape
        assert q.device.index==self.runtime.device
        self.calls+=1
        qq=q.squeeze(0);kk=k.repeat_interleave(8,1).squeeze(0);vv=v.repeat_interleave(8,1).squeeze(0)
        kp=torch.zeros((8,2048,128),dtype=torch.float16,device=q.device);kp[:,:2043]=kk
        vp=torch.zeros_like(kp);vp[:,:2043]=vv
        # Once per real attention call,not cached across different layers/inputs.
        pk=kernel.pack_w(kp);pv=kernel.pack_w(vp.transpose(-2,-1).contiguous())
        result=torch.empty_like(qq)
        for start in range(0,2043,512):
            end=min(start+512,2043);m=end-start
            score=self.gemm(qq[:,start:end].contiguous(),pk,2048)[:,:,:2043]*(128**-0.5)
            mask=torch.arange(2043,device=q.device)[None,:]>torch.arange(start,end,device=q.device)[:,None]
            p=torch.softmax(score.masked_fill(mask,float('-inf')),-1)
            high=p.to(torch.float16);low=((p-high.float())*1024).to(torch.float16)
            ah=torch.zeros((8,m,2048),dtype=torch.float16,device=q.device);ah[:,:,:2043]=high
            al=torch.zeros_like(ah);al[:,:,:2043]=low
            out=self.gemm(ah,pv,128)+self.gemm(al,pv,128)*(1.0/1024)
            assert out.dtype==torch.float32
            self.before_half_float_checks+=1
            result[:,start:end]=out.to(torch.float16);self.half_outputs+=1
        return result.unsqueeze(0)
    def snapshot(self):
        return {'enabled':self.enabled,'calls':self.calls,'kernel_calls':self.kernel_calls,
                'half_output_chunks':self.half_outputs,'actual_FP32_before_Half_chunks':self.before_half_float_checks,
                'meta_shapes':list(map(str,self.metas)),'no_model_weight_shadow':True,
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def uninstall(self):
        current=self.candidate._scaled_dot_product_attention
        if getattr(current,'__self__',None)is self and getattr(current,'__func__',None)is self.forward.__func__:
            self.candidate._scaled_dot_product_attention=self.original
        else:raise RuntimeError('private attention symbol changed unexpectedly')
        self.enabled=False
