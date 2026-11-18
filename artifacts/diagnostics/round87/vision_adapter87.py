#!/usr/bin/env python3
"""Private vision Linear adapter: original Half weights,FP32 bias,one Half cast."""
from pathlib import Path
import hashlib

SHAPES={(3456,1152):(8064,1152,3456),(1152,1152):(8064,1152,1152),
        (4304,1152):(8064,1152,4352),(1152,4304):(8064,4352,1152),
        (4608,4608):(2016,4608,4608),(5120,4608):(2016,4608,5120)}


def signature(value):
    return(value.data_ptr(),value.storage_offset(),tuple(value.shape),tuple(value.stride()),
           str(value.dtype),str(value.device),value._version)


class VisionBiasAdapter87:
    def __init__(self,candidate,runtime,kernel_dir):
        self.candidate=candidate;self.runtime=runtime;self.kernel_dir=Path(kernel_dir)
        self.cls=candidate._NpuBF16CompatibleLinear;self.original=self.cls.forward
        self.enabled=False;self.entries={};self.metas={};self.calls=0;self.disabled_calls=0
        self.kernel_calls=0;self.control_count=0
        outer=self
        def forward(layer,input_):
            if not outer.enabled:
                outer.disabled_calls+=1
                return outer.original(layer,input_)
            return outer.forward(layer,input_)
        self.wrapper=forward;self.cls.forward=forward
    def prime(self,vision):
        import torch
        import vision_bias87 as kernel
        modules=[(name,module)for name,module in vision.named_modules()if type(module)is self.cls]
        assert len(modules)==116,'actual27blocks*4linears+4mergers*2linears must be116'
        for name,module in modules:
            n,k=module.weight.shape;assert(n,k)in SHAPES,(name,n,k)
            assert module.weight.dtype==module.bias.dtype==torch.float16
            assert module.weight.device.type in('npu','privateuseone')
            m,kp,np=SHAPES[(n,k)];key=(m,kp,np)
            if key not in self.metas:self.metas[key]=kernel.build_kernel(m=m,k=kp,n=np,out_dir=self.kernel_dir)
            meta=self.metas[key];assert meta['dst_dtype']=='float16'and meta['uses_fp32_bias']
            w=torch.zeros((np,kp),dtype=torch.float16,device=module.weight.device);w[:n,:k]=module.weight.detach()
            pw=w.reshape(np,kp//16,16).permute(1,0,2).contiguous()
            pb=torch.zeros(np,dtype=torch.float32,device=w.device);pb[:n]=module.bias.detach().float()
            assert torch.equal(pw.permute(1,0,2).reshape(np,kp)[:n,:k],module.weight.detach())
            assert int(torch.count_nonzero(w[n:]).item())==0 and int(torch.count_nonzero(w[:n,k:]).item())==0
            assert torch.equal(pb[:n].to(torch.float16),module.bias.detach())and int(torch.count_nonzero(pb[n:]).item())==0
            self.entries[id(module)]={'name':name,'m':m,'k':k,'n':n,'kp':kp,'np':np,'meta':meta,'w':pw,'bias':pb,
                                      'weight_sig':signature(module.weight),'bias_sig':signature(module.bias)}
        self.aux=torch.zeros(16,dtype=torch.float16,device=next(vision.parameters()).device)
        return self.snapshot()
    def set_enabled(self,value):
        if value:assert len(self.entries)==116,'vision bias adapter requires all original116weights primed'
        self.enabled=bool(value)
    def pack_input(self,entry,input_):
        import torch
        assert input_.dtype==torch.float16 and input_.shape==(entry['m'],entry['k'])
        assert input_.device.index==self.runtime.device
        mp=((entry['m']+127)//128)*128
        flat=torch.zeros((mp,entry['kp']),dtype=torch.float16,device=input_.device)
        flat[:entry['m'],:entry['k']]=input_
        return flat.reshape(mp,entry['kp']//16,16).permute(1,0,2).contiguous()
    def forward(self,layer,input_):
        import torch
        self.calls+=1;entry=self.entries[id(layer)]
        assert signature(layer.weight)==entry['weight_sig']and signature(layer.bias)==entry['bias_sig']
        a=self.pack_input(entry,input_);mp=((entry['m']+127)//128)*128
        out=torch.empty((entry['np']//16,mp,16),dtype=torch.float16,device=input_.device)
        self.runtime.launch(entry['meta'],[a,entry['w'],entry['bias'],self.aux,out],synchronize=True)
        self.kernel_calls+=1
        nd=out.permute(1,0,2).reshape(mp,entry['np'])
        return nd[:entry['m'],:entry['n']].contiguous()
    def controls(self,layer,input_):
        import torch
        entry=self.entries[id(layer)];a=self.pack_input(entry,input_)
        mp=((entry['m']+127)//128)*128
        back=a.permute(1,0,2).reshape(mp,entry['kp'])
        prefix=bool(torch.equal(back[:entry['m'],:entry['k']],input_))
        tail=int(torch.count_nonzero(back[entry['m']:]).item())+int(torch.count_nonzero(back[:entry['m'],entry['k']:]).item())
        assert prefix and tail==0
        self.control_count+=1
        return{'original_prefix_bit_exact':prefix,'zero_only_M_K_padding':tail==0,
               'geometry':{k:entry[k]for k in('m','k','n','kp','np')}}
    def snapshot(self):
        return{'enabled':self.enabled,'cached_weights':len(self.entries),'kernel_shapes':list(map(str,self.metas)),
               'calls':self.calls,'kernel_launches':self.kernel_calls,'padding_controls':self.control_count,
               'packed_Half_weight_bytes':sum(e['w'].numel()*e['w'].element_size()for e in self.entries.values()),
               'FP32_bias_cache_bytes':sum(e['bias'].numel()*e['bias'].element_size()for e in self.entries.values()),
               'FP32_weight_shadows':0,'one_Half_cast_inside_kernel_after_FP32_bias':True,
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def uninstall(self):
        assert self.cls.forward is self.wrapper
        self.cls.forward=self.original;self.enabled=False;self.entries.clear()
