#!/usr/bin/env python3
"""Single-patch Conv3d as originalHalf GEMM/Floatbias/oneHalf cast (private only)."""
from pathlib import Path
import hashlib
from types import MethodType


def signature(value):
    return(value.data_ptr(),tuple(value.shape),tuple(value.stride()),value.storage_offset(),value._version,str(value.dtype),str(value.device))


class PatchBiasAdapter89:
    def __init__(self,vision,runtime,kernel_dir):
        import torch
        import vision_bias87 as kernel
        self.module=vision.patch_embed.proj;self.runtime=runtime;self.enabled=False;self.calls=0
        p=self.module
        assert isinstance(p,torch.nn.Conv3d)and p.weight.dtype==p.bias.dtype==torch.float16
        assert tuple(p.weight.shape)==(1152,3,2,16,16)
        assert tuple(p.kernel_size)==tuple(p.stride)==(2,16,16)
        assert tuple(p.padding)==(0,0,0)and tuple(p.dilation)==(1,1,1)and p.groups==1
        self.geometry={'M':8064,'K':1536,'N':1152};self.weight_sig=signature(p.weight);self.bias_sig=signature(p.bias)
        self.meta=kernel.build_kernel(m=8064,k=1536,n=1152,out_dir=Path(kernel_dir))
        assert self.meta['dst_dtype']=='float16'and self.meta['uses_fp32_bias']and self.meta['single_final_half_cast']
        w=torch.zeros((1152,1536),dtype=torch.float16,device=p.weight.device);w[:]=p.weight.detach().reshape(1152,1536)
        self.packed_w=w.reshape(1152,96,16).permute(1,0,2).contiguous()
        self.packed_bias=p.bias.detach().float().contiguous()
        assert torch.equal(self.packed_w.permute(1,0,2).reshape(1152,1536),p.weight.detach().reshape(1152,1536))
        assert torch.equal(self.packed_bias.to(torch.float16),p.bias.detach())
        self.aux=torch.zeros(16,dtype=torch.float16,device=p.weight.device)
        self.original=p.forward;self.had_instance_forward='forward'in p.__dict__
        outer=self
        def forward(module,x):
            if not outer.enabled:return outer.original(x)
            return outer.forward(x)
        self.function=forward;self.wrapper=MethodType(forward,p);p.forward=self.wrapper
    def set_enabled(self,value):
        assert isinstance(value,bool);self.enabled=value
    def forward(self,input_):
        import torch
        assert input_.dtype==torch.float16 and input_.shape==(8064,3,2,16,16)
        assert input_.device.index==self.runtime.device
        assert signature(self.module.weight)==self.weight_sig and signature(self.module.bias)==self.bias_sig
        flat=input_.reshape(8064,1536)
        packed_a=flat.reshape(8064,96,16).permute(1,0,2).contiguous()
        out=torch.empty((72,8064,16),dtype=torch.float16,device=input_.device)
        self.runtime.launch(self.meta,[packed_a,self.packed_w,self.packed_bias,self.aux,out],synchronize=True)
        self.calls+=1
        return out.permute(1,0,2).reshape(8064,1152).contiguous().reshape(8064,1152,1,1,1)
    def packing_control(self,input_):
        import torch
        flat=input_.reshape(8064,1536)
        packed=flat.reshape(8064,96,16).permute(1,0,2).contiguous()
        restored=packed.permute(1,0,2).reshape(8064,1536)
        assert torch.equal(restored,flat)
        return{'original_Half_flatten_prefix_bit_exact':True,'no_M_K_N_padding_needed':True,
               'native_Conv_kernel_equals_stride_and_one_output_voxel':True,'geometry':self.geometry}
    def snapshot(self):
        return{'enabled':self.enabled,'kernel_launches':self.calls,'geometry':self.geometry,
               'packed_Half_weight_bytes':self.packed_w.numel()*self.packed_w.element_size(),
               'Float_bias_bytes':self.packed_bias.numel()*self.packed_bias.element_size(),
               'Float_weight_shadow':False,'one_Half_cast_after_Float_bias':True,
               'no_convolution_parameter_change':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'kernel_name':self.meta['kernel_name'],'kernel_object_SHA256':self.meta['obj_sha256']}
    def uninstall(self):
        current=self.module.forward
        assert getattr(current,'__self__',None)is self.module and getattr(current,'__func__',None)is self.function
        if self.had_instance_forward:self.module.forward=self.original
        else:delattr(self.module,'forward')
        self.enabled=False
