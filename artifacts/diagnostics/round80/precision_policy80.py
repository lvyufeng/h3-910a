#!/usr/bin/env python3
"""Controlled process-local HF32 policy and FP64 diagnostic checks.

This is a private probe process;old policy values are restored in finally.
No model/reference source changes or machine-wide/CANN configuration edits.
The CPU FP16 model remains the precision acceptance reference;FP64 here is a
local diagnostic for interpreting the two implementations,not a new teacher.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from torch.nn import functional as F
from evidence_io80 import EvidenceWriter

ROOT=Path('/tmp/h3_round80_policy01')
writer=EvidenceWriter(ROOT/'policy.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':{},'production_changed':False,'CPU_reference_redefined':False,'whole_model_precision_accepted':False,'whole_goal_complete':False}
old_matmul=None;old_conv=None;have_policy=False


def metric(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double();e=a-b
    return {'finite':bool(torch.isfinite(a).all()and torch.isfinite(b).all()),'max_abs_error':float(e.abs().max()),'RMSE':float(e.square().mean().sqrt()),'relative_RMS':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30))}


try:
    torch.set_num_threads(8);torch.npu.set_device(0);torch.manual_seed(801235)
    try:
        old_matmul=bool(torch.npu.matmul.allow_hf32);old_conv=bool(torch.npu.conv.allow_hf32);have_policy=True
        r['policy_before']={'matmul_allow_hf32':old_matmul,'conv_allow_hf32':old_conv}
    except AttributeError:r['policy_before']={'available':False}
    p=torch.softmax(torch.randn(1,16,256,256)*2,dim=-1);v=torch.randn(1,16,256,72,dtype=torch.float16)
    with torch.inference_mode():
        cpu=p@v.float();high=p.double()@v.double();pn=p.to('npu:0');vn=v.to('npu:0').float()
        case={'CPU_FP32_vs_FP64':metric(cpu,high)}
        for mode in('default','false','true'):
            if mode!='default'and not have_policy:continue
            if mode!='default':torch.npu.matmul.allow_hf32=(mode=='true')
            out=pn@vn;torch.npu.synchronize()
            case[mode]={'actual_matmul_allow_hf32':bool(torch.npu.matmul.allow_hf32)if have_policy else None,'NPU_FP32_vs_CPU_FP32':metric(out,cpu),'NPU_FP32_vs_FP64':metric(out,high)}
        r['cases']['Attention_PV']=case;writer.write(r)
        if have_policy:torch.npu.matmul.allow_hf32=old_matmul
        gamma=(torch.randn(1152)*.1+1).to(torch.float16);beta=(torch.randn(1152)*.1).to(torch.float16)
        x=(torch.randn(32,1152)*3).to(torch.float16);x[0].fill_(4096);x[1]=4096+torch.arange(1152,dtype=torch.float32).remainder(7).to(torch.float16)*4
        cpu=F.layer_norm(x,(1152,),gamma,beta,1e-6)
        high=F.layer_norm(x.double(),(1152,),gamma.double(),beta.double(),1e-6)
        a=x.to('npu:0');g=gamma.to('npu:0');b=beta.to('npu:0')
        native=F.layer_norm(a,(1152,),g,b,1e-6)
        shifted=a.float()-a.float()[..., :1];centered=shifted-shifted.mean(-1,keepdim=True)
        stable=(centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-6)*g.float()+b.float()).to(torch.float16)
        r['cases']['LayerNorm']={label:{'CPU_FP16_vs_FP64':metric(cpu[indices],high[indices]),'NPU_native_vs_FP64':metric(native[indices],high[indices]),'NPU_stable_vs_FP64':metric(stable[indices],high[indices])}for label,indices in(('constant',[0]),('near_constant_high_offset',[1]),('ordinary',list(range(2,32))))}
        r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:
    if have_policy:
        torch.npu.matmul.allow_hf32=old_matmul;torch.npu.conv.allow_hf32=old_conv
        r['policy_restored']={'matmul_allow_hf32':bool(torch.npu.matmul.allow_hf32),'conv_allow_hf32':bool(torch.npu.conv.allow_hf32)}
    writer.write(r)
