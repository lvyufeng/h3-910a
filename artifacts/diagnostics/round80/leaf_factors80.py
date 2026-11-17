#!/usr/bin/env python3
"""Same-input FP16 LayerNorm stability and Linear bias-rounding factor controls.

CPU nativeFP16 remains the reference. Stable FP32 LayerNorm is a candidate
operator decomposition,not a replacement teacher;native no-bias GEMM and explicit
FP16-bias-after-GEMM controls distinguish an extra rounding from GEMM error.
No production edits,global precision toggles or BF16 requirement.
"""
import hashlib
import json
from pathlib import Path
import signal
import torch
import torch_npu
from torch.nn import functional as F
from evidence_io80 import EvidenceWriter

ROOT=Path('/tmp/h3_round80_leaf_factors01')
writer=EvidenceWriter(ROOT/'factors.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':{},'production_changed':False,'whole_model_precision_accepted':False,'whole_goal_complete':False}


def metric(a,b):
    a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();assert a.shape==b.shape and a.dtype==b.dtype
    e=a.double()-b.double();finite=bool(torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all())
    return {'finite':finite,'raw_bits_equal':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'elements':a.numel(),
            'max_abs_error':float(e.abs().max())if finite else None,'RMSE':float(e.square().mean().sqrt())if finite else None,
            'relative_RMS':float(e.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-30))if finite else None,
            'changed_fraction':float((a!=b).float().mean())if finite else None}


def stable_layernorm(x,gamma,beta,eps=1e-6):
    xf=x.float();shifted=xf-xf[..., :1];centered=shifted-shifted.mean(-1,keepdim=True)
    variance=centered.square().mean(-1,keepdim=True)
    return(centered*torch.rsqrt(variance+eps)*gamma.float()+beta.float()).to(x.dtype)


def case(name,a,b):
    proof=metric(a,b);assert proof['finite'];r['cases'][name]=proof;writer.write(r)
    print(json.dumps({'case':name,**proof}),flush=True)


try:
    signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in()).throw(SystemExit('FP16factors interrupted')))
    torch.set_num_threads(8);torch.npu.set_device(0);torch.manual_seed(801234)
    gamma=(torch.randn(1152)*.1+1).to(torch.float16);beta=(torch.randn(1152)*.1).to(torch.float16)
    x=(torch.randn(32,1152)*3).to(torch.float16)
    x[0].fill_(4096);x[1]=4096+torch.arange(1152,dtype=torch.float32).remainder(7).to(torch.float16)*4
    with torch.inference_mode():
        cpu=F.layer_norm(x,(1152,),gamma,beta,1e-6)
        a=x.to('npu:0');g=gamma.to('npu:0');b=beta.to('npu:0')
        native=F.layer_norm(a,(1152,),g,b,1e-6);stable=stable_layernorm(a,g,b)
        for label,indices in(('constant',[0]),('high_offset_nonconstant',[1]),('ordinary',list(range(2,32)))):
            case('LayerNorm_native_'+label,native[indices],cpu[indices]);case('LayerNorm_stable_'+label,stable[indices],cpu[indices])
        case('LayerNorm_stable_repeat',stable,stable_layernorm(a,g,b))
    torch.manual_seed(80017)
    x=(torch.randn(128,1152)*3).to(torch.float16);w=(torch.randn(3456,1152)*.02).to(torch.float16);b=(torch.randn(3456)*.02).to(torch.float16)
    with torch.inference_mode():
        cpu_with=F.linear(x,w,b);cpu_without=F.linear(x,w)
        cpu_two_rounds=(cpu_without+b).to(torch.float16)
        xn=x.to('npu:0');wn=w.to('npu:0');bn=b.to('npu:0')
        native_with=F.linear(xn,wn,bn);native_without=F.linear(xn,wn)
        explicit_two_rounds=native_without+bn
        case('Linear_no_bias_native_vs_CPU',native_without,cpu_without)
        case('Linear_with_bias_native_vs_CPU',native_with,cpu_with)
        case('Linear_native_with_bias_vs_CPU_two_rounding_factor',native_with,cpu_two_rounds)
        case('Linear_native_with_bias_vs_NPU_two_rounding_factor',native_with,explicit_two_rounds)
        case('CPU_two_rounding_factor_vs_CPU_native_fused_bias',cpu_two_rounds,cpu_with)
        case('Linear_FP32_bias_island_vs_CPU',F.linear(xn.float(),wn.float(),bn.float()).to(torch.float16),cpu_with)
    # Capability probe,not an assumed API promise. Unsupported hardware/types are
    # recorded and are not grounds to replay another command or modify CANN.
    r['grouped_matmul_FP16_inputs_FP32_output']={}
    try:
        with torch.inference_mode():
            result=torch_npu.npu_grouped_matmul([xn],[wn.t().contiguous()],split_item=0,group_type=-1,output_dtype=torch.float32)
        assert len(result)==1
        out=result[0];r['grouped_matmul_FP16_inputs_FP32_output']={'supported':True,'dtype':str(out.dtype),'shape':list(out.shape),'same_input_FP32_CPU':metric(out,(x.float()@w.float().t()))}
    except BaseException as error:r['grouped_matmul_FP16_inputs_FP32_output']={'supported':False,'failure':{'type':type(error).__name__,'message':str(error)}}
    r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
