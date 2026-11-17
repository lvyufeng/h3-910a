#!/usr/bin/env python3
"""Same-input CPU FP16 vs910A FP16 leaf-operator controls,not a model gold/pass.

Keep CPU nativeFP16 operators as references. Extra explicitFP32-on-NPU cases
are candidate factor controls,not changes to the teacher or claims that all
model arithmetic should be FP32. No production or global precision-policy edit.
"""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import torch_npu
from torch.nn import functional as F
from evidence_io80 import EvidenceWriter

ROOT=Path('/tmp/h3_round80_leaf_ops01')
writer=EvidenceWriter(ROOT/'operators.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':{},
   'torch':torch.__version__,'NPU_device':'npu:0','BF16_reference_required':False,'original_CPU_model_job_replayed':False,
   'production_changed':False,'whole_model_precision_accepted':False,'whole_goal_complete':False}


def info(x):
    value=x.detach().cpu().contiguous()
    return {'shape':list(value.shape),'dtype':str(value.dtype),'finite':bool(torch.isfinite(value.float()).all()),
            'max_abs':float(value.float().abs().max()),'sha256':hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()}


def metric(a,b):
    a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();assert a.shape==b.shape and a.dtype==b.dtype
    finite=bool(torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all());e=a.double()-b.double()
    return {'raw_bits_equal':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'finite':finite,'elements':a.numel(),
            'max_abs_error':float(e.abs().max())if finite else None,'RMSE':float(e.square().mean().sqrt())if finite else None,
            'relative_RMS':float(e.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-30))if finite else None,
            'changed_fraction':float((a!=b).float().mean())if finite else None}


def run_case(name,cpu_func,npu_func,args):
    started=time.perf_counter();cpu_args=[x.detach().cpu()for x in args]
    with torch.inference_mode():reference=cpu_func(*cpu_args)
    npu_args=[x.to('npu:0')for x in cpu_args]
    row={'inputs':[info(x)for x in cpu_args],'CPU_reference_output':info(reference)}
    with torch.inference_mode():one=npu_func(*npu_args);two=npu_func(*npu_args)
    torch.npu.synchronize();row['same_input_CPU_comparison']=metric(one,reference);row['NPU_repeat']=metric(two,one)
    row['seconds_excluded_performance']=time.perf_counter()-started
    assert row['same_input_CPU_comparison']['finite']and row['NPU_repeat']['raw_bits_equal']
    r['cases'][name]=row;writer.write(r)
    print(json.dumps({'case':name,**row['same_input_CPU_comparison']}),flush=True)
    return reference,one


try:
    signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in()).throw(SystemExit('LeafFP16probe interrupted')))
    torch.set_num_threads(8);torch.npu.set_device(0);torch.manual_seed(801234)
    x=torch.cat((torch.linspace(-12,12,24577),torch.tensor([-60000.,-1000.,-100.,-20.,-6.,-1.,0.,1.,6.,20.,100.,1000.,60000.]))).to(torch.float16)
    run_case('GELU_tanh_native_FP16',lambda x:F.gelu(x,approximate='tanh'),lambda x:F.gelu(x,approximate='tanh'),[x])
    run_case('GELU_tanh_explicit_NPU_FP32_island',lambda x:F.gelu(x,approximate='tanh'),lambda x:F.gelu(x.float(),approximate='tanh').to(torch.float16),[x])
    run_case('GELU_none_native_FP16',lambda x:F.gelu(x,approximate='none'),lambda x:F.gelu(x,approximate='none'),[x])
    run_case('SiLU_native_FP16',F.silu,F.silu,[x])
    up=torch.linspace(-1,1,x.numel()).to(torch.float16)
    run_case('SwiGLU_native_two_FP16_roundings',lambda x,u:F.silu(x)*u,lambda x,u:F.silu(x)*u,[x,up])
    gamma=(torch.randn(1152)*.1+1).to(torch.float16);beta=(torch.randn(1152)*.1).to(torch.float16)
    norm_x=(torch.randn(32,1152)*3).to(torch.float16)
    norm_x[0].fill_(4096);norm_x[1]=4096+torch.arange(1152,dtype=torch.float32).remainder(7).to(torch.float16)*4
    run_case('LayerNorm_native_FP16_eps1e6',lambda x,w,b:F.layer_norm(x,(1152,),w,b,1e-6),lambda x,w,b:F.layer_norm(x,(1152,),w,b,1e-6),[norm_x,gamma,beta])
    run_case('LayerNorm_explicit_NPU_FP32_island',lambda x,w,b:F.layer_norm(x,(1152,),w,b,1e-6),lambda x,w,b:F.layer_norm(x.float(),(1152,),w.float(),b.float(),1e-6).to(torch.float16),[norm_x,gamma,beta])
    linear_x=(torch.randn(128,1152)*3).to(torch.float16);weight=(torch.randn(3456,1152)*.02).to(torch.float16);bias=(torch.randn(3456)*.02).to(torch.float16)
    run_case('Vision_QKV_Linear_native_FP16_M128',F.linear,F.linear,[linear_x,weight,bias])
    run_case('Vision_QKV_Linear_explicit_NPU_FP32_island',F.linear,lambda x,w,b:F.linear(x.float(),w.float(),b.float()).to(torch.float16),[linear_x,weight,bias])
    conv_x=(torch.randn(8,3,2,16,16)*2).to(torch.float16);conv_w=(torch.randn(1152,3,2,16,16)*.01).to(torch.float16);conv_b=(torch.randn(1152)*.01).to(torch.float16)
    run_case('Patch_Conv3d_native_FP16',lambda x,w,b:F.conv3d(x,w,b,stride=(2,16,16)),lambda x,w,b:F.conv3d(x,w,b,stride=(2,16,16)),[conv_x,conv_w,conv_b])
    q=torch.randn(1,16,256,72,dtype=torch.float16);k=torch.randn_like(q);v=torch.randn_like(q)
    run_case('Attention_QK_FP32_from_identical_FP16_inputs',lambda q,k:q.float()@k.float().transpose(-2,-1),lambda q,k:q.float()@k.float().transpose(-2,-1),[q,k])
    p=torch.softmax(torch.randn(1,16,256,256)*2,dim=-1)
    run_case('Attention_PV_FP32_probabilities_and_FP16_values',lambda p,v:p@v.float(),lambda p,v:p@v.float(),[p,v])
    r['available_matmul_related_NPU_schemas']={}
    for name in('npu_bmmV2','npu_matmul','npu_linear','npu_bmm','npu_grouped_matmul'):
        try:
            obj=getattr(torch.ops.npu,name);r['available_matmul_related_NPU_schemas'][name]={key:str(schema)for key,schema in obj._schemas.items()}
        except(AttributeError,RuntimeError):r['available_matmul_related_NPU_schemas'][name]=None
    r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
