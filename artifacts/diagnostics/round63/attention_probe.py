#!/usr/bin/env python3
"""Actual VAE head64 noncausal SDPA contracts/layout/FP32 eager comparisons."""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from torch.nn import functional as F

ROOT=Path('/tmp/h3_round63_decode01');CAP=ROOT/'capture'/'rank0';OUT=ROOT/'attention_probe.json'
if OUT.exists():raise RuntimeError('Refuse actual VAE attention evidence overwrite')
torch.npu.set_device(0);torch.set_num_threads(8)
r={'cases':{},'scope':'Actual first decoder-tile/layer0 FP16 QKV,32heads,1797keys,D64,noncausal/no mask. Native NPU output, not CPU/CUDA or full-video quality reference.'}


def metric(a,b):
    x,y=a.detach().cpu().contiguous(),b.detach().cpu().contiguous();e=x.float().double()-y.float().double()
    return {'raw_bits_equal':torch.equal(x.view(torch.uint8),y.view(torch.uint8)), 'finite':bool(torch.isfinite(x.float()).all() and torch.isfinite(y.float()).all()),
            'elements':x.numel(),'word_mismatches':int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),
            'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.float().double().square().mean().sqrt().clamp_min(1e-30))}


def sanitize(x):return torch.where(torch.isfinite(x),x,torch.zeros((),dtype=x.dtype,device=x.device))


def native(q,k,v,layout='original'):
    if layout=='bsnd_contiguous':q,k,v=[x.contiguous() for x in (q,k,v)]
    q,k,v=[x.transpose(1,2) for x in (q,k,v)]
    if layout=='bhsd_contiguous':q,k,v=[x.contiguous() for x in (q,k,v)]
    return sanitize(F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False).transpose(1,2))


def explicit(q,k,v,scale_kind='post',head_chunk=32):
    with torch.autocast('npu',enabled=False):
        qq,kk,vv=[x.transpose(1,2).float().contiguous() for x in (q,k,v)]
        if scale_kind=='split':
            factor=q.shape[-1]**-.25;qq=qq*factor;kk=kk*factor
        ys=[]
        for start in range(0,q.shape[2],head_chunk):
            a,b,c=[x[:,start:start+head_chunk] for x in (qq,kk,vv)]
            scores=a@b.transpose(-2,-1)
            if scale_kind=='post':scores=scores*(q.shape[-1]**-.5)
            p=torch.softmax(scores,dim=-1)
            ys.append(p@c)
        return sanitize(torch.cat(ys,dim=1).transpose(1,2).to(q.dtype))


def timed(fn):
    fn();torch.npu.synchronize();groups=[]
    for _ in range(3):
        t=time.perf_counter()
        for _ in range(5):out=fn()
        torch.npu.synchronize();groups.append((time.perf_counter()-t)*1000/5)
    return out,{'groups_ms':groups,'median_ms':statistics.median(groups)}


try:
    d=torch.load(CAP/'attention0_inputs.pt',weights_only=True)
    q,k,v=[d[x].to('npu').contiguous() for x in ('q','k','v')]
    # Reconstruct actual V's interleaved QKV backing, not an automatically
    # dense downloaded snapshot, and prove its values survive reconstruction.
    backing=torch.zeros(*v.shape[:-1],3*v.shape[-1],dtype=v.dtype,device=v.device)
    vv=backing[...,2*v.shape[-1]:];vv.copy_(v);v=vv
    actual=torch.load(CAP/'attention0_output.pt',weights_only=True)
    r['input_meta']={key:{'shape':list(value.shape),'dtype':str(value.dtype),'stride':list(value.stride()),'format':torch_npu.get_npu_format(value)} for key,value in zip(('q','k','v'),(q,k,v))}
    functions={'native_original':lambda:native(q,k,v), 'native_bsnd_contiguous':lambda:native(q,k,v,'bsnd_contiguous'),
               'native_bhsd_contiguous':lambda:native(q,k,v,'bhsd_contiguous'),
               'explicit_fp32_post':lambda:explicit(q,k,v,'post'), 'explicit_fp32_split':lambda:explicit(q,k,v,'split'),
               'explicit_fp32_split_head8':lambda:explicit(q,k,v,'split',8)}
    baseline=None
    with torch.inference_mode(),torch.autocast('npu',dtype=torch.float16):
        for name,fn in functions.items():
            out,t=timed(fn)
            if baseline is None:baseline=out.detach().cpu()
            row={'timing':t,'vs_actual':metric(out,actual),'vs_native_same_input':metric(out,baseline)}
            r['cases'][name]=row;OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'name':name,**row}),flush=True)
    r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
