#!/usr/bin/env python3
"""Same real QKV: isolate causal attention precision and complete call timing.

CPU full-key FP32 matmul/softmax/PV is the numerical reference. NPU variants
include all casts and mask application. GQA repeat is explicit and semantics
are exactly upper-left self-attention; no key truncation/sparsity is used.
"""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from torch.nn import functional as F

ROOT=Path('/tmp/h3_round57_text01');OUT=ROOT/'attention_reference.json'
if OUT.exists():raise RuntimeError('Refuse reference evidence overwrite')
torch.npu.set_device(0);torch.set_num_threads(16)
r={'cases':[],'scope':'Complete real rank0 QKV attention A/B against CPU FP32 causal full-key reference; not all-rank/all-layer model quality'}
_MASKS={}
def causal_mask(n,device):
 key=(n,device)
 if key not in _MASKS:_MASKS[key]=torch.ones((n,n),dtype=torch.bool,device=device).triu(1)
 return _MASKS[key]
def math(q,k,v,dtype=torch.float32):
 with torch.autocast('npu',enabled=False):
  qf=q.to(dtype).contiguous();kf=k.to(dtype).contiguous();vf=v.to(dtype).contiguous()
  groups=qf.shape[1]//kf.shape[1]
  if groups!=1:kf=kf.repeat_interleave(groups,dim=1);vf=vf.repeat_interleave(groups,dim=1)
  scores=torch.matmul(qf,kf.transpose(-2,-1))*(qf.shape[-1]**-0.5)
  scores=scores.masked_fill(causal_mask(qf.shape[-2],qf.device),-torch.inf)
  probabilities=torch.softmax(scores,dim=-1)
  return torch.matmul(probabilities,vf).to(q.dtype)
def metric(a,b):
 x=a.detach().float().cpu();y=b.detach().float().cpu();e=x-y;finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
 return {'finite':finite,'max_abs_error':float(e.abs().max()) if finite else None,'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)) if finite else None}
try:
 with torch.inference_mode():
  for layer in (0,24,40,48,49):
   d=torch.load(ROOT/f'attention_layer_{layer:02d}.pt',weights_only=True);qc,kc,vc=d['q'],d['k'],d['v']
   # CPU full-key causal reference, storing FP32 output for all query rows.
   qf=qc.float();kf=kc.float().repeat_interleave(qc.shape[1]//kc.shape[1],dim=1);vf=vc.float().repeat_interleave(qc.shape[1]//vc.shape[1],dim=1)
   scores=qf@kf.transpose(-2,-1)/(qc.shape[-1]**.5);scores.masked_fill_(torch.ones(qc.shape[-2],qc.shape[-2],dtype=torch.bool).triu(1),-torch.inf)
   probabilities=scores.softmax(-1);ref=probabilities@vf
   q,k,v=[x.to('npu') for x in (qc,kc,vc)]
   variants={
    'native_bf16_gqa':lambda:F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True,dropout_p=0.),
    'explicit_fp32_full_math':lambda:math(q,k,v),
    'fp32_sdpa_gqa':lambda:F.scaled_dot_product_attention(q.float(),k.float(),v.float(),is_causal=True,enable_gqa=True,dropout_p=0.).bfloat16(),
    'fp16_sdpa_gqa':lambda:F.scaled_dot_product_attention(q.half(),k.half(),v.half(),is_causal=True,enable_gqa=True,dropout_p=0.).bfloat16(),
    'fp16_full_math':lambda:math(q,k,v,torch.float16)}
   item={'layer':layer,'shape':list(qc.shape),'kv_heads':kc.shape[1],'q_maxabs':float(qc.abs().max()),'k_maxabs':float(kc.abs().max()),'v_maxabs':float(vc.abs().max()),
         'captured_native_vs_fp32':metric(d['output'],ref),'variants':{}}
   for name,call in variants.items():
    try:
     call();torch.npu.synchronize();times=[]
     for _ in range(3):
      t=time.perf_counter();out=call();torch.npu.synchronize();times.append((time.perf_counter()-t)*1000)
     item['variants'][name]={'reference':metric(out,ref),'captured_native':metric(out,d['output']),'groups_ms':times,'median_ms':statistics.median(times)}
     if name=='explicit_fp32_full_math':
      p=ROOT/f'fp32_attention_layer_{layer:02d}.pt'
      if p.exists():raise RuntimeError('Refuse reference tensor overwrite')
      with p.open('xb') as f:torch.save({'reference':ref,'npu_fp32_output':out.detach().cpu()},f)
    except Exception as exc:item['variants'][name]={'failure':{'type':type(exc).__name__,'message':str(exc)}}
   r['cases'].append(item);OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(item),flush=True)
   del q,k,v,kf,vf,scores,probabilities,ref
   torch.npu.empty_cache()
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
