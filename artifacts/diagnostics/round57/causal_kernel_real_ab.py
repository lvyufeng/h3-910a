#!/usr/bin/env python3
"""Warm complete causal wrapper timing on all five actual captured text layers."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from h3_text_causal57 import causal_fp32

ROOT=Path('/tmp/h3_round57_text01')
p=argparse.ArgumentParser();p.add_argument('--fast',action='store_true');args=p.parse_args()
OUT=ROOT/('causal_fast_real_ab.json' if args.fast else 'causal_kernel_real_ab.json')
if OUT.exists():raise RuntimeError('Refuse causal A/B evidence overwrite')
torch.npu.set_device(0)
s=importlib.util.spec_from_file_location('causal_candidate',ROOT/('candidate_causal_fast_fa910a.py' if args.fast else 'candidate_causal_fa910a.py'));fa=importlib.util.module_from_spec(s);s.loader.exec_module(fa)
r={'cases':[],'fast_prefix_traversal':args.fast,'scope':'Same actual QKV, FP32 eager vs causal TBE, complete wrapper casts/repeat/pad/crop; no whole-model lossless proof'}
def wrapper(q,k,v):
 source=q.shape[-2];n=((source+127)//128)*128
 with torch.autocast('npu',enabled=False):
  qf=q.half();kf=k.half().repeat_interleave(q.shape[1]//k.shape[1],dim=1);vf=v.half().repeat_interleave(q.shape[1]//v.shape[1],dim=1)
  if n!=source:qf,kf,vf=fa._pad_qkv(qf,kf,vf,n)
  return fa._flash_attention_core(qf,kf,vf,key_seq_limit=source,causal=True)[:,:,:source].contiguous().to(q.dtype)
def metric(a,b):
 x=a.float().cpu();y=b.float().cpu();e=x-y
 return {'finite':bool(torch.isfinite(x).all()),'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))}
try:
 with torch.inference_mode():
  for layer in (0,24,40,48,49):
   data=torch.load(ROOT/f'attention_layer_{layer:02d}.pt',weights_only=True);q,k,v=[data[x].to('npu') for x in ('q','k','v')]
   ref=torch.load(ROOT/f'fp32_attention_layer_{layer:02d}.pt',weights_only=True)['reference']
   item={'layer':layer,'variants':{}}
   for label,call in [('fp32_eager',lambda:causal_fp32(q,k,v)),('causal_tbe',lambda:wrapper(q,k,v))]:
    call();torch.npu.synchronize();groups=[]
    for _ in range(3):
     t=time.perf_counter()
     for _ in range(5):out=call()
     torch.npu.synchronize();groups.append((time.perf_counter()-t)*1000/5)
    item['variants'][label]={'groups_ms':groups,'median_ms':statistics.median(groups),'reference':metric(out,ref)}
   r['cases'].append(item);print(json.dumps(item),flush=True)
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
