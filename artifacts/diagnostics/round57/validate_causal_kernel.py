#!/usr/bin/env python3
"""Compile/test causal triangular masks independently before model integration."""
import importlib.util
import json
from pathlib import Path
import time
import torch
import torch_npu

ROOT=Path('/tmp/h3_round57_text01');OUT=ROOT/'causal_kernel_validation.json'
if OUT.exists():raise RuntimeError('Refuse causal validation evidence overwrite')
torch.npu.set_device(0)
s=importlib.util.spec_from_file_location('causal_candidate',ROOT/'candidate_causal_fa910a.py');fa=importlib.util.module_from_spec(s);s.loader.exec_module(fa)
r={'cases':[],'scope':'Isolated causal TBE prototype; not lossless/full text/model quality proof'}
def metric(a,b):
 x=a.float().cpu();y=b.float().cpu();e=x-y
 return {'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all()),'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))}
try:
 with torch.inference_mode():
  for n,valid in ((128,127),(256,243),(512,511)):
   q=torch.zeros(1,2,n,128,dtype=torch.float16,device='npu');k=q.clone();v=torch.arange(n,dtype=torch.float32).reshape(1,1,n,1).expand(1,2,n,128).to('npu').half().contiguous()
   v[:,:,valid:,:].fill_(60000)
   ref=torch.arange(valid,dtype=torch.float32).div(2).reshape(1,1,valid,1).expand(1,2,valid,128)
   out=fa._flash_attention_core(q,k,v,scale=1.,key_seq_limit=valid,causal=True);torch.npu.synchronize()
   item={'n':n,'valid':valid,'uniform_prefix_value':metric(out[:,:,:valid],ref)}
   assert item['uniform_prefix_value']['finite'] and item['uniform_prefix_value']['max_abs_error']<.3
   r['cases'].append(item);print(json.dumps(item),flush=True)
  d=torch.load(ROOT/'attention_layer_49.pt',weights_only=True);source_n=d['q'].shape[-2];n=((source_n+127)//128)*128
  q=d['q'].half().to('npu');k=d['k'].half().to('npu').repeat_interleave(q.shape[1]//d['k'].shape[1],1);v=d['v'].half().to('npu').repeat_interleave(q.shape[1]//d['v'].shape[1],1)
  q,k,v=fa._pad_qkv(q,k,v,n)
  torch.npu.synchronize();t=time.perf_counter();out=fa._flash_attention_core(q,k,v,key_seq_limit=source_n,causal=True);torch.npu.synchronize()
  ref=torch.load(ROOT/'fp32_attention_layer_49.pt',weights_only=True)['reference']
  r['actual_layer49']={'elapsed_s':time.perf_counter()-t,'source_n':source_n,'kernel_n':n,'fp32_reference':metric(out[:,:,:source_n],ref)}
  r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
