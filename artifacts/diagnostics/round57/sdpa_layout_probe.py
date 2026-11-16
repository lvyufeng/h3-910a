#!/usr/bin/env python3
"""Same real QKV geometry/layout replay of native NPU GQA versus FP32 causal.

Captured tensors are contiguous CPU copies; reconstruct actual noncontiguous
BNSD transpose views separately so standalone failure is not confused with the
measured runtime path. No production changes or speed extrapolation.
"""
import json
from pathlib import Path
import time
import torch
import torch_npu
from torch.nn import functional as F
from h3_text_causal57 import causal_fp32

ROOT=Path('/tmp/h3_round57_text01');OUT=ROOT/'sdpa_layout_probe.json'
if OUT.exists():raise RuntimeError('Refuse layout probe overwrite')
torch.npu.set_device(0)
r={'cases':[],'scope':'Standalone actual QKV with explicit layout replays; production timings measured separately'}
try:
 with torch.inference_mode():
  data=torch.load(ROOT/'attention_layer_49.pt',weights_only=True);q,k,v=[data[x].to('npu') for x in ('q','k','v')]
  def transpose_view(t):return t.transpose(1,2).contiguous().transpose(1,2)
  for label,values in [('contiguous_bnsd',(q,k,v)),('runtime_like_transpose_views',tuple(transpose_view(t) for t in (q,k,v))),('explicit_repeated_kv',(q,k.repeat_interleave(8,dim=1),v.repeat_interleave(8,dim=1)) )]:
   a,b,c=values;item={'label':label,'strides':[list(t.stride()) for t in values]}
   try:
    torch.npu.synchronize();t=time.perf_counter();out=F.scaled_dot_product_attention(a,b,c,is_causal=True,enable_gqa=a.shape[1]!=b.shape[1],dropout_p=0.);torch.npu.synchronize()
    item['seconds']=time.perf_counter()-t;item['finite']=bool(torch.isfinite(out.detach().float().cpu()).all())
   except Exception as exc:item['failure']={'type':type(exc).__name__,'message':str(exc)}
   r['cases'].append(item)
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
