#!/usr/bin/env python3
"""Isolated NPU internal-format replay; flag restored, not production policy."""
import json
from pathlib import Path
import time
import torch
import torch_npu
from torch.nn import functional as F

ROOT=Path('/tmp/h3_round57_text01');OUT=ROOT/'sdpa_format_policy_probe.json'
if OUT.exists():raise RuntimeError('Refuse format-policy probe overwrite')
torch.npu.set_device(0)
original=torch.npu.config.allow_internal_format
r={'original_internal_format':original,'cases':[],'scope':'One isolated process changes its format policy and restores it; no production flag changes or complete quality proof'}
try:
 with torch.inference_mode():
  d=torch.load(ROOT/'attention_layer_49.pt',weights_only=True);ref=torch.load(ROOT/'fp32_attention_layer_49.pt',weights_only=True)['reference']
  for flag in (False,True):
   torch.npu.config.allow_internal_format=flag
   q,k,v=[d[x].to('npu').contiguous() for x in ('q','k','v')]
   item={'allow_internal_format':flag}
   try:
    torch.npu.synchronize();t=time.perf_counter();out=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True,dropout_p=0.);torch.npu.synchronize()
    a=out.float().cpu();e=a-ref
    item.update({'one_call_s':time.perf_counter()-t,'finite':bool(torch.isfinite(a).all()),'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/ref.square().mean().sqrt())})
   except Exception as exc:item['failure']={'type':type(exc).__name__,'message':str(exc)}
   r['cases'].append(item)
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 torch.npu.config.allow_internal_format=original;r['flag_restored']=torch.npu.config.allow_internal_format==original
 OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
