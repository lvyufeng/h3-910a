#!/usr/bin/env python3
"""GPU-denominator same-input refiner comparison; no model inference or tolerance waiver."""
import argparse,hashlib,json
from pathlib import Path
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu',required=True);p.add_argument('--npu',required=True);p.add_argument('--sampling',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
gpu,npu=Path(a.gpu),Path(a.npu)
reference_input=torch.load(a.sampling,map_location='cpu',weights_only=True)
tags=reference_input['text_tags']
result={'scope':__doc__,'completed':False,'GPU_reference_denominator':True,'numeric_tolerance':None,'quality_or_precision_accepted':False,'stages':{}}
def metric(x,y):
 assert x.shape==y.shape and x.dtype==y.dtype,(x.shape,y.shape,x.dtype,y.dtype)
 finite=bool(torch.isfinite(x.float()).all() and torch.isfinite(y.float()).all())
 d=x.double()-y.double();yy=y.double()
 return {'finite':finite,'relative_RMS':float(d.square().sum().sqrt()/yy.square().sum().sqrt().clamp_min(1e-30)), 'RMSE':float(d.square().mean().sqrt()),'max_abs_error':float(d.abs().max()),'GPU_max_abs':float(yy.abs().max()),'NPU_max_abs':float(x.double().abs().max()),'changed_fraction':float((x!=y).double().mean()),'raw_bits_equal':torch.equal(x.view(torch.uint8),y.view(torch.uint8))}
for rank in [0,1]:
 for name in ['condition_proj','refiner_0_qkv','refiner_0_attention','refiner_0_MLP','refiner_0_residual','refiner_1_qkv','refiner_1_attention','refiner_1_MLP','refiner_1_residual','refiner_final']:
  g=gpu/f'{name}_rank{rank}.pt';n=npu/f'{name}_rank{rank}.pt'
  if not g.exists() or not n.exists():continue
  x=torch.load(n,map_location='cpu',weights_only=True)['value'];y=torch.load(g,map_location='cpu',weights_only=True)['value']
  row={'shape':list(y.shape),'dtype':str(y.dtype),'all_rows':metric(x,y),'by_token_tag':{}}
  if len(x)==len(tags):
   for tag in torch.unique(tags):
    mask=tags==tag;row['by_token_tag'][str(int(tag))]=metric(x[mask],y[mask])
  result['stages'][f'{name}_rank{rank}']=row
  print(json.dumps({'stage':name,'rank':rank,**row['all_rows']},ensure_ascii=False),flush=True)
result['completed']=True
result['full_refiner_both_ranks_present']=all(f'refiner_final_rank{rank}' in result['stages'] for rank in [0,1])
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
