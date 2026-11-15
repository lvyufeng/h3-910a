#!/usr/bin/env python3
"""Actual NPU validation of model-specific HF norm/outer-product RoPE candidate."""
import ast
import json
import os
from pathlib import Path
from typing import Any
import torch
import torch_npu
from torch import nn
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm,Qwen3VLTextRotaryEmbedding
from safetensors import safe_open
from vllm_omni.diffusion.layers.norm import RMSNorm as DiffusionRMSNorm

ROOT=Path('/tmp/h3_round56_audit01');CAP=Path('/tmp/h3_round56_capture01');OLD=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'text_math_validation.json'
if OUT.exists():raise RuntimeError('Refuse text-math validation overwrite')
torch.npu.set_device(0);torch.set_num_threads(16);os.environ['VLLM_OMNI_910A_NO_FUSED_NORM']='1'
config=Qwen3VLConfig.from_pretrained(str(MODEL)).text_config
names={'MiniMaxH3Qwen3VLRMSNorm','MiniMaxH3Qwen3VLTextRotaryEmbedding','_apply_interleaved_mrope','_text_hf_math_enabled','_NpuBF16CompatibleLinear','_vision_fp32_enabled'}
tree=ast.parse((ROOT/'candidate_text_math_encoder.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
ns={'torch':torch,'nn':nn,'Any':Any,'DiffusionRMSNorm':DiffusionRMSNorm,'F':torch.nn.functional}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'candidate_math','exec'),ns)
r={'rope_cases':[],'norm_cases':[],'scope':'Actual NPU model-specific formulas on same real tensors; isolated operators not complete encoder/lossless-video proof'}
def metric(a,b):
 x=a.detach().cpu().contiguous();y=b.detach().cpu().contiguous();e=x.float()-y.float()
 return {'shape':list(x.shape),'finite':bool(torch.isfinite(x.float()).all() and torch.isfinite(y.float()).all()),
         'raw_bits_equal':x.dtype==y.dtype and torch.equal(x.view(torch.uint8),y.view(torch.uint8)),
         'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.float().square().mean().sqrt().clamp_min(1e-12))}
try:
 with torch.inference_mode():
  inp=torch.load(OLD/'text_inputs.pt',weights_only=True);x=inp['inputs_embeds'];p=inp['positions']
  teacher_rope=Qwen3VLTextRotaryEmbedding(config);candidate_rope=ns['MiniMaxH3Qwen3VLTextRotaryEmbedding'](config)
  os.environ['VLLM_OMNI_910A_TEXT_HF_MATH']='1'
  for label,xc,pc in [('actual',x,p),('small_two_batch',torch.zeros(2,17,5120,dtype=torch.bfloat16),torch.arange(17).reshape(1,1,17).expand(3,2,17).clone())]:
   ref=teacher_rope(xc,pc);xn=xc.to('npu');pn=pc.to('npu')
   for amp in (False,True):
    with torch.autocast('npu',dtype=torch.float16,enabled=amp):got=candidate_rope(xn,pn)
    row={'name':label,'fp16_autocast':amp,'comparisons':[metric(a,b) for a,b in zip(got,ref)]}
    assert all(z['raw_bits_equal'] for z in row['comparisons']);r['rope_cases'].append(row)
  # Verify existing CPU behavior is unchanged even with the option enabled.
  ref=teacher_rope(x,p);cpu=candidate_rope(x,p)
  assert all(z['raw_bits_equal'] for z in [metric(a,b) for a,b in zip(cpu,ref)])
  r['cpu_rope_unchanged']=True
  # Actual last-layer inputs come from the focused unmodified NPU capture.
  for capture in ('layer49_norm1','layer49_norm2'):
   if not (CAP/(capture+'.json')).exists():raise RuntimeError('Actual norm capture not ready: '+capture)
   data=torch.load(CAP/(capture+'.pt'),weights_only=True);args=data['args'];xc=args[0];residual=args[1] if len(args)>1 else None
   name='input_layernorm' if capture.endswith('norm1') else 'post_attention_layernorm'
   key='model.language_model.layers.49.'+name+'.weight';index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
   with safe_open(str(MODEL/index[key]),framework='pt',device='cpu') as f:gamma=f.get_tensor(key).bfloat16()
   teacher=Qwen3VLTextRMSNorm(config.hidden_size,config.rms_norm_eps).to(torch.bfloat16);teacher.weight.copy_(gamma)
   old=DiffusionRMSNorm(config.hidden_size,config.rms_norm_eps,dtype=torch.bfloat16).to('npu');old.weight.copy_(gamma.to('npu'))
   new=ns['MiniMaxH3Qwen3VLRMSNorm'](config.hidden_size,config.rms_norm_eps,dtype=torch.bfloat16).to('npu');new.weight.copy_(gamma.to('npu'))
   xn=xc.to('npu');rn=residual.to('npu') if residual is not None else None
   reference_residual=xc+residual if residual is not None else None
   reference=teacher(reference_residual if residual is not None else xc)
   a=old(xn,rn);b=new(xn,rn);ao=a[0] if isinstance(a,tuple) else a;bo=b[0] if isinstance(b,tuple) else b
   item={'capture':capture,'forward_dispatch':getattr(getattr(new,'_forward_method',None),'__qualname__',None),'old_vs_hf':metric(ao,reference),'new_vs_hf':metric(bo,reference)}
   assert item['new_vs_hf']['relative_rms']<5e-4
   if residual is not None:item['updated_residual_vs_hf']=metric(b[1],reference_residual)
   os.environ['VLLM_OMNI_910A_TEXT_HF_MATH']='0';disabled=new(xn,rn);do=disabled[0] if isinstance(disabled,tuple) else disabled
   assert metric(do,ao)['raw_bits_equal'];item['disabled_exactly_original']=True
   os.environ['VLLM_OMNI_910A_TEXT_HF_MATH']='1';r['norm_cases'].append(item)
  # Adversarial BF16 residual rounding and finite outliers, same model formula.
  torch.manual_seed(56);xc=torch.randn(7,128).bfloat16();xc[0].zero_();xc[1].fill_(1);xc[2,0]=20736
  residual=torch.full_like(xc,0.003936767578125);gamma=torch.randn(128).bfloat16()
  teacher=Qwen3VLTextRMSNorm(128,1e-6).to(torch.bfloat16);teacher.weight.copy_(gamma)
  new=ns['MiniMaxH3Qwen3VLRMSNorm'](128,1e-6,dtype=torch.bfloat16).to('npu');new.weight.copy_(gamma.to('npu'))
  merged=xc+residual;ref=teacher(merged);got,updated=new(xc.to('npu'),residual.to('npu'))
  r['adversarial_norm']={'output':metric(got,ref),'residual':metric(updated,merged)}
  assert r['adversarial_norm']['output']['relative_rms']<5e-4 and r['adversarial_norm']['residual']['raw_bits_equal']
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
