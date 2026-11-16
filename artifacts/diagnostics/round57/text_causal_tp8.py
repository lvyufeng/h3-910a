#!/usr/bin/env python3
"""Text-only TP8 forward on captured visual/text inputs; no DiT/VAE work.

Uses actual encoder weight loaders and effective NPU RoPE/GQA/packed-SwiGLU
patches. Two complete forwards compare original text math against the opt-in
candidate. All weights remain BF16; vision is a shared real captured input.
"""
import ast
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from typing import Any,Iterable
import torch
import torch_npu
from torch import nn
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT=Path('/tmp/h3_round57_text01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
OUT=ROOT/f'text_causal_rank{rank}.json'
if OUT.exists():raise RuntimeError('Refuse complete text reference overwrite')
torch.npu.set_device(rank);torch.set_num_threads(2);os.environ['VLLM_OMNI_910A_NO_FUSED_NORM']='1';os.environ['VLLM_OMNI_910A_NO_FUSED_ACT']='1'
from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel
from vllm_omni.diffusion.models.minimax_h3 import encoder as baseline
from vllm_omni.platforms.npu.models.minimax_h3 import _apply_rotary_pos_emb_npu,_scaled_dot_product_attention_npu,_forward_minimax_h3_qwen3vl_text_mlp_npu
spec=importlib.util.spec_from_file_location('text_candidate56',ROOT/'candidate_text_linear_encoder.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
candidate._apply_rotary_pos_emb=_apply_rotary_pos_emb_npu
from h3_text_causal57 import causal_fp32
candidate._scaled_dot_product_attention=causal_fp32
candidate.MiniMaxH3Qwen3VLTextMLP.forward=_forward_minimax_h3_qwen3vl_text_mlp_npu
r={'rank':rank,'cases':{},'fp32_linear_compute':True,'causal_fp32_compute':True,'scope':'Complete text-only TP8 on shared actual visual input, FP32 causal attention/linears with BF16 storage/I-O; experimental stack, not whole generation quality or pure attention attribution'}
try:
 init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl')
 initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
 group=GroupCoordinator([list(range(size))],rank,'hccl')
 config=Qwen3VLConfig.from_pretrained(str(MODEL)).text_config
 model=candidate.MiniMaxH3Qwen3VLTextModel(group,config,50,torch.bfloat16).to(dtype=torch.bfloat16).eval()
 owner=candidate.MiniMaxH3Qwen3VLEncoder.__new__(candidate.MiniMaxH3Qwen3VLEncoder);nn.Module.__init__(owner);owner.text_model=model
 owner.vision=nn.Module();owner.quant_config=None
 index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
 def weights():
  for shard in sorted(set(index.values())):
   keys=[k for k,s in index.items() if s==shard and k.startswith('model.language_model.layers.') and int(k.split('.')[3])<50]
   if not keys:continue
   with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
    for k in keys:yield k,f.get_tensor(k)
  shape=model.embed_tokens.weight.shape
  # Captured inputs_embeds bypass embedding forward. Seed this unused shard
  # explicitly so actual load_weights completeness checks remain meaningful.
  yield 'model.language_model.embed_tokens.weight',torch.zeros(config.vocab_size,config.hidden_size,dtype=torch.bfloat16)
 t=time.perf_counter();loaded=owner.load_weights(weights());r['load_s']=time.perf_counter()-t;r['loaded_parameter_tensors']=len(loaded)
 # Remove unused vocabulary shard before moving the retained language stack to NPU.
 model.embed_tokens=nn.Identity();model.to(torch.device('npu',rank))
 inp=torch.load(CAP/'text_inputs.pt',weights_only=True)
 x=inp['inputs_embeds'].to('npu');positions=inp['positions'].to('npu');mask=inp['visual_pos_masks'].to('npu');deep=[v.to('npu') for v in inp['deepstack_visual_embeds']]
 outputs={}
 for label,enabled in [('causal_fp32_old_math',False),('causal_fp32_hf_math',True)]:
  os.environ['VLLM_OMNI_910A_TEXT_FP32_LINEAR']='1'
  os.environ['VLLM_OMNI_910A_TEXT_HF_MATH']='1' if enabled else '0'
  torch.distributed.barrier();torch.npu.synchronize();t=time.perf_counter()
  with torch.inference_mode():out=model(x,positions,visual_pos_masks=mask,deepstack_visual_embeds=deep)
  torch.npu.synchronize();dt=time.perf_counter()-t
  cpu_out=out.detach().cpu().contiguous()
  r['cases'][label]={'local_forward_s':dt,'dtype':str(out.dtype),'shape':list(out.shape),'finite':bool(torch.isfinite(cpu_out.float()).all())}
  if rank==0:
   outputs[label]=cpu_out.clone()
   actual=torch.load(CAP/'text_layer_49.pt',weights_only=True);e=outputs[label].float()-actual.float()
   r['cases'][label]['vs_old_actual']={'raw_bits_equal':torch.equal(outputs[label].contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8)),
                                      'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/actual.float().square().mean().sqrt())}
   teacher=torch.load(Path('/tmp/h3_round56_audit01/cpu_last_hf.pt'),weights_only=True)['output'];e=outputs[label].float()-teacher.float()
   r['cases'][label]['vs_cpu_fp32_oracle']={'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/teacher.float().square().mean().sqrt())}
  print(json.dumps({'rank':rank,'case':label,**r['cases'][label]}),flush=True)
  # A second call for each warmed variant isolates first-use compilation.
  torch.distributed.barrier();torch.npu.synchronize();t=time.perf_counter()
  with torch.inference_mode():repeat=model(x,positions,visual_pos_masks=mask,deepstack_visual_embeds=deep)
  torch.npu.synchronize();r['cases'][label]['second_forward_s']=time.perf_counter()-t
  OUT.write_text(json.dumps(r,indent=2)+'\n')
 if rank==0:
  target=ROOT/'text_causal_outputs.pt'
  if target.exists():raise RuntimeError('Refuse output evidence overwrite')
  with target.open('xb') as f:torch.save(outputs,f)
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
 if torch.distributed.is_initialized():
  destroy_model_parallel();torch.distributed.destroy_process_group()
