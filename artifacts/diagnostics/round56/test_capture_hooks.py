#!/usr/bin/env python3
"""CPU regression for default-off/write-once late activation hooks."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('audit56',ROOT/'h3_text_audit56.py');audit=importlib.util.module_from_spec(s);s.loader.exec_module(audit)
class Norm(nn.Module):
 def forward(self,x,residual=None):return x if residual is None else (x,residual)
class Attention(nn.Module):
 def forward(self,x,position_embeddings=None):return x*2
class Rotary(nn.Module):
 def forward(self,x,positions):return (torch.ones(1,2,8),torch.zeros(1,2,8))
class Last(nn.Module):
 def __init__(self):
  super().__init__();self.input_layernorm=Norm();self.post_attention_layernorm=Norm();self.self_attn=Attention();self.mlp=nn.Module();self.mlp.gate_up_proj=nn.Identity();self.mlp.down_proj=nn.Identity()
 def forward(self,x,position_embeddings=None):
  y=self.self_attn(self.input_layernorm(x),position_embeddings=position_embeddings);y,residual=self.post_attention_layernorm(y,x)
  return residual+self.mlp.down_proj(self.mlp.gate_up_proj(y))
class Text(nn.Module):
 def __init__(self):
  super().__init__();self.rotary_emb=Rotary();self.layers=nn.ModuleList([nn.Identity() for _ in range(49)]+[Last()])
 def forward(self,x,positions):return self.layers[49](x,position_embeddings=self.rotary_emb(x,positions))
class Encoder(nn.Module):
 def __init__(self):super().__init__();self.text_model=Text()
x=torch.randn(1,2,8,requires_grad=True);ids=torch.tensor([1,2]);positions=torch.zeros(3,1,2,dtype=torch.long)
with patch.dict(os.environ,{},clear=False),patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_rank',return_value=0):
 os.environ.pop('VLLM_OMNI_910A_TEXT_AUDIT_DIR',None);e=Encoder();audit.install(e,ids,{})
 assert not audit._SAVED and not audit._ATTACHED
with tempfile.TemporaryDirectory(prefix='h3_round56_capture_unit_') as work:
 root=Path(work).resolve();assert root.name.startswith('h3_round56_capture_unit_')
 with patch.dict(os.environ,{'VLLM_OMNI_910A_TEXT_AUDIT_DIR':str(root)}),patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_rank',return_value=0):
  e=Encoder();before=e.text_model(x,positions);audit.install(e,ids,{})
  after=e.text_model(x,positions);second=e.text_model(x,positions)
  assert torch.equal(before,after) and torch.equal(after,second)
  assert len(list(root.glob('*.pt')))==10
  p=root/'layer49_input.pt';v=torch.load(p,weights_only=True);assert not v['args'][0].requires_grad
  for m in root.glob('*.json'):
   r=json.loads(m.read_text());assert r['tensor_bytes']<=512*1024*1024
 assert root.resolve()==root and root.name.startswith('h3_round56_capture_unit_')
with tempfile.TemporaryDirectory(prefix='h3_round56_capture_unit_cap_') as work:
 root=Path(work).resolve();assert root.name.startswith('h3_round56_capture_unit_cap_')
 audit._SAVED.clear();audit._MAX_BYTES=1
 with patch.dict(os.environ,{'VLLM_OMNI_910A_TEXT_AUDIT_DIR':str(root)}),patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_rank',return_value=0):
  try:audit.save('encoder_input',x)
  except RuntimeError:pass
  else:raise AssertionError('Expected capture byte cap')
  assert not list(root.iterdir())
 assert root.resolve()==root and root.name.startswith('h3_round56_capture_unit_cap_')
r={'passed':True,'default_off_no_hooks':True,'first_and_second_outputs_identical':True,'write_once':True,'detached':True,'byte_cap':True,'capture_files':10}
(ROOT/'capture_hook_tests.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
