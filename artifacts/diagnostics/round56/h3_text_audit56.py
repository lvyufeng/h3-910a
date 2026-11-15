"""Default-off, bounded rank-zero late-text activation/context captures."""
import json
import os
from pathlib import Path
import time

_SAVED=set();_ATTACHED=set();_MAX_BYTES=512*1024*1024
_ALLOWED={'encoder_input','text_input','text_rotary','layer49_input','layer49_output','layer49_norm1','layer49_attention','layer49_norm2','layer49_gate_up','layer49_down'}
def _enabled():return bool(os.environ.get('VLLM_OMNI_910A_TEXT_AUDIT_DIR'))
def _rank_zero():
 import torch
 return torch.distributed.is_initialized() and torch.distributed.get_rank()==0

def _context():
 import torch
 return {'autocast_enabled':torch.is_autocast_enabled('npu'),'autocast_dtype':str(torch.get_autocast_dtype('npu')),
         'default_device':str(torch.get_default_device()),
         'selected_environment':{key:os.environ.get(key) for key in ('VLLM_OMNI_910A_NO_FUSED_NORM','VLLM_OMNI_910A_NO_FUSED_ACT','VLLM_OMNI_910A_VISION_FP32','VLLM_OMNI_910A_TE_DTYPE')}}
def _bytes(value):
 import torch
 if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
 if isinstance(value,dict):return sum(_bytes(v) for v in value.values())
 if isinstance(value,(tuple,list)):return sum(_bytes(v) for v in value)
 return 0

def _copy(value):
 import torch
 if isinstance(value,torch.Tensor):return value.detach().to('cpu').contiguous().clone()
 if isinstance(value,dict):return {k:_copy(v) for k,v in value.items()}
 if isinstance(value,tuple):return tuple(_copy(v) for v in value)
 if isinstance(value,list):return [_copy(v) for v in value]
 return value

def save(name,value):
 if not _enabled() or not _rank_zero() or name in _SAVED:return None
 import torch
 assert name in _ALLOWED
 root=Path(os.environ['VLLM_OMNI_910A_TEXT_AUDIT_DIR']).resolve(strict=True)
 if not root.is_dir() or not root.name.startswith('h3_round56_capture'):raise RuntimeError('Unexpected audit path')
 size=_bytes(value)
 if size>_MAX_BYTES:raise RuntimeError('Late-text capture exceeds explicit byte cap')
 target=root/(name+'.pt');meta=root/(name+'.json')
 if target.exists() or meta.exists():raise RuntimeError('Refuse earlier capture overwrite')
 t=time.perf_counter()
 with target.open('xb') as f:torch.save(_copy(value),f)
 with meta.open('x') as f:json.dump({'name':name,'tensor_bytes':size,'file_bytes':target.stat().st_size,'capture_s':time.perf_counter()-t,'scope':'Rank0 diagnostic synchronizing copy, not a performance benchmark'},f,indent=2)
 _SAVED.add(name);print('[H3TEXTAUDIT] '+name,flush=True)
 return None

def _output(name):
 def capture(module,args,value):
  save(name,{'args':args,'output':value,'context':_context(),'module':type(module).__name__,
             'tp_rank':getattr(module,'_tp_rank',None)})
  return None
 return capture

def install(encoder,input_ids,vision):
 if not _enabled() or not _rank_zero():return None
 save('encoder_input',{'input_ids':input_ids,**vision,'context':_context()})
 if id(encoder) in _ATTACHED:return None
 _ATTACHED.add(id(encoder));model=encoder.text_model;last=model.layers[49]
 def text_pre(module,args,kwargs):
  save('text_input',{'args':args,'kwargs':kwargs,'context':_context()})
  return None
 def last_pre(module,args,kwargs):
  save('layer49_input',{'args':args,'kwargs':kwargs,'context':_context()})
  return None
 model.register_forward_pre_hook(text_pre,with_kwargs=True)
 model.rotary_emb.register_forward_hook(_output('text_rotary'))
 last.register_forward_pre_hook(last_pre,with_kwargs=True);last.register_forward_hook(_output('layer49_output'))
 for module,name in ((last.input_layernorm,'layer49_norm1'),(last.self_attn,'layer49_attention'),
                     (last.post_attention_layernorm,'layer49_norm2'),(last.mlp.gate_up_proj,'layer49_gate_up'),(last.mlp.down_proj,'layer49_down')):
  module.register_forward_hook(_output(name))
 return None
