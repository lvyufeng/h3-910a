#!/usr/bin/env python3
"""Eight-rank actual coordinator test of model-specific embedding promotion.

Row projections already use FP32 reduction and are not changed. Real embeddings
plus tiny values/large finite BF16 values test the sole model-specific helper.
"""
import ast
import json
import os
from pathlib import Path
import time
from typing import Any
import torch
import torch_npu
from safetensors import safe_open

ROOT=Path('/tmp/h3_round56_audit01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
OUT=ROOT/f'embedding_rank{rank}.json'
if OUT.exists():raise RuntimeError('Refuse embedding collective evidence overwrite')
torch.npu.set_device(rank);torch.set_num_threads(2);torch.distributed.init_process_group('hccl')
tree=ast.parse((ROOT/'source/group_coordinator.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GroupCoordinator');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='all_reduce')
ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_coordinator','exec'),ns)
class Group:
 all_reduce=ns['all_reduce']
 def __init__(self):self.world_size=size;self.rank_in_group=rank;self.device_group=torch.distributed.group.WORLD
ns={'torch':torch,'Any':Any};tree=ast.parse((ROOT/'candidate_text_math_encoder.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_text_hf_embedding_all_reduce','_text_hf_math_enabled')]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'embedding_helper','exec'),ns);group=Group();helper=ns['_text_hf_embedding_all_reduce']
r={'rank':rank,'cases':[],'scope':'Real8-rank coordinator test of embedding-only range/promotion helper; text row projection reduction is unchanged'}
def metric(a,b):
 x=a.detach().float().cpu();y=b.detach().float().cpu();e=x-y;finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
 return {'finite':finite,'raw_bits_equal':bool(torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.cpu().contiguous().view(torch.uint8))),
         'max_abs_error':float(e.abs().max()) if finite else None,'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-30)) if finite else None}
try:
 ids=torch.load(CAP/'encoder_input.pt',weights_only=True)['input_ids'].unsqueeze(0);index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];key='model.language_model.embed_tokens.weight'
 with safe_open(str(MODEL/index[key]),framework='pt',device='cpu') as f:weight=f.get_tensor(key).bfloat16();embedding=torch.nn.functional.embedding(ids,weight)
 vocab=weight.shape[0];mask=(ids>=rank*(vocab//size))&(ids<(rank+1)*(vocab//size));local=embedding.clone();local.masked_fill_(~mask[:,:,None],0)
 tests=[('actual_embedding',local,embedding),('small_subnormal',torch.full((3,16),2.3283064365386963e-10 if rank==0 else 0.,dtype=torch.bfloat16),torch.full((3,16),2.3283064365386963e-10,dtype=torch.bfloat16)),('finite_above_fp16_range',torch.full((3,16),131072. if rank==0 else 0.,dtype=torch.bfloat16),torch.full((3,16),131072.,dtype=torch.bfloat16))]
 for label,value,expected in tests:
  os.environ['VLLM_OMNI_910A_TEXT_HF_MATH']='0';old=helper(group,value.to('npu'))
  os.environ['VLLM_OMNI_910A_TEXT_HF_MATH']='1';new=helper(group,value.to('npu'))
  item={'case':label,'old':metric(old,expected),'new':metric(new,expected)}
  assert item['new']['raw_bits_equal'] and item['new']['finite'];r['cases'].append(item)
 os.environ['VLLM_OMNI_910A_TEXT_HF_MATH']='0'
 test=local.to('npu');direct=group.all_reduce(test.clone());disabled=helper(group,test.clone());assert metric(disabled,direct)['raw_bits_equal'];r['disabled_exact']=True
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
 torch.distributed.destroy_process_group()
