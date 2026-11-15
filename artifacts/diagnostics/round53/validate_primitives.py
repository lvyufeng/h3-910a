#!/usr/bin/env python3
"""Source-extracted encoder/coordinator primitives; optionally eight-rank HCCL.

The real deployed Omni all_reduce method is executed, not vLLM's different
coordinator. Small powers-of-two matrices permit exact CPU reference checks.
"""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import types
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent

def methods(path,names,base=None):
    tree=ast.parse(path.read_text())
    picked=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if base is not None:
        for node in picked:
            if isinstance(node,ast.ClassDef):
                node.bases=[ast.Name(id='LinearBase',ctx=ast.Load()) if node.name.endswith('Linear') else node.bases[0]]
    ns={'torch':torch,'nn':nn,'F':F,'Any':Any,'LinearBase':nn.Module,'QuantizationConfig':object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=picked,type_ignores=[])),str(path),'exec'),ns)
    return ns

old=methods(ROOT/'source/encoder.py',{'_tp_range','MiniMaxH3Qwen3VLVocabParallelEmbedding','MiniMaxH3Qwen3VLRowParallelLinear'},base=True)
new=methods(ROOT/'candidate_encoder.py',{'_tp_range','MiniMaxH3Qwen3VLVocabParallelEmbedding','MiniMaxH3Qwen3VLRowParallelLinear'},base=True)
tree=ast.parse((ROOT/'source/group_coordinator.py').read_text())
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GroupCoordinator')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='all_reduce')
ns={'torch':torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ROOT/'source/group_coordinator.py'),'exec'),ns)

class RealMethodGroup:
    def __init__(self,rank,size,device_group):
        self.rank_in_group=rank; self.world_size=size; self.device_group=device_group
    all_reduce=ns['all_reduce']

class ReturnOnlyGroup:
    def __init__(self,rank,size,total):
        self.rank_in_group=rank; self.world_size=size; self.total=total
    def all_reduce(self,input_):
        return self.total.to(input_.device,input_.dtype).clone()

def row(mod,group,weight,full_input):
    c=mod['MiniMaxH3Qwen3VLRowParallelLinear']
    layer=c.__new__(c); nn.Module.__init__(layer)
    layer.group=group; layer.input_is_parallel=False
    layer._tp_rank=group.rank_in_group; layer._tp_size=group.world_size
    layer.input_size_per_partition=weight.shape[1]//group.world_size
    layer.output_dtype=weight.dtype
    layer.weight=nn.Parameter(weight[:,group.rank_in_group*layer.input_size_per_partition:(group.rank_in_group+1)*layer.input_size_per_partition].clone())
    layer.quant_method=types.SimpleNamespace(apply=lambda owner,x:F.linear(x,owner.weight))
    return layer(full_input)

p=argparse.ArgumentParser(); p.add_argument('--hccl',action='store_true'); args=p.parse_args()
rank=int(os.environ.get('RANK','0')); size=int(os.environ.get('WORLD_SIZE','8')) if args.hccl else 8
device=torch.device('cpu')
if args.hccl:
    import torch_npu
    torch.npu.set_device(rank); device=torch.device('npu',rank)
    torch.distributed.init_process_group('hccl')
result={'rank':rank,'world_size':size,'device':str(device),'scope':'source-extracted primitive methods with real HCCL, not the full Qwen model','tests':[]}
try:
    vocab=256; hidden=32
    ids=torch.tensor([[0,31,32,63,127,128,191,224,255]],device=device)
    full_weight=(torch.arange(vocab*hidden).reshape(vocab,hidden).remainder(31)-15).float().div(16).to(torch.bfloat16)
    expected=F.embedding(ids.cpu(),full_weight)
    weight=(torch.arange(16*64).reshape(16,64).remainder(7)-3).float().div(16).to(torch.bfloat16)
    x=(torch.arange(9*64).reshape(9,64).remainder(5)-2).float().div(8).to(torch.bfloat16)
    row_expected=F.linear(x.float(),weight.float()).to(torch.bfloat16)
    if args.hccl:
        g=RealMethodGroup(rank,size,torch.distributed.group.WORLD)
        for mod,label in ((old,'deployed'),(new,'return_value_fixed')):
            emb=mod['MiniMaxH3Qwen3VLVocabParallelEmbedding'](g,vocab,hidden,torch.bfloat16).to(device)
            emb.weight_loader(emb.weight,full_weight.to(device))
            out=emb(ids).detach().cpu()
            rr=row(mod,g,weight.to(device),x.to(device)).detach().cpu()
            assert torch.equal(out,expected),label+' embedding mismatch'
            assert torch.equal(rr,row_expected),label+' row mismatch'
            result['tests'].append({'case':label+'_hccl_contiguous','embedding_exact_cpu':True,'row_exact_cpu':True})
        inp=torch.full((4,16),rank+1.,device=device)
        out=g.all_reduce(inp)
        assert out.data_ptr()==inp.data_ptr() and torch.equal(out.cpu(),torch.full((4,16),36.))
        result['hccl_contiguous_mutates_original']=True
        # Coordinator's contiguous() copy creates an out-of-place result for a view.
        inp=torch.full((4,16),rank+1.,device=device).t()
        out=g.all_reduce(inp)
        assert out.data_ptr()!=inp.data_ptr() and torch.equal(out.cpu(),torch.full((16,4),36.))
        assert torch.equal(inp.cpu(),torch.full((16,4),rank+1.))
        result['hccl_noncontiguous_requires_return']=True
    else:
        for r in range(8):
            g=ReturnOnlyGroup(r,8,expected)
            fixed=new['MiniMaxH3Qwen3VLVocabParallelEmbedding'](g,vocab,hidden,torch.bfloat16)
            fixed.weight_loader(fixed.weight,full_weight)
            wrong=old['MiniMaxH3Qwen3VLVocabParallelEmbedding'](g,vocab,hidden,torch.bfloat16)
            wrong.weight_loader(wrong.weight,full_weight)
            assert torch.equal(fixed(ids),expected)
            assert not torch.equal(wrong(ids),expected)
            rg=ReturnOnlyGroup(r,8,row_expected.float())
            assert torch.equal(row(new,rg,weight,x),row_expected)
            assert not torch.equal(row(old,rg,weight,x),row_expected)
        result['tests'].append({'case':'return_only_coordinator','fixed_exact_cpu_all_8_ranks':True,'old_wrong_all_8_ranks':True})
    result['passed']=True
except BaseException as exc:
    result['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    if args.hccl:
        (ROOT/f'hccl_rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        torch.distributed.destroy_process_group()
    else:
        (ROOT/'cpu_primitives.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
