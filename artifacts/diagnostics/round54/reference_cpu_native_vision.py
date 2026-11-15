#!/usr/bin/env python3
"""Isolate the CPU port/HF visual math before blaming an NPU backend.

Constructs only source-extracted vision classes, not vLLM runtime text/DiT. Loads
actual visual weights and checks real interpolation/rotary and full visual output.
"""
import ast
import json
from pathlib import Path
import time
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

ROOT=Path('/tmp/h3_round54_capture01');CODE=Path('/tmp/h3_round54_validation01/source/encoder.py')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'cpu_native_vision_reference.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite visual comparison')
torch.set_num_threads(16)
tree=ast.parse(CODE.read_text())
names={'_rotate_half','_apply_rotary_pos_emb_vision','_vision_sdpa_bf16_compatible','_NpuBF16CompatibleLayerNorm'}
selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (n.name in names or n.name.startswith('MiniMaxH3Qwen3VLVision'))]
ns={'torch':torch,'nn':nn,'F':F,'Any':Any}
exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(CODE),'exec'),ns)
config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config;config._attn_implementation='sdpa'
inputs=torch.load(ROOT/'encoder_input.pt',weights_only=True)
hf=torch.load(ROOT/'hf_vision_outputs.pt',weights_only=True)
cap,capdeep=torch.load(ROOT/'vision_outputs.pt',weights_only=True)
report={'scope':'Actual visual weights and processor pixels on CPU native port vs independent CPU HF; isolates port/dtype differences from NPU','comparisons':{}}

def metrics(a,b):
    assert a.shape==b.shape
    x=a.float();y=b.float();e=x-y
    return {'shape':list(x.shape),'bit_exact':bool(torch.equal(a,b)),
            'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))}
try:
    start=time.perf_counter()
    model=ns['MiniMaxH3Qwen3VLVisionModel'](config).to(torch.bfloat16).eval()
    wanted=set(dict(model.named_parameters()));state={}
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    for shard in sorted(set(index.values())):
        keys=[k for k,s in index.items() if s==shard and k.startswith('model.visual.') and k[len('model.visual.'):] in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
            for k in keys:state[k[len('model.visual.'):]]=f.get_tensor(k).to(torch.bfloat16)
    model.load_state_dict(state,strict=True);report['load_s']=time.perf_counter()-start
    # A small HF skeleton sharing the same actual embeddings isolates the
    # interpolation and RoPE tables without running a second full tower.
    with torch.device('meta'):teacher=Qwen3VLVisionModel(config)
    teacher.pos_embed=model.pos_embed
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
    teacher.rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(config.hidden_size//config.num_heads//2)
    grid=inputs['image_grid_thw']
    native_pos=model.fast_pos_embed_interpolate(grid)
    hf_pos=teacher.fast_pos_embed_interpolate(grid).to(torch.bfloat16)
    report['position_interpolation']=metrics(native_pos,hf_pos)
    report['rotary_table']=metrics(model.rot_pos_emb(grid),teacher.rot_pos_emb(grid))
    # FP32 weights/products/reduction rather than early BF16 interpolation.
    method=next(n for n in next(c for c in selected if isinstance(c,ast.ClassDef) and c.name=='MiniMaxH3Qwen3VLVisionModel').body if isinstance(n,ast.FunctionDef) and n.name=='fast_pos_embed_interpolate')
    source=ast.unparse(method)
    candidate=source.replace('dtype=self.pos_embed.weight.dtype','dtype=torch.float32')
    candidate=candidate.replace('self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]','self.pos_embed(idx_tensor).float() * weight_tensor[:, :, None]')
    candidate=candidate.replace('pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]','pos_embeds.sum(dim=0)')
    candidate=candidate.replace('return torch.cat(patch_pos_embeds_permute)','return torch.cat(patch_pos_embeds_permute).to(self.pos_embed.weight.dtype)')
    env=dict(ns);exec(candidate,env)
    corrected=env['fast_pos_embed_interpolate'](model,grid)
    report['fp32_interpolation_candidate']=metrics(corrected,hf_pos)
    # Store the fast table check immediately in case the full tower takes longer.
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    start=time.perf_counter()
    with torch.inference_mode():out,deep=model(inputs['pixel_values'],grid)
    report['native_cpu_forward_s']=time.perf_counter()-start
    report['comparisons']['native_cpu_vs_hf_merged']=metrics(out,hf['merged'])
    report['comparisons']['native_cpu_vs_captured_npu_merged']=metrics(out,cap)
    report['comparisons']['native_cpu_vs_hf_deepstack']=[metrics(a,b) for a,b in zip(deep,hf['deepstack'])]
    report['completed']=True
    print(json.dumps(report,indent=2),flush=True)
except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(report,indent=2)+'\n');print('Evidence:',OUT,flush=True)
