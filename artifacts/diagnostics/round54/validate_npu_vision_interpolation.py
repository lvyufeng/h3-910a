#!/usr/bin/env python3
"""Validate only the vision interpolation candidate on actual NPU pixels/weights."""
import ast
import json
from pathlib import Path
import time
from typing import Any
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT=Path('/tmp/h3_round54_capture01');CODE=Path('/tmp/h3_round54_validation01/candidate_vision_interpolation.py')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
OUT=ROOT/'npu_vision_interpolation_candidate.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite candidate validation')
torch.npu.set_device(0)
tree=ast.parse(CODE.read_text()); names={'_rotate_half','_apply_rotary_pos_emb_vision','_vision_sdpa_bf16_compatible','_NpuBF16CompatibleLayerNorm'}
selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (n.name in names or n.name.startswith('MiniMaxH3Qwen3VLVision'))]
ns={'torch':torch,'nn':nn,'F':F,'Any':Any}
exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(CODE),'exec'),ns)
config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config
inputs=torch.load(ROOT/'encoder_input.pt',weights_only=True);hf=torch.load(ROOT/'hf_vision_outputs.pt',weights_only=True)
old,olddeep=torch.load(ROOT/'vision_outputs.pt',weights_only=True)
report={'scope':'Actual CPU HF visual reference vs captured deployed NPU output and isolated candidate NPU output; not whole-model/video quality proof'}

def metric(a,b):
    x=a.detach().float().cpu();y=b.detach().float().cpu();assert x.shape==y.shape;e=x-y
    return {'shape':list(x.shape),'finite':bool(torch.isfinite(x).all()),'max_abs_error':float(e.abs().max()),
            'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))}
try:
    model=ns['MiniMaxH3Qwen3VLVisionModel'](config).to(torch.bfloat16).eval()
    wanted=set(dict(model.named_parameters()));state={};index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    for shard in sorted(set(index.values())):
        keys=[k for k,s in index.items() if s==shard and k.startswith('model.visual.') and k[len('model.visual.'):] in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
            for k in keys:state[k[len('model.visual.'):]]=f.get_tensor(k).to(torch.bfloat16)
    model.load_state_dict(state,strict=True)
    # CPU path must still reproduce the original interpolation formula exactly.
    original=ast.parse((CODE.parent/'source/encoder.py').read_text())
    fn=next(n for n in next(c for c in original.body if isinstance(c,ast.ClassDef) and c.name=='MiniMaxH3Qwen3VLVisionModel').body if isinstance(n,ast.FunctionDef) and n.name=='fast_pos_embed_interpolate')
    env={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'original_interp','exec'),env)
    with torch.inference_mode():
        a=model.fast_pos_embed_interpolate(inputs['image_grid_thw']);b=env['fast_pos_embed_interpolate'](model,inputs['image_grid_thw'])
    assert torch.equal(a,b);report['cpu_path_interpolation_bit_exact_unchanged']=True
    model.to('npu');pixels=inputs['pixel_values'].to('npu');grid=inputs['image_grid_thw'].to('npu')
    torch.npu.synchronize();start=time.perf_counter()
    with torch.inference_mode():out,deep=model(pixels,grid)
    torch.npu.synchronize();report['candidate_cold_forward_s']=time.perf_counter()-start
    report['old_merged_vs_hf']=metric(old,hf['merged']);report['new_merged_vs_hf']=metric(out,hf['merged'])
    report['old_deepstack_vs_hf']=[metric(a,b) for a,b in zip(olddeep,hf['deepstack'])]
    report['new_deepstack_vs_hf']=[metric(a,b) for a,b in zip(deep,hf['deepstack'])]
    report['completed']=True
    print(json.dumps(report,indent=2),flush=True)
except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(report,indent=2)+'\n');print('Evidence:',OUT,flush=True)
