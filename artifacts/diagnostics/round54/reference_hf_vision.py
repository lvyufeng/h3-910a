#!/usr/bin/env python3
"""Independent installed HF CPU vision tower and multimodal position reference.

Consumes real processor tensors and original visual checkpoint weights. No native
visual embeddings are used to compute the reference. Per-feature comparisons are
on full outputs, not sampled query rows. Parameters/I/O keep BF16 as deployed.
"""
import json
from pathlib import Path
import time
import types
import torch
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel,Qwen3VLVisionRotaryEmbedding,Qwen3VLModel

ROOT=Path('/tmp/h3_round54_capture01'); MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
OUT=ROOT/'hf_vision_reference.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite reference evidence')
for name in ('encoder_input','vision_outputs','text_inputs'):
    if not (ROOT/(name+'.json')).exists():raise RuntimeError('Capture not complete: '+name)
torch.set_num_threads(16)
inputs=torch.load(ROOT/'encoder_input.pt',weights_only=True)
text=torch.load(ROOT/'text_inputs.pt',weights_only=True)
actual,actual_deep=torch.load(ROOT/'vision_outputs.pt',weights_only=True)
report={'scope':'Independent installed HF CPU BF16 visual checkpoint on actual pixel/grid inputs and multimodal positions; not end-to-end generation-quality proof'}

def metrics(a,b):
    assert a.shape==b.shape
    x=a.float();y=b.float();e=x-y;finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    return {'shape':list(x.shape),'finite':finite,'max_abs_error':float(e.abs().max()) if finite else None,
            'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)) if finite else None,
            'reference_rms':float(y.square().mean().sqrt()) if finite else None}

try:
    conf=Qwen3VLConfig.from_pretrained(str(MODEL),trust_remote_code=False)
    conf.vision_config._attn_implementation='sdpa'
    start=time.perf_counter()
    with torch.device('meta'):model=Qwen3VLVisionModel(conf.vision_config)
    model.rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(conf.vision_config.hidden_size//conf.vision_config.num_heads//2)
    wanted=set(dict(model.named_parameters())); state={}
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    for shard in sorted(set(index.values())):
        keys=[k for k,s in index.items() if s==shard and k.startswith('model.visual.') and k[len('model.visual.'):] in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
            for key in keys:state[key[len('model.visual.'):]]=f.get_tensor(key).to(torch.bfloat16)
    missing,unexpected=model.load_state_dict(state,strict=False,assign=True)
    assert not missing and not unexpected,(missing,unexpected)
    assert all(not p.is_meta for p in model.parameters());model.eval()
    report['parameter_tensors']=len(state);report['load_s']=time.perf_counter()-start
    report['pixel_shape']=list(inputs['pixel_values'].shape); report['grid_thw']=inputs['image_grid_thw'].tolist()
    start=time.perf_counter()
    with torch.inference_mode():out=model(inputs['pixel_values'],inputs['image_grid_thw'],return_dict=True)
    report['forward_s']=time.perf_counter()-start
    report['merged_output']=metrics(actual,out.pooler_output)
    report['deepstack']=[metrics(a,b) for a,b in zip(actual_deep,out.deepstack_features)]
    assert len(actual_deep)==len(out.deepstack_features)==3
    # Recompute multimodal positions via the installed HF helper (not native port).
    owner=types.SimpleNamespace(config=conf)
    owner.get_vision_position_ids=types.MethodType(Qwen3VLModel.get_vision_position_ids,owner)
    ids=inputs['input_ids'].unsqueeze(0)
    mm=torch.zeros_like(ids);mm[ids==conf.image_token_id]=1;mm[ids==conf.video_token_id]=2
    positions,delta=Qwen3VLModel.get_rope_index(owner,ids,mm,image_grid_thw=inputs['image_grid_thw'])
    report['multimodal_positions_exact']=bool(torch.equal(positions,text['positions']))
    report['multimodal_positions_max_error']=int((positions-text['positions']).abs().max())
    save=ROOT/'hf_vision_outputs.pt'
    if save.exists():raise RuntimeError('Refuse to overwrite HF output tensors')
    with save.open('xb') as f:torch.save({'merged':out.pooler_output,'deepstack':out.deepstack_features,'positions':positions},f)
    report['completed']=True
    print(json.dumps(report,indent=2),flush=True)
except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Evidence:',OUT,flush=True)
