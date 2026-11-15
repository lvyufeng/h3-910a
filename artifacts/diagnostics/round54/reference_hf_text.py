#!/usr/bin/env python3
"""Installed HF Qwen layers on CPU, same real embeddings/positions/DeepStack.

Using the captured native vision embeddings isolates the language-model port;
it is not an independent validation of the image encoder. Compare pre-DeepStack
layer outputs, exactly where the native layer hooks were attached. No final LM
norm is compared to H3's intentionally unnormalized layer-50 output.
"""
import argparse
import json
from pathlib import Path
import signal
import time
import torch
from torch import nn
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel,Qwen3VLTextRotaryEmbedding

p=argparse.ArgumentParser();p.add_argument('--layers',type=int,default=3);a=p.parse_args()
assert 1<=a.layers<=50
ROOT=Path('/tmp/h3_round54_capture01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
OUT=ROOT/f'hf_text_reference_layers{a.layers}.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite a reference result')
if not (ROOT/'text_inputs.json').exists():raise RuntimeError('Text input capture is not ready')
torch.set_num_threads(16)
inputs=torch.load(ROOT/'text_inputs.pt',weights_only=True)
ids=torch.load(ROOT/'encoder_input.pt',weights_only=True)['input_ids']
report={'layers':a.layers,'comparisons':[],
        'scope':'Real installed HF CPU BF16 language layers with shared captured native vision/DeepStack inputs; not complete independent encoder or full-model quality validation'}

def interrupted(signum, frame):
    raise SystemExit("CPU HF reference interrupted by signal %d" % signum)


signal.signal(signal.SIGTERM, interrupted)


def metrics(x,y):
    assert x.shape==y.shape
    xf=x.float(); yf=y.float(); diff=xf-yf
    finite=bool(torch.isfinite(xf).all()) and bool(torch.isfinite(yf).all())
    return {'shape':list(x.shape),'finite':finite,'reference_rms':float(yf.square().mean().sqrt()) if finite else None,
            'max_abs_error':float(diff.abs().max()) if finite else None,
            'relative_rms':float(diff.square().mean().sqrt()/yf.square().mean().sqrt().clamp_min(1e-12)) if finite else None}

try:
    config=Qwen3VLConfig.from_pretrained(str(MODEL),trust_remote_code=False).text_config
    config.num_hidden_layers=a.layers;config.use_cache=False;config._attn_implementation='sdpa'
    start=time.perf_counter()
    with torch.device('meta'):model=Qwen3VLTextModel(config)
    # Forward uses captured inputs_embeds. Avoid allocating unused vocabulary
    # parameters in this teacher; compare actual embedding rows separately.
    model.embed_tokens=nn.Identity()
    model.rotary_emb=Qwen3VLTextRotaryEmbedding(config)
    wanted=set(dict(model.named_parameters()))
    state={}; loaded_sources=[]
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    for shard in sorted(set(index.values())):
        keys=[key for key,fn in index.items() if fn==shard and key.startswith('model.language_model.') and key[len('model.language_model.'):] in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
            for key in keys:
                state[key[len('model.language_model.'):]]=f.get_tensor(key).to(torch.bfloat16)
        loaded_sources.append(shard)
    missing,unexpected=model.load_state_dict(state,strict=False,assign=True)
    assert not missing and not unexpected,(missing,unexpected)
    assert all(not value.is_meta for value in model.parameters())
    model.eval()
    report['loaded_parameter_tensors']=len(state);report['loaded_shards']=loaded_sources
    report['load_wall_s']=time.perf_counter()-start
    embed_key='model.language_model.embed_tokens.weight'
    with safe_open(str(MODEL/index[embed_key]),framework='pt',device='cpu') as f:
        embedding=torch.nn.functional.embedding(ids.unsqueeze(0),f.get_tensor(embed_key).to(torch.bfloat16))
    native_embedding=torch.load(ROOT/'embedding.pt',weights_only=True)
    report['embedding']=metrics(native_embedding,embedding)
    native_rope=torch.load(ROOT/'text_rotary.pt',weights_only=True)
    with torch.inference_mode():hf_rope=model.rotary_emb(inputs['inputs_embeds'],inputs['positions'])
    report['rotary']=[metrics(x,y) for x,y in zip(native_rope,hf_rope)]
    visual_mask=inputs.get('visual_pos_masks')
    def hook(index):
        def compare(module,args,out):
            file=ROOT/f'text_layer_{index:02d}.pt'
            if not file.exists():return
            native=torch.load(file,weights_only=True)
            item={'layer':index,**metrics(native,out)}
            if visual_mask is not None:
                item['text_positions']=metrics(native[~visual_mask],out[~visual_mask])
                item['visual_positions']=metrics(native[visual_mask],out[visual_mask])
            report['comparisons'].append(item)
            # Preserve partial progress even if a long CPU reference reaches
            # its deadline. These pre-DeepStack comparison checkpoints do not
            # substitute for a complete layer-50 output.
            OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps(item),flush=True)
        return compare
    for i,layer in enumerate(model.layers):layer.register_forward_hook(hook(i))
    start=time.perf_counter()
    with torch.inference_mode():
        model(inputs_embeds=inputs['inputs_embeds'],position_ids=inputs['positions'],
              visual_pos_masks=visual_mask,deepstack_visual_embeds=inputs.get('deepstack_visual_embeds'),use_cache=False)
    report['forward_wall_s']=time.perf_counter()-start
    report['input_shape']=list(inputs['inputs_embeds'].shape)
    report['input_dtype']=str(inputs['inputs_embeds'].dtype)
    report['completed']=True
except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Evidence:',OUT,flush=True)
