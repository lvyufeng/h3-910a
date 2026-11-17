#!/usr/bin/env python3
"""Unchanged-BF16 HF language repeat using independently generated HF multimodal inputs.

Not shared NPU visual features: HF vision/embedding/positions/DeepStack from the
current independent full multimodal reference. This repeat certifies language
repeat stability and saves selected PRE-DeepStack decoder outputs to compare
current experimental captures. It is not a second independent vision repeat.
"""
import hashlib
import inspect
import json
from pathlib import Path
import signal
import time
import torch
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
from evidence_io74 import EvidenceWriter

ROOT=Path('/tmp/h3_round74_repeat01');HF=Path('/tmp/h3_round74_hf01');CAP=Path('/tmp/h3_round74_capture01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
writer=EvidenceWriter(ROOT/'language_repeat01.json');torch.set_num_threads(16)
r={'scope':__doc__,'layers':[],'selected':[],'fp32_linear_adapter_used':False,'completed':False,
   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'hf_source_sha256':hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest()}
FINAL=None


def stop(signum,frame):raise SystemExit('HF language repeat interrupted:'+str(signum))
signal.signal(signal.SIGTERM,stop)


def raw(v):return hashlib.sha256(memoryview(v.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def metric(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    x,y=a.contiguous().reshape(-1),b.contiguous().reshape(-1);err=0.;ref=0.;mx=0.;different=0;finite=True
    for s in range(0,x.numel(),1048576):
        xx,yy=x[s:s+1048576],y[s:s+1048576];different+=int((xx.view(torch.uint8)!=yy.view(torch.uint8)).sum())
        xd,yd=xx.double(),yy.double();valid=bool(torch.isfinite(xd).all()and torch.isfinite(yd).all());finite=finite and valid
        if valid:
            e=xd-yd;err+=float(e.square().sum());ref+=float(yd.square().sum());mx=max(mx,float(e.abs().max()))
    return {'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.numel(),'raw_bits_equal':different==0,'different_bytes':different,
            'finite':finite,'max_abs_error':mx if finite else None,'relative_rms':(err/max(ref,1e-30))**.5 if finite else None}


try:
    payload=torch.load(HF/'hf_multimodal_inputs.pt',map_location='cpu',weights_only=True)
    visual=payload['visual_pos_masks'];r['independent_HF_multimodal_input_sha256']=hashlib.sha256((HF/'hf_multimodal_inputs.pt').read_bytes()).hexdigest()
    config=Qwen3VLConfig.from_pretrained(str(MODEL)).text_config;config.num_hidden_layers=50;config.use_cache=False;config._attn_implementation='sdpa'
    with torch.device('meta'):model=hf.Qwen3VLTextModel(config)
    model.rotary_emb=hf.Qwen3VLTextRotaryEmbedding(config,device='cpu')
    wanted=set(dict(model.named_parameters()));index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];state={}
    for shard in sorted(set(index.values())):
        keys=[k for k,s in index.items()if s==shard and k.startswith('model.language_model.')and k[len('model.language_model.'):]in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu')as f:
            for k in keys:
                value=f.get_tensor(k);assert value.dtype==torch.bfloat16
                state[k[len('model.language_model.'):]]=value
    assert set(state)==wanted
    model.load_state_dict(state,strict=True,assign=True);del state;model.eval()
    assert all(p.dtype==torch.bfloat16 and not p.is_meta for p in model.parameters())and all(not b.is_meta for b in model.buffers())
    r['loaded_parameters']=len(wanted);r['original_hf_finalnorm_retained']=True
    def normpre(m,args):
        global FINAL
        FINAL=args[0].detach().clone()
    handles=[model.norm.register_forward_pre_hook(normpre)]
    starts={}
    def pre(i):
        def hook(m,args,kwargs):starts[i]=time.perf_counter()
        return hook
    def post(i):
        def hook(m,args,out):
            row={'layer':i,'cpu_wall_s':time.perf_counter()-starts[i],'sha256':raw(out),'finite':bool(torch.isfinite(out.float()).all())}
            r['layers'].append(row)
            if i in(0,1,2,9,24,49):
                original=torch.load(CAP/f'text_layer{i:02d}.pt',map_location='cpu',weights_only=True)
                compare={'layer':i,'boundary':'RawdecoderoutputPREmodelDeepStackinjectionat0/1/2',
                         'full':metric(original,out),'text':metric(original[~visual],out[~visual]),'visual':metric(original[visual],out[visual])}
                path=ROOT/f'hf_layer{i:02d}.pt'
                with path.open('xb')as f:torch.save(out.detach().clone(),f)
                compare['file_bytes']=path.stat().st_size;compare['file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                r['selected'].append(compare)
            writer.write(r);print(json.dumps(row),flush=True)
        return hook
    for i,layer in enumerate(model.layers):handles.extend((layer.register_forward_pre_hook(pre(i),with_kwargs=True),layer.register_forward_hook(post(i))))
    start=time.perf_counter()
    with torch.inference_mode():out=model(inputs_embeds=payload['inputs_embeds'],position_ids=payload['position_ids'],visual_pos_masks=visual,
                                         deepstack_visual_embeds=payload['deepstack_visual_embeds'],attention_mask=torch.ones((1,2043),dtype=torch.int64),use_cache=False)
    r['cpu_reference_wall_s']=time.perf_counter()-start
    assert FINAL is not None and len(r['layers'])==50
    path=ROOT/'hf_repeat_selected50.pt'
    with path.open('xb')as f:torch.save({'pre_final_norm':FINAL,'post_final_norm':out.last_hidden_state},f)
    r['repeat_final_sha256']=raw(FINAL)
    if(HF/'hf_selected50.pt').is_file():
        first=torch.load(HF/'hf_selected50.pt',map_location='cpu',weights_only=True)
        r['reference_repeat_stability']=metric(FINAL,first['pre_final_norm'])
        assert r['reference_repeat_stability']['raw_bits_equal']
    r['completed']=True
except BaseException as e:
    r['failure']={'type':type(e).__name__,'message':str(e)};raise
finally:writer.write(r)
