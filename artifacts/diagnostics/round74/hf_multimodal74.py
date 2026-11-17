#!/usr/bin/env python3
"""Independent unchanged-BF16 HF multimodal reference at H3's selected layer50.

Original model/input/checkpoint, genuine HF vision+embedding+M-RoPE+DeepStack;
no shared experimental visual inputs and no FP32-linear adaptation. Language
layers are truncated to50 because H3 selects their pre-final-norm hidden states.
The original final HF norm still executes; a pre-hook captures the correct H3
comparison boundary without replacing any operator or norm arithmetic.
CPU reference wall time is NOT 910A inference performance/official video quality.
"""
import hashlib
import inspect
import json
from pathlib import Path
import signal
import time
import torch
from torch import nn
from safetensors import safe_open
import transformers
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
from evidence_io74 import EvidenceWriter

ROOT=Path('/tmp/h3_round74_hf01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
INPUT=Path('/tmp/h3_round54_capture01/encoder_input.pt')
REFERENCE=Path('/tmp/h3_round70_components01/fresh_encoder_reference.pt')
writer=EvidenceWriter(ROOT/'hf_reference01.json')
torch.set_num_threads(16)
r={'scope':__doc__,'stages':{},'language_layers':[], 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'versions':{'torch':torch.__version__,'transformers':transformers.__version__},
   'reference_is_original_bf16_HF':True,'shared_native_visual_features_used':False,'fp32_linear_adapter_used':False,
   'full_multimodal_reference_completed':False,'official_goal_complete':False,'production_modified':False}
SAVED={};LANG_INPUTS={};PRENORM=None;PARAM_TENSORS={};LINEAR_INPUT_DTYPES=set()


def interrupted(signum,frame):raise SystemExit('CPU HF reference interrupted:'+str(signum))
signal.signal(signal.SIGTERM,interrupted)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def raw(value):
    cpu=value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()


def info(value):
    if isinstance(value,torch.Tensor):
        return {'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.numel()*value.element_size(),
                'sha256':raw(value),'finite':bool(torch.isfinite(value.float()).all())}
    if isinstance(value,(list,tuple)):return [info(item)for item in value]
    if isinstance(value,dict):return {name:info(item)for name,item in value.items()}
    if value is None:return None
    raise ValueError('Unexpected HF reference feature type')


def metric(actual,reference):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype
    a,b=actual.detach().cpu().contiguous(),reference.detach().cpu().contiguous()
    af,bf=a.reshape(-1),b.reshape(-1)
    bits=True;finite=True;err2=0.;ref2=0.;maximum=0.;different=0
    for start in range(0,a.numel(),1048576):
        x,y=af[start:start+1048576],bf[start:start+1048576]
        n=int((x.view(torch.uint8)!=y.view(torch.uint8)).sum());different+=n;bits=bits and n==0
        xd,yd=x.double(),y.double();valid=bool(torch.isfinite(xd).all()and torch.isfinite(yd).all());finite=finite and valid
        if valid:
            e=xd-yd;maximum=max(maximum,float(e.abs().max()));err2+=float(e.square().sum());ref2+=float(yd.square().sum())
    return {'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.numel(),'raw_bits_equal':bits,'different_bytes':different,
            'finite':finite,'max_abs_error':maximum if finite else None,'relative_rms':(err2/max(ref2,1e-30))**.5 if finite else None}


def parameters_sha(model):
    h=hashlib.sha256()
    for name,value in model.named_parameters():
        h.update(name.encode());h.update(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return h.hexdigest()


def save_payload(name,payload):
    path=ROOT/(name+'.pt')
    # Reference artifacts are bounded and write-once,never loader weights.
    size=sum(item['bytes']for item in _flatten_info(info(payload)))
    if size>256*1024*1024:raise ValueError('Reference payload exceeds bounded tensor bytes')
    with path.open('xb')as f:torch.save(payload,f)
    r['stages'][name]={'tensors':info(payload),'file_bytes':path.stat().st_size,'file_sha256':file_sha(path)}
    writer.write(r)


def _flatten_info(value):
    if isinstance(value,dict):
        if 'bytes'in value:return[value]
        return[item for child in value.values()for item in _flatten_info(child)]
    if isinstance(value,list):return[item for child in value for item in _flatten_info(child)]
    return[]


try:
    r['hf_modeling_path']=hf.__file__;r['hf_modeling_sha256']=file_sha(hf.__file__)
    r['torch_linear_source_path']=inspect.getsourcefile(nn.Linear)
    r['torch_linear_source_sha256']=file_sha(r['torch_linear_source_path'])
    inputs=torch.load(INPUT,map_location='cpu',weights_only=True)
    assert inputs['input_ids'].shape==(2043,)and inputs['pixel_values'].shape==(8064,1536)and inputs['pixel_values'].dtype==torch.float16
    assert inputs['image_grid_thw'].tolist()==[[1,48,84],[1,48,84]]and inputs['pixel_values_videos']is None and inputs['video_grid_thw']is None
    r['actual_inputs_before']=info(inputs);r['input_file_sha256']=file_sha(INPUT)
    comparison=torch.load(REFERENCE,map_location='cpu',weights_only=True)
    r['experimental_reference_path']=str(REFERENCE);r['experimental_reference_sha256']=file_sha(REFERENCE)
    config=Qwen3VLConfig.from_pretrained(str(MODEL),trust_remote_code=False)
    r['original_config_language_layers']=config.text_config.num_hidden_layers
    config.text_config.num_hidden_layers=50
    config.text_config.use_cache=False
    config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa'
    start=time.perf_counter()
    with torch.device('meta'):model=hf.Qwen3VLModel(config)
    # Restore only nonpersistent rotary buffers from their original constructors.
    # Assign checkpoint params directly,never cast full model buffers to BF16.
    model.visual.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding(config.vision_config.hidden_size//config.vision_config.num_heads//2)
    model.language_model.rotary_emb=hf.Qwen3VLTextRotaryEmbedding(config.text_config,device='cpu')
    wanted=set(dict(model.named_parameters()))
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    state={};loaded_keys=[]
    for shard in sorted(set(index.values())):
        keys=[key for key,filename in index.items()if filename==shard and key.startswith('model.')and key[len('model.'):]in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu')as f:
            for key in keys:
                value=f.get_tensor(key)
                assert value.dtype==torch.bfloat16
                state[key[len('model.'):]]=value;loaded_keys.append(key)
    assert set(state)==wanted
    model.load_state_dict(state,strict=True,assign=True);del state
    model.eval()
    assert len(model.language_model.layers)==50 and all(p.dtype==torch.bfloat16 and not p.is_meta and p.device.type=='cpu'for p in model.parameters())
    assert all(not b.is_meta for b in model.buffers())
    r['cpu_load_s']=time.perf_counter()-start;r['loaded_parameter_tensors']=len(wanted)
    r['loaded_original_checkpoint_keys']=loaded_keys
    r['complete_parameter_bytes']=sum(p.numel()*p.element_size()for p in model.parameters())
    r['parameter_dtypes']={dtype:sum(str(p.dtype)==dtype for p in model.parameters())for dtype in {str(p.dtype)for p in model.parameters()}}
    r['parameters_before']=parameters_sha(model);r['checkpoint_index_sha256']=file_sha(MODEL/'model.safetensors.index.json')
    r['unchanged_linear_modules']=sum(isinstance(m,nn.Linear)for m in model.modules())
    writer.write(r)
    handles=[]
    vision_start=[0.]
    def vision_pre(module,args,kwargs):vision_start[0]=time.perf_counter()
    def vision_post(module,args,out):
        payload={'merged':out.pooler_output.detach().clone(),'deepstack':[v.detach().clone()for v in out.deepstack_features]}
        SAVED['vision']=payload
        r['vision_reference_wall_s']=time.perf_counter()-vision_start[0]
        r['vision_vs_experimental']={'merged':metric(comparison['vision_output'][0],payload['merged']),
                                    'deepstack':[metric(a,b)for a,b in zip(comparison['vision_output'][1],payload['deepstack'])]}
        save_payload('hf_vision',payload)
        print(json.dumps({'stage':'hfvision','wall_s':r['vision_reference_wall_s'],'merged_vs_experimental':r['vision_vs_experimental']['merged']}),flush=True)
    def embedding_post(module,args,out):
        SAVED['embedding']=out.detach().clone()
        r['embedding_vs_experimental']=metric(comparison['embedding_output'],SAVED['embedding'])
    def text_pre(module,args,kwargs):
        payload={name:kwargs[name].detach().clone()if isinstance(kwargs[name],torch.Tensor)else[v.detach().clone()for v in kwargs[name]] for name in('inputs_embeds','position_ids','visual_pos_masks','deepstack_visual_embeds')}
        LANG_INPUTS.update(payload)
        r['multimodal_text_input_vs_experimental']={name:metric(comparison['text_inputs'][old],payload[name])for name,old in(('inputs_embeds','inputs_embeds'),('position_ids','positions'),('visual_pos_masks','visual_pos_masks'))}
        r['multimodal_text_input_vs_experimental']['deepstack']=[metric(a,b)for a,b in zip(comparison['text_inputs']['deepstack_visual_embeds'],payload['deepstack_visual_embeds'])]
        save_payload('hf_multimodal_inputs',payload)
        print(json.dumps({'stage':'hftextinputs','input_rms':r['multimodal_text_input_vs_experimental']['inputs_embeds']['relative_rms'],
                          'positions_bits':r['multimodal_text_input_vs_experimental']['position_ids']['raw_bits_equal']}),flush=True)
    def norm_pre(module,args):
        global PRENORM
        PRENORM=args[0].detach().clone()
    layer_start={}
    def pre_layer(i):
        def hook(module,args,kwargs):layer_start[i]=time.perf_counter()
        return hook
    def post_layer(i):
        def hook(module,args,out):
            row={'layer':i,'cpu_reference_wall_s':time.perf_counter()-layer_start[i],'output':info(out)}
            r['language_layers'].append(row);writer.write(r)
            print(json.dumps({'stage':'hflayer','layer':i,'wall_s':row['cpu_reference_wall_s'],'finite':row['output']['finite']}),flush=True)
        return hook
    def linear_pre(module,args):
        LINEAR_INPUT_DTYPES.add(str(args[0].dtype))
        assert args[0].dtype==torch.bfloat16 and module.weight.dtype==torch.bfloat16
    handles.append(model.visual.register_forward_pre_hook(vision_pre,with_kwargs=True))
    handles.append(model.visual.register_forward_hook(vision_post))
    handles.append(model.language_model.embed_tokens.register_forward_hook(embedding_post))
    handles.append(model.language_model.register_forward_pre_hook(text_pre,with_kwargs=True))
    handles.append(model.language_model.norm.register_forward_pre_hook(norm_pre))
    for i,layer in enumerate(model.language_model.layers):
        handles.append(layer.register_forward_pre_hook(pre_layer(i),with_kwargs=True));handles.append(layer.register_forward_hook(post_layer(i)))
    for module in model.modules():
        if isinstance(module,nn.Linear):handles.append(module.register_forward_pre_hook(linear_pre))
    ids=inputs['input_ids'][None]
    mm=torch.zeros_like(ids,dtype=torch.int32);mm[ids==config.image_token_id]=1;mm[ids==config.video_token_id]=2
    start=time.perf_counter()
    with torch.inference_mode():
        output=model(input_ids=ids,pixel_values=inputs['pixel_values'],image_grid_thw=inputs['image_grid_thw'],
                     attention_mask=torch.ones_like(ids),mm_token_type_ids=mm,use_cache=False,return_dict=True)
    r['full_cpu_reference_wall_s']=time.perf_counter()-start
    assert PRENORM is not None and PRENORM.shape==(1,2043,5120)and len(r['language_layers'])==50
    r['full_multimodal_pre_final_norm_vs_experimental']=metric(comparison['complete_output'],PRENORM[0])
    r['reference_comparison_boundary']='Original HF 50 retained decoder layers,pre-final-HF-norm. Finalnormstill executes unchanged but is not H3selectedboundary.'
    r['linear_input_dtypes_observed']=sorted(LINEAR_INPUT_DTYPES)
    save_payload('hf_selected50',{'pre_final_norm':PRENORM,'hf_post_final_norm':output.last_hidden_state})
    for handle in handles:handle.remove()
    r['parameters_after']=parameters_sha(model);r['actual_inputs_after']=info(inputs)
    assert r['parameters_before']==r['parameters_after']and r['actual_inputs_before']==r['actual_inputs_after']
    r['full_multimodal_reference_completed']=True;r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)}
    r['completed_language_layers']=len(r['language_layers'])
    raise
finally:
    writer.write(r)
