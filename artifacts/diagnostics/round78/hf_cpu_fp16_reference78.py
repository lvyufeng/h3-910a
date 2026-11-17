#!/usr/bin/env python3
"""Independent official-HF CPU FP16 full multimodal encoder reference.

Convert original dense checkpoint parameters to FP16,then execute original HF
vision/embedding/M-RoPE/DeepStack and fifty decoder layers on CPU. Preserve HF's
normal FP32 reduction/rotary-buffer islands;no FP32 Linear adapter,no BF16
compute mandate,no reused BF16 outputs and no optimized AscendC/TBE operators.
The captured pre-final-norm layer50 is H3's selected boundary. CPU cost is not
910A performance. Reference generation alone is not NPU precision acceptance.
"""
import collections
import hashlib
import inspect
import json
from pathlib import Path
import signal
import time
import torch
from torch import nn
from torch.nn.attention import SDPBackend,sdpa_kernel
from safetensors import safe_open
import transformers
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
from evidence_io78 import EvidenceWriter

ROOT=Path('/tmp/h3_round78_hf_fp1601')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
INPUT=Path('/tmp/h3_round54_capture01/encoder_input.pt')
writer=EvidenceWriter(ROOT/'reference.json')
r={'scope':__doc__,'stages':{},'vision_layers':[],'language_layers':[],
   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'versions':{'torch':torch.__version__,'transformers':transformers.__version__},
   'reference_is_independent_HF_CPU_FP16':True,'BF16_output_agreement_required':False,
   'shared_experimental_visual_features_used':False,'FP32_linear_adapter_used':False,
   'old_BF16_outputs_reused_or_cast_as_gold':False,'model_loads_on_cpu_only':True,
   'full_multimodal_reference_completed':False,'NPU_precision_accepted':False,'whole_goal_complete':False}
SAVED={};PRENORM=None;LINEAR_INPUT_DTYPES=set();LINEAR_OUTPUT_DTYPES=set();LAYER_COUNTS=collections.Counter();handles=[]


def interrupted(signum,frame):raise SystemExit('CPU FP16 reference interrupted:'+str(signum))
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
        result={'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.numel()*value.element_size(),
                'sha256':raw(value),'finite':bool(torch.isfinite(value.float()).all())}
        if value.is_floating_point():result['max_abs']=float(value.detach().float().abs().max())if value.numel()else 0.0
        return result
    if isinstance(value,(list,tuple)):return[info(item)for item in value]
    if isinstance(value,dict):return{name:info(item)for name,item in value.items()}
    if value is None:return None
    raise ValueError('Unexpected reference feature')


def flatten(value):
    if isinstance(value,dict):
        if'bytes'in value:return[value]
        return[item for child in value.values()for item in flatten(child)]
    if isinstance(value,list):return[item for child in value for item in flatten(child)]
    return[]


def save_payload(name,payload):
    identities=info(payload);size=sum(item['bytes']for item in flatten(identities))
    assert size<256*1024*1024 and all(item['finite']for item in flatten(identities))
    path=ROOT/(name+'.pt')
    with path.open('xb')as f:torch.save(payload,f)
    r['stages'][name]={'tensors':identities,'file_bytes':path.stat().st_size,'file_sha256':file_sha(path)}
    writer.write(r)


def params_sha(model):
    h=hashlib.sha256()
    for name,parameter in model.named_parameters():h.update(name.encode());h.update(memoryview(parameter.detach().contiguous().view(torch.uint8).numpy()))
    return h.hexdigest()


try:
    torch.set_num_threads(16);r['CPU_num_threads']=torch.get_num_threads();r['CPU_mkldnn_enabled']=torch.backends.mkldnn.enabled
    preflight=json.loads((ROOT/'preflight.json').read_text())
    math_preflight=json.loads((ROOT/'math_preflight.json').read_text())
    r['preflight_file_sha256']=file_sha(ROOT/'preflight.json');r['math_preflight_file_sha256']=file_sha(ROOT/'math_preflight.json')
    assert preflight['completed']and preflight['all_required_preflight_operators_supported']
    assert math_preflight['completed']and math_preflight['supported']and math_preflight['backend']=='SDPBackend.MATH'
    assert math_preflight['torch']==torch.__version__
    assert preflight['versions']['torch']==torch.__version__ and preflight['versions']['transformers']==transformers.__version__
    assert file_sha(hf.__file__)==preflight['HF_modeling_source_sha256']=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
    r['HF_modeling_source_path']=hf.__file__;r['HF_modeling_source_sha256']=file_sha(hf.__file__)
    r['torch_Linear_source_sha256']=file_sha(inspect.getsourcefile(nn.Linear))
    assert file_sha(INPUT)=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
    inputs=torch.load(INPUT,map_location='cpu',weights_only=True)
    assert inputs['input_ids'].shape==(2043,)and inputs['pixel_values'].shape==(8064,1536)and inputs['pixel_values'].dtype==torch.float16
    assert inputs['image_grid_thw'].tolist()==[[1,48,84],[1,48,84]]and inputs['pixel_values_videos']is None and inputs['video_grid_thw']is None
    r['actual_inputs_before']=info(inputs);r['input_file_sha256']=file_sha(INPUT)
    config=Qwen3VLConfig.from_pretrained(str(MODEL),trust_remote_code=False,local_files_only=True)
    assert file_sha(MODEL/'config.json')=='d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427'
    assert file_sha(MODEL/'model.safetensors.index.json')=='06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a'
    r['original_language_layer_count']=config.text_config.num_hidden_layers
    config.text_config.num_hidden_layers=50;config.text_config.use_cache=False
    config.dtype=torch.float16;config.text_config.dtype=torch.float16;config.vision_config.dtype=torch.float16
    config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa'
    start=time.perf_counter()
    with torch.device('meta'):model=hf.Qwen3VLModel(config)
    model.visual.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding(config.vision_config.hidden_size//config.vision_config.num_heads//2)
    model.language_model.rotary_emb=hf.Qwen3VLTextRotaryEmbedding(config.text_config,device='cpu')
    wanted=dict(model.named_parameters());index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    original_state={};r['checkpoint_shards']={}
    for shard in sorted(set(index.values())):
        keys=[key for key,filename in index.items()if filename==shard and key.startswith('model.')and key[len('model.'):]in wanted]
        if not keys:continue
        r['checkpoint_shards'][shard]={'bytes':(MODEL/shard).stat().st_size,'sha256':file_sha(MODEL/shard)}
        with safe_open(str(MODEL/shard),framework='pt',device='cpu')as f:
            for key in keys:
                value=f.get_tensor(key);assert value.dtype==torch.bfloat16
                original_state[key[len('model.'):]]=value
    assert set(original_state)==set(wanted)
    source_sha=hashlib.sha256()
    for name in wanted:source_sha.update(name.encode());source_sha.update(memoryview(original_state[name].contiguous().view(torch.uint8).numpy()))
    r['original_selected_parameter_sha256']=source_sha.hexdigest()
    assert r['original_selected_parameter_sha256']=='a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e'
    state={};r['parameter_conversion']={};total_changed=0
    for name in wanted:
        original=original_state.pop(name);converted=original.to(torch.float16)
        finite=bool(torch.isfinite(converted).all());assert finite,'OriginalparameterisnotfinitewhenconvertedtoFP16:'+name
        changed=0
        a=original.reshape(-1);b=converted.reshape(-1)
        for begin in range(0,a.numel(),16*1024*1024):
            end=begin+16*1024*1024
            changed+=int((a[begin:end].view(torch.int16)!=b[begin:end].to(torch.bfloat16).view(torch.int16)).sum())
        total_changed+=changed
        r['parameter_conversion'][name]={'shape':list(original.shape),'original_dtype':str(original.dtype),'reference_dtype':str(converted.dtype),
                                          'reference_sha256':raw(converted),'reference_finite':finite,'BF16_roundtrip_changed_elements_diagnostic_only':changed}
        state[name]=converted
    assert not original_state
    model.load_state_dict(state,strict=True,assign=True);del state,original_state
    model.eval();assert len(model.language_model.layers)==50
    assert all(p.dtype==torch.float16 and not p.is_meta and p.device.type=='cpu'for p in model.parameters())
    assert all(not b.is_meta and b.device.type=='cpu'for b in model.buffers())
    r['parameter_dtypes']=dict(collections.Counter(str(p.dtype)for p in model.parameters()));r['buffer_dtypes']=dict(collections.Counter(str(b.dtype)for b in model.buffers()))
    r['BF16_roundtrip_parameter_changes_not_an_acceptance_gate']=total_changed
    r['loaded_parameter_tensors']=len(wanted);r['complete_parameter_bytes']=sum(p.numel()*p.element_size()for p in model.parameters())
    r['CPU_weight_load_convert_s_excluded']=time.perf_counter()-start;r['parameters_before']=params_sha(model)
    r['unchanged_HF_Linear_modules']=sum(isinstance(module,nn.Linear)for module in model.modules())
    r['FP32_islands']='OriginalHF RMSNorm reduction,rotary construction/trigonometry and visionrotary application. Textrotary cos/sin returnFP16 as originalHF. Linearaccumulationislibrary-dependentandnotinstrumented;SDPBackend.MATH usesdocumentedFP32 intermediates. No explicitF32LinearadapterorBF16activationconversion.'
    writer.write(r)
    vision_start=[0.]
    def vision_pre(module,args,kwargs):vision_start[0]=time.perf_counter()
    def vision_post(module,args,out):
        payload={'merged':out.pooler_output.detach().clone(),'deepstack':[v.detach().clone()for v in out.deepstack_features]}
        SAVED['vision']=payload;r['vision_reference_wall_s']=time.perf_counter()-vision_start[0];save_payload('FP16_vision',payload)
        print(json.dumps({'stage':'CPU_FP16_vision','wall_s':r['vision_reference_wall_s'],'finite':info(payload['merged'])['finite']}),flush=True)
    def embedding_post(module,args,out):SAVED['embedding']=out.detach().clone();r['embedding_output']=info(SAVED['embedding'])
    def text_pre(module,args,kwargs):
        payload={name:kwargs[name].detach().clone()if isinstance(kwargs[name],torch.Tensor)else[v.detach().clone()for v in kwargs[name]]for name in('inputs_embeds','position_ids','visual_pos_masks','deepstack_visual_embeds')}
        save_payload('FP16_multimodal_inputs',payload)
    def norm_pre(module,args):
        global PRENORM
        PRENORM=args[0].detach().clone()
    starts={}
    def layer_pre(kind,index):
        def hook(module,args,kwargs):starts[(kind,index)]=time.perf_counter()
        return hook
    def layer_post(kind,index):
        def hook(module,args,out):
            row={'layer':index,'CPU_wall_s':time.perf_counter()-starts[(kind,index)],'output':info(out),'boundary':'rawdecoderPREDeepStackinjection'if kind=='language'else'visionblock'}
            assert row['output']['finite']and row['output']['dtype']=='torch.float16'
            r[kind+'_layers'].append(row);LAYER_COUNTS[kind]+=1
            if kind=='language'and index in(0,1,2,9,24,49):save_payload('FP16_layer%02d'%index,out.detach().clone())
            writer.write(r);print(json.dumps({'stage':'CPU_FP16_'+kind,'layer':index,'wall_s':row['CPU_wall_s'],'finite':row['output']['finite']}),flush=True)
        return hook
    def linear_pre(module,args):LINEAR_INPUT_DTYPES.add(str(args[0].dtype))
    def linear_post(module,args,out):
        LINEAR_OUTPUT_DTYPES.add(str(out.dtype))
        if not torch.isfinite(out).all():
            r['first_nonfinite_Linear']={'module':LINEAR_NAMES[id(module)],'input':info(args[0]),'output':info(out)}
            writer.write(r)
            raise AssertionError('OriginalHF CPUFP16 Linear produced nonfinite output:'+LINEAR_NAMES[id(module)])
    LINEAR_NAMES={id(module):name for name,module in model.named_modules()if isinstance(module,nn.Linear)}
    handles.extend((model.visual.register_forward_pre_hook(vision_pre,with_kwargs=True),model.visual.register_forward_hook(vision_post),
                    model.language_model.embed_tokens.register_forward_hook(embedding_post),model.language_model.register_forward_pre_hook(text_pre,with_kwargs=True),model.language_model.norm.register_forward_pre_hook(norm_pre)))
    for index,layer in enumerate(model.language_model.layers):handles.extend((layer.register_forward_pre_hook(layer_pre('language',index),with_kwargs=True),layer.register_forward_hook(layer_post('language',index))))
    for index,layer in enumerate(model.visual.blocks):handles.extend((layer.register_forward_pre_hook(layer_pre('vision',index),with_kwargs=True),layer.register_forward_hook(layer_post('vision',index))))
    for module in model.modules():
        if isinstance(module,nn.Linear):handles.extend((module.register_forward_pre_hook(linear_pre),module.register_forward_hook(linear_post)))
    ids=inputs['input_ids'][None];mask=torch.ones_like(ids)
    mm=torch.zeros_like(ids,dtype=torch.int32)
    mm[ids==config.image_token_id]=1;mm[ids==config.video_token_id]=2
    r['multimodal_token_type_ids']=info(mm)
    r['comparison_SDPA_backend']='PyTorchSDPBackend.MATH;FP16input/output,standardFP32intermediateaccumulation;noCPUflashoroptionalexternalkernels'
    start=time.perf_counter()
    with torch.inference_mode(),sdpa_kernel(backends=[SDPBackend.MATH]):
        output=model(input_ids=ids,attention_mask=mask,pixel_values=inputs['pixel_values'],image_grid_thw=inputs['image_grid_thw'],mm_token_type_ids=mm,use_cache=False)
    r['CPU_reference_compute_wall_s']=time.perf_counter()-start
    assert PRENORM is not None and PRENORM.shape==(1,2043,5120)and PRENORM.dtype==torch.float16
    assert LAYER_COUNTS['language']==50 and LAYER_COUNTS['vision']==len(model.visual.blocks)
    assert LINEAR_INPUT_DTYPES=={'torch.float16'}and LINEAR_OUTPUT_DTYPES=={'torch.float16'}
    r['Linear_input_dtypes_observed']=sorted(LINEAR_INPUT_DTYPES);r['Linear_output_dtypes_observed']=sorted(LINEAR_OUTPUT_DTYPES)
    r['comparison_boundary']='OriginalHFselected50decoderlayersPREfinalnorm;originalfinalnormexecutesunchanged,capturedviaprehook'
    r['pre_final_norm_matches_layer49_identity']=raw(PRENORM)==r['language_layers'][-1]['output']['sha256']
    assert r['pre_final_norm_matches_layer49_identity']
    save_payload('FP16_selected50',{'pre_final_norm':PRENORM,'HF_post_final_norm':output.last_hidden_state})
    first_pre=PRENORM.clone();first_post=output.last_hidden_state.detach().clone()
    for handle in handles:handle.remove()
    handles.clear();r['first_full_reference_completed_repeat_pending']=True;writer.write(r)
    handles.append(model.language_model.norm.register_forward_pre_hook(norm_pre))
    start=time.perf_counter();print(json.dumps({'stage':'CPU_FP16_full_repeat_started'}),flush=True)
    with torch.inference_mode(),sdpa_kernel(backends=[SDPBackend.MATH]):
        repeat=model(input_ids=ids,attention_mask=mask,pixel_values=inputs['pixel_values'],image_grid_thw=inputs['image_grid_thw'],mm_token_type_ids=mm,use_cache=False)
    r['CPU_full_repeat_wall_s']=time.perf_counter()-start
    r['repeat_pre_final_norm_bits_equal']=torch.equal(first_pre.contiguous().view(torch.uint8),PRENORM.contiguous().view(torch.uint8))
    r['repeat_post_final_norm_bits_equal']=torch.equal(first_post.contiguous().view(torch.uint8),repeat.last_hidden_state.contiguous().view(torch.uint8))
    assert r['repeat_pre_final_norm_bits_equal']and r['repeat_post_final_norm_bits_equal']
    r['repeat_pre_final_norm']=info(PRENORM)
    for handle in handles:handle.remove()
    handles.clear();r['parameters_after']=params_sha(model);r['actual_inputs_after']=info(inputs)
    assert r['parameters_before']==r['parameters_after']and r['actual_inputs_before']==r['actual_inputs_after']
    r['full_multimodal_reference_completed']=True;r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};r['completed_language_layers']=len(r['language_layers']);raise
finally:
    for handle in handles:handle.remove()
    writer.write(r)
