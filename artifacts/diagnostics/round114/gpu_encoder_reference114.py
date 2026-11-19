#!/usr/bin/env python3
"""Original-HF 4090 FP16 and BF16 full multimodal selected-layer-50 references.

Whole unsharded Linear weights reside across four GPUs by layer placement, not TP.
Real images, M-RoPE, DeepStack and all fifty decoder layers execute. The original
HF final norm executes unchanged, while H3's pre-final-norm boundary is captured.
No CPU inference, distilled/quantized model, hidden-state cache or fake norm.
"""
from pathlib import Path
import collections,gc,hashlib,json,time


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def info(value):
    import torch
    x=value.detach().cpu().contiguous()
    return {'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size(),
            'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'finite':bool(torch.isfinite(x).all()),'max_abs':float(x.float().abs().max())if x.numel()else 0.0}


def metric(candidate,reference):
    import torch
    a=candidate.detach().to('cuda:0').double();b=reference.detach().to('cuda:0').double()
    assert a.shape==b.shape and bool(torch.isfinite(a).all())and bool(torch.isfinite(b).all())
    d=a-b;return {'relative_rms':float(d.square().mean().sqrt()/b.square().mean().sqrt()),
                 'max_abs_error':float(d.abs().max()),'candidate_dtype':str(candidate.dtype),'reference_dtype':str(reference.dtype),
                 'reference_not_cast_to_FP16':True,'numerical_tolerance':None}


def main():
    import torch
    from torch import nn
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import transformers
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    from accelerate import dispatch_model
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from evidence_io114 import EvidenceWriter
    root=Path('/tmp/h3_gpu114_01');checkpoint=root/'encoder';data=root/'projection'
    writer=EvidenceWriter(root/'GPU_encoder_reference114.json')
    result={'scope':__doc__,'completed':False,'whole_goal_complete':False,'full_encoder_precision_accepted':False,
            'numerical_tolerance':None,'versions':{'torch':torch.__version__,'transformers':transformers.__version__},'cases':{}}
    writer.write(result);handles=[];model=None
    try:
        assert torch.cuda.device_count()==8 and all('RTX 4090'in torch.cuda.get_device_name(i)for i in range(4))
        assert torch.cuda.is_bf16_supported()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        assert file_sha(hf.__file__)=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
        packed=json.loads((checkpoint/'package_manifest.json').read_text());assert packed['completed']and packed['kind']=='encoder'
        assert packed['original_parameter_aggregate_SHA256']=='a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e'
        assert len(packed['parameters'])==903
        for name,row in packed['files'].items():assert file_sha(checkpoint/name)==row['sha256']
        assert file_sha(checkpoint/'config.json')=='d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427'
        package=json.loads((data/'package_manifest.json').read_text());fixture_path=data/package['encoder_fixture']['relative_path']
        assert file_sha(fixture_path)==package['encoder_fixture']['file_SHA256']=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
        fixture=torch.load(fixture_path,map_location='cpu',weights_only=True)
        assert fixture['input_ids'].shape==(2043,)and fixture['pixel_values'].shape==(8064,1536)
        assert fixture['pixel_values'].dtype==torch.float16 and fixture['image_grid_thw'].tolist()==[[1,48,84],[1,48,84]]
        assert fixture['pixel_values_videos']is None and fixture['video_grid_thw']is None
        npu_outputs={}
        for name,row in package['precision_inputs'].items():
            if name=='CPU_original':continue
            p=data/row['relative_path'];assert file_sha(p)==row['file_SHA256']
            x=torch.load(p,map_location='cpu',weights_only=True)[row['tensor_key']]
            assert info(x)['sha256']==row['tensor_SHA256'];npu_outputs[name]=x
        result['BF16_parameters_direct_original_not_FP16_roundtrip']=True
        result['originalHF_source_SHA256']=file_sha(hf.__file__)
        device_map={'visual':0,'language_model.embed_tokens':0,'language_model.rotary_emb':0,'language_model.norm':3}
        for i in range(50):device_map['language_model.layers.%d'%i]=min(i//13,3)
        result['device_map']=device_map
        def owner(name):
            matches=[(k,v)for k,v in device_map.items()if name==k or name.startswith(k+'.')]
            assert matches,name
            return max(matches,key=lambda pair:len(pair[0]))[1]
        def synchronize():
            for device in range(4):torch.cuda.synchronize(device)
        for label,dtype in(('FP16',torch.float16),('BF16',torch.bfloat16)):
            config=Qwen3VLConfig.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
            original_layers=config.text_config.num_hidden_layers;assert original_layers==64
            config.text_config.num_hidden_layers=50;config.text_config.use_cache=False
            config.dtype=dtype;config.text_config.dtype=dtype;config.vision_config.dtype=dtype
            config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa'
            with torch.device('meta'):model=hf.Qwen3VLModel(config)
            model.visual.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding(config.vision_config.hidden_size//config.vision_config.num_heads//2)
            model.language_model.rotary_emb=hf.Qwen3VLTextRotaryEmbedding(config.text_config,device='cpu')
            assert set(dict(model.named_parameters()))==set(packed['parameters'])
            for name,row in packed['parameters'].items():
                with safe_open(str(checkpoint/row['packed_file']),framework='pt',device='cpu')as f:original=f.get_tensor(name)
                assert info(original)['sha256']==row['original_BF16_SHA256']and original.dtype==torch.bfloat16
                value=original.to(dtype=dtype)
                if label=='FP16':assert info(value)['sha256']==row['FP16_converted_SHA256'],name
                set_module_tensor_to_device(model,name,owner(name),value=value,dtype=dtype)
                del original,value
            for name,buffer in list(model.named_buffers()):
                assert not buffer.is_meta,name
                set_module_tensor_to_device(model,name,owner(name),value=buffer,dtype=buffer.dtype)
            assert all(p.dtype==dtype and p.device.type=='cuda'for p in model.parameters())
            assert all(b.dtype==torch.float32 and b.device.type=='cuda'for b in model.buffers())
            model=dispatch_model(model,device_map=device_map,main_device=0,force_hooks=True)
            model.eval();before={name:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype),str(p.device))for name,p in model.named_parameters()}
            case={'original_language_layers':original_layers,'retained_selected_layers':50,'parameter_dtypes':dict(collections.Counter(str(p.dtype)for p in model.parameters())),
                  'buffer_dtypes':dict(collections.Counter(str(b.dtype)for b in model.buffers())),
                  'parameters':len(before),'original_final_norm_executes':True,'raw50':[],'vision27':[],
                  'comparison_SDPA_backend':'SDPBackend.MATH on CUDA; FP32 intermediates allowed, no FP32 Linear weights'}
            captured={};observed_inputs=set();observed_outputs=set()
            def pre_norm(module,args):captured['pre_final_norm']=args[0].detach().clone()
            def vision_hook(module,args,out):
                captured['merged']=out.pooler_output.detach().cpu().contiguous()
                captured['deepstack']=[v.detach().cpu().contiguous()for v in out.deepstack_features]
            def layer_hook(kind,index):
                def post(module,args,out):
                    row=info(out);assert row['finite']and row['dtype']==str(dtype)
                    case[kind].append({'layer':index,'output':row})
                return post
            def linear_pre(module,args):observed_inputs.add(str(args[0].dtype))
            def linear_post(module,args,out):
                observed_outputs.add(str(out.dtype))
                if not bool(torch.isfinite(out).all()):raise AssertionError('Original GPU encoder Linear produced nonfinite output')
            handles.append(model.language_model.norm.register_forward_pre_hook(pre_norm))
            handles.append(model.visual.register_forward_hook(vision_hook))
            for i,block in enumerate(model.language_model.layers):handles.append(block.register_forward_hook(layer_hook('raw50',i)))
            for i,block in enumerate(model.visual.blocks):handles.append(block.register_forward_hook(layer_hook('vision27',i)))
            for module in model.modules():
                if isinstance(module,nn.Linear):handles.extend((module.register_forward_pre_hook(linear_pre),module.register_forward_hook(linear_post)))
            ids=fixture['input_ids'][None].to('cuda:0');mask=torch.ones_like(ids)
            mm=torch.zeros_like(ids,dtype=torch.int32);mm[ids==config.image_token_id]=1;mm[ids==config.video_token_id]=2
            kwargs={'input_ids':ids,'attention_mask':mask,'pixel_values':fixture['pixel_values'].to('cuda:0'),
                    'image_grid_thw':fixture['image_grid_thw'].to('cuda:0'),'mm_token_type_ids':mm,'use_cache':False}
            synchronize();start=time.perf_counter()
            with torch.inference_mode(),sdpa_kernel(backends=[SDPBackend.MATH]):output=model(**kwargs)
            synchronize();case['GPU_reference_wall_s_not_NPU_latency']=time.perf_counter()-start
            pre=captured['pre_final_norm'].cpu().contiguous();post=output.last_hidden_state.detach().cpu().contiguous()
            assert pre.shape==(1,2043,5120)and pre.dtype==dtype
            assert len(case['raw50'])==50 and len(case['vision27'])==27
            assert info(pre)['sha256']==case['raw50'][-1]['output']['sha256']
            assert observed_inputs==observed_outputs=={str(dtype)}
            case['Linear_input_dtypes']=list(observed_inputs);case['Linear_output_dtypes']=list(observed_outputs)
            case['pre_final_norm']=info(pre);case['HF_post_final_norm']=info(post)
            case['vision_merged']=info(captured['merged']);case['DeepStack']=[info(x)for x in captured['deepstack']]
            case['NPU_FP16_comparisons']={name:metric(value,pre)for name,value in npu_outputs.items()}
            for h in handles:h.remove()
            handles.clear();captured.clear();handles.append(model.language_model.norm.register_forward_pre_hook(pre_norm))
            synchronize();start=time.perf_counter()
            with torch.inference_mode(),sdpa_kernel(backends=[SDPBackend.MATH]):repeat=model(**kwargs)
            synchronize();case['GPU_full_repeat_wall_s']=time.perf_counter()-start
            case['repeat_pre_bits_equal']=torch.equal(pre.view(torch.uint16),captured['pre_final_norm'].cpu().contiguous().view(torch.uint16))
            case['repeat_post_bits_equal']=torch.equal(post.view(torch.uint16),repeat.last_hidden_state.cpu().contiguous().view(torch.uint16))
            assert case['repeat_pre_bits_equal']and case['repeat_post_bits_equal']
            after={name:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype),str(p.device))for name,p in model.named_parameters()}
            assert before==after;case['all_parameters_storage_versions_shapes_dtypes_unchanged']=True
            dest=root/('GPU_%s_selected50_114.pt'%label);assert not dest.exists()
            with dest.open('xb')as f:torch.save({'pre_final_norm':pre,'HF_post_final_norm':post,'vision_merged':case['vision_merged'],'DeepStack':case['DeepStack']},f)
            case['saved_output']={'path':str(dest),'file_SHA256':file_sha(dest)}
            result['cases'][label]=case;writer.write(result)
            for h in handles:h.remove()
            handles.clear();del model,output,repeat,pre,post,captured,kwargs;model=None;gc.collect()
            for device in range(4):
                with torch.cuda.device(device):torch.cuda.empty_cache()
        result['completed']=True;writer.write(result)
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error)};writer.write(result);raise
    finally:
        for h in handles:h.remove()


if __name__=='__main__':main()
