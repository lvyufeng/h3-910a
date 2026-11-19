#!/usr/bin/env python3
"""Official-HF 4090 FP16 and original-BF16 full multimodal encoder references.

Weights are downloaded ON this GPU host from the pinned official revision and read in
their original tensor dtype; BF16 never comes from an FP16 round trip. Whole Linear
weights are placed across GPUs by layer, not tensor-parallel sharded. All fifty retained
decoder layers plus the twenty-seven vision blocks and three DeepStack mergers execute on
the real captured images. The original HF final norm runs unchanged.
"""
from pathlib import Path
import argparse, collections, gc, hashlib, json, time


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def info(value):
    import torch
    x=value.detach().cpu().contiguous();finite=torch.isfinite(x)
    out={'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size(),
         'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),'finite':bool(finite.all())}
    if x.is_floating_point():out['max_abs']=float(x[finite].abs().max())if bool(finite.any())else None
    return out


def metric(candidate,reference):
    import torch
    a=candidate.detach().float().double();b=reference.detach().float().double()
    assert a.shape==b.shape
    both=torch.isfinite(a)&torch.isfinite(b)
    assert bool(both.all()),'nonfinite values in compared reference'
    d=a[both]-b[both]
    return {'candidate':info(candidate),'reference':info(reference),
            'relative_rms':float(d.square().mean().sqrt()/b[both].square().mean().sqrt()),
            'max_abs_error':float(d.abs().max()),
            'reference_not_cast_to_FP16':str(reference.dtype)!='torch.float16',
            'numerical_tolerance':None,'declared_pass':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/mseco/h3_gpu_reference114'))
    parser.add_argument('--dtypes',default='BF16,FP16')
    args=parser.parse_args();root=args.root.resolve(strict=True)
    checkpoint=root/'checkpoint'/'FL2VA/text_encoder'
    assert (checkpoint/'config.json').is_file(),'Pinned official download is incomplete; refusing to run.'
    import torch
    from torch import nn
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import transformers
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    from accelerate import dispatch_model
    from safetensors import safe_open
    destination=root/'GPU_encoder_reference114.json'
    result={'scope':__doc__,'completed':False,'whole_goal_complete':False,'full_encoder_precision_accepted':False,
            'numerical_tolerance':None,'versions':{'torch':torch.__version__,'transformers':transformers.__version__},
            'weights_downloaded_on_GPU_host':True,'no_local_weight_relay':True,'cases':{}}
    def publish():
        tmp=destination.with_name(destination.name+'.tmp')
        tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(destination)
    publish()
    try:
        assert torch.cuda.device_count()>=8 and all('RTX 4090'in torch.cuda.get_device_name(i)for i in range(8))
        assert torch.cuda.is_bf16_supported()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        assert file_sha(hf.__file__)=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
        download=json.loads((root/'direct_download_verified114.json').read_text());assert download['completed']
        fixture_path=root/'fixtures'/'encoder_input.pt'
        assert file_sha(fixture_path)=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
        fixture=torch.load(fixture_path,map_location='cpu',weights_only=True)
        assert fixture['pixel_values'].dtype==torch.float16
        probe_config=Qwen3VLConfig.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
        result['original_language_layer_count']=probe_config.text_config.num_hidden_layers
        result['actual_vision_config']={key:getattr(probe_config.vision_config,key)
                                        for key in('depth','hidden_size','num_heads','intermediate_size',
                                                   'num_position_embeddings','spatial_merge_size','patch_size','temporal_patch_size')}
        assert result['actual_vision_config']['depth']==27 and result['actual_vision_config']['hidden_size']==1152
        index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
        available={name for name in set(index.values())if (checkpoint/name).is_file()}
        missing_shards=sorted(set(index.values())-available)
        result['checkpoint_files']={name:file_sha(checkpoint/name)for name in sorted(available)}
        result['shards_present']=sorted(available);result['shards_absent']=missing_shards
        assert missing_shards==['model-00012-of-00014.safetensors','model-00013-of-00014.safetensors'],missing_shards
        ids=fixture['input_ids'][None];mask=torch.ones_like(ids)
        for label in args.dtypes.split(','):
            dtype={'BF16':torch.bfloat16,'FP16':torch.float16}[label]
            cfg=Qwen3VLConfig.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
            cfg.text_config.num_hidden_layers=50;cfg.text_config.use_cache=False
            cfg.dtype=dtype;cfg.text_config.dtype=dtype;cfg.vision_config.dtype=dtype
            for attribute in('_attn_implementation',):
                setattr(cfg,attribute,'sdpa');setattr(cfg.vision_config,attribute,'sdpa');setattr(cfg.text_config,attribute,'sdpa')
            with torch.device('meta'):model=hf.Qwen3VLModel(cfg)
            model.visual.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding(cfg.vision_config.hidden_size//cfg.vision_config.num_heads//2)
            model.language_model.rotary_emb=hf.Qwen3VLTextRotaryEmbedding(cfg.text_config,device='cpu')
            wanted=dict(model.named_parameters());state={}
            for shard in sorted(available):
                keys=[k for k,f in index.items()if f==shard and k.startswith('model.')and k[6:]in wanted]
                if not keys:continue
                with safe_open(str(checkpoint/shard),framework='pt',device='cpu')as f:
                    for key in keys:
                        original=f.get_tensor(key);assert original.dtype==torch.bfloat16,key
                        value=original if dtype==torch.bfloat16 else original.to(dtype=dtype)
                        if not bool(torch.isfinite(value.float()).all()):raise ValueError('Nonfinite converted parameter: '+key)
                        state[key[6:]]=value
            assert set(state)==set(wanted),sorted(set(wanted)-set(state))[:4]
            model.load_state_dict(state,strict=True,assign=True);del state
            device_map={'visual':0,'language_model.embed_tokens':0,'language_model.rotary_emb':0,'language_model.norm':7}
            for i in range(50):device_map['language_model.layers.%d'%i]=1+i*7//50
            model=dispatch_model(model,device_map=device_map,main_device=0,force_hooks=True)
            model.eval();assert all(p.dtype==dtype for p in model.parameters())
            before={n:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype))for n,p in model.named_parameters()}
            case={'requested_dtype':label,'parameters':len(before),'retained_language_layers':50,
                  'original_vision_blocks':len(model.visual.blocks),
                  'weights_read_in_original_BF16_then_cast_to_requested_dtype':True,
                  'comparison_attention':'SDPBackend.MATH FP32 intermediates; no FP32 Linear weights',
                  'device_map':device_map,'raw50':[],'vision27':[]}
            captured={};observed_in=set();observed_out=set();handles=[]
            def pre_norm(module,arguments):captured['pre']=arguments[0].detach().clone()
            def vision_post(module,arguments,output):
                captured['merged']=output.pooler_output.detach().cpu().contiguous()
                captured['deepstack']=[v.detach().cpu().contiguous()for v in output.deepstack_features]
            def hook(kind,index):
                def post(module,arguments,output):
                    row=info(output);assert row['finite']and row['dtype']==str(dtype)
                    case[kind].append({'layer':index,'output':row})
                return post
            def linear_pre(module,arguments):observed_in.add(str(arguments[0].dtype))
            def linear_post(module,arguments,output):
                observed_out.add(str(output.dtype))
                if not bool(torch.isfinite(output).all()):raise AssertionError('GPU original-HF encoder Linear nonfinite: '+label)
            handles.append(model.language_model.norm.register_forward_pre_hook(pre_norm))
            handles.append(model.visual.register_forward_hook(vision_post))
            for i,block in enumerate(model.language_model.layers):handles.append(block.register_forward_hook(hook('raw50',i)))
            for i,block in enumerate(model.visual.blocks):handles.append(block.register_forward_hook(hook('vision27',i)))
            for module in model.modules():
                if isinstance(module,nn.Linear):
                    handles.extend((module.register_forward_pre_hook(linear_pre),module.register_forward_hook(linear_post)))
            mm=torch.zeros_like(ids,dtype=torch.int32);mm[ids==cfg.image_token_id]=1;mm[ids==cfg.video_token_id]=2
            pixels=fixture['pixel_values'].to(dtype=dtype)
            kwargs={'input_ids':ids,'attention_mask':mask,'pixel_values':pixels,
                    'image_grid_thw':fixture['image_grid_thw'],'mm_token_type_ids':mm,'use_cache':False}
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(),sdpa_kernel(backends=[SDPBackend.MATH]):output=model(**kwargs)
            torch.cuda.synchronize();case['GPU_reference_wall_s_not_NPU_latency']=time.perf_counter()-start
            pre=captured['pre'].cpu().contiguous();post=output.last_hidden_state.detach().cpu().contiguous()
            assert pre.shape==(1,2043,5120)and pre.dtype==dtype
            assert len(case['raw50'])==50 and len(case['vision27'])==27
            assert info(pre)['sha256']==case['raw50'][-1]['output']['sha256'],'pre-final-norm boundary mismatch'
            assert observed_in==observed_out=={str(dtype)},(observed_in,observed_out)
            case['Linear_input_dtypes']=sorted(observed_in);case['Linear_output_dtypes']=sorted(observed_out)
            case['pre_final_norm']=info(pre);case['HF_post_final_norm']=info(post)
            case['vision_merged']=info(captured['merged']);case['DeepStack']=[info(x)for x in captured['deepstack']]
            for h in handles:h.remove()
            handles.clear();handles.append(model.language_model.norm.register_forward_pre_hook(pre_norm))
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(),sdpa_kernel(backends=[SDPBackend.MATH]):repeat=model(**kwargs)
            torch.cuda.synchronize();case['GPU_full_repeat_wall_s']=time.perf_counter()-start
            case['repeat_pre_bits_equal']=torch.equal(pre.view(torch.uint8),captured['pre'].cpu().contiguous().view(torch.uint8))
            case['repeat_post_bits_equal']=torch.equal(post.view(torch.uint8),repeat.last_hidden_state.cpu().contiguous().view(torch.uint8))
            assert case['repeat_pre_bits_equal']and case['repeat_post_bits_equal']
            after={n:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype))for n,p in model.named_parameters()}
            assert before==after;case['all_parameters_storage_versions_shapes_dtypes_unchanged']=True
            saved=root/('GPU_%s_selected50_114.pt'%label);assert not saved.exists()
            with saved.open('xb')as f:torch.save({'pre_final_norm':pre,'HF_post_final_norm':post,
                                                  'vision_merged':captured['merged'],'DeepStack':captured['deepstack']},f)
            case['saved_output']={'path':str(saved),'file_SHA256':file_sha(saved)}
            result['cases'][label]=case
            if label=='FP16'and'BF16'in result['cases']:
                bf16=torch.load(str(root/'GPU_BF16_selected50_114.pt'),map_location='cpu',weights_only=True)['pre_final_norm']
                case['FP16_vs_original_BF16_pre_final_norm']=metric(pre,bf16)
            publish()
            for h in handles:h.remove()
            handles.clear();del model,output,repeat,pre,post,captured,kwargs,pixels;gc.collect()
            for device in range(8):
                with torch.cuda.device(device):torch.cuda.empty_cache()
        result['completed']=True;publish()
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error)};publish();raise


if __name__=='__main__':main()