#!/usr/bin/env python3
"""Actual Half vision Conv3d/nonlinear leaves and CPU-entry reset,not full text timing."""
from pathlib import Path
import json
import time
import statistics
ROOT=Path('/tmp/h3_round89_vision02')
LEAVES=('blocks.0.norm1','blocks.0.norm2','blocks.0.mlp.act_fn',
        'deepstack_merger_list.0.norm','deepstack_merger_list.0.act_fn','merger.norm','merger.act_fn')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io89 import EvidenceWriter
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from runtime5gm83 import Runtime5GM
    from vision_adapter87 import VisionBiasAdapter87
    from patch_bias89 import PatchBiasAdapter89
    from original_hf_entry89 import recompute
    from vision_leaf89 import run_controls,VisionNonlinearAdapter89
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    writer=EvidenceWriter(ROOT/'vision89.json')
    r={'scope':__doc__,'completed':False,'source_sha256':previous.file_sha(Path(__file__)),
       'cases':{},'full_encoder_precision_accepted':False,'whole_goal_complete':False,
       'numerical_tolerance':None,'R88_slow_attention_not_installed':True,'CPU_rebase_not_full_encoder_timing':True}
    bias=None;patch=None;nonlinear=None;handles=[];rebase_handle=None
    try:
        proof=previous.load_cpu_gate(r);candidate=previous.load_candidate(r)
        assert previous.file_sha(previous.FIXTURE)=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),local_files_only=True).vision_config
        assert(cfg.hidden_size,cfg.intermediate_size,cfg.num_heads)==(1152,4304,16)
        payload=torch.load(previous.FIXTURE,map_location='cpu',weights_only=True)
        with torch.device('meta'):vision=candidate.MiniMaxH3Qwen3VLVisionModel(cfg)
        wanted=set(vision.state_dict());index=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map'];state={}
        for shard in sorted(set(index.values())):
            keys=[name for name,s in index.items()if s==shard and name.startswith('model.visual.')and name[len('model.visual.'):]in wanted]
            if not keys:continue
            with safe_open(str(previous.MODEL/shard),framework='pt',device='cpu')as h:
                for name in keys:
                    value=h.get_tensor(name).to(torch.float16);key=name[len('model.visual.'):]
                    assert previous.tensor_info(value)['sha256']==proof['parameter_conversion']['visual.'+key]['reference_sha256']
                    state[key]=value
        assert set(state)==wanted
        cpu,receipt=recompute(cfg,state,payload,proof,previous.compare,previous.tensor_info)
        r['original_HF_CPU_entry_reference']=receipt;writer.write(r)
        vision.load_state_dict(state,strict=True,assign=True);del state
        vision.to(device='npu:0');vision.eval()
        vision.rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(cfg.hidden_size//cfg.num_heads//2)
        assert all(p.dtype==torch.float16 for p in vision.parameters())
        before=previous.parameters_sha(vision)
        pixels=payload['pixel_values'].to('npu:0');grids=payload['image_grid_thw'].to('npu:0')
        pixels_sha=previous.tensor_info(pixels)['sha256']
        oracle=torch.load(previous.CPU_ROOT/'FP16_vision.pt',map_location='cpu',weights_only=True)
        bias=VisionBiasAdapter87(candidate,Runtime5GM(0),ROOT/'bias_kernels');bias.prime(vision);bias.set_enabled(False)
        inputs={};captured={};modules=dict(vision.named_modules())
        def input_hook(name):
            def hook(module,args):inputs[name]=args[0].detach().clone()
            return hook
        for name in LEAVES:handles.append(modules[name].register_forward_pre_hook(input_hook(name)))
        def patch_post(module,args,out):captured['patch']=out.detach().clone()
        def entry_pre(module,args):captured['entry']=args[0].detach().clone()
        def block_post(module,args,out):captured['block0']=out.detach().clone()
        handles.extend((vision.patch_embed.register_forward_hook(patch_post),
                        vision.blocks[0].register_forward_pre_hook(entry_pre),
                        vision.blocks[0].register_forward_hook(block_post)))
        with torch.inference_mode():native_merged,native_ds=vision(pixels,grids)
        for h in handles:h.remove()
        handles.clear()
        r['actual_entry']={'patch_vs_original_CPU_Half_Conv3d':previous.compare(captured['patch'],cpu['patch']),
                           'initial_vs_original_CPU_Half':previous.compare(captured['entry'],cpu['entry']),
                           'block0_vs_original_CPU_Half':previous.compare(captured['block0'],cpu['block0'])}
        with torch.inference_mode():
            native_pos=vision.fast_pos_embed_interpolate(grids)
            r['actual_entry']['positions_vs_original_CPU_Half']=previous.compare(native_pos,cpu['positions'])
            assert previous.compare(captured['patch']+native_pos,captured['entry'])['raw_bits_equal']
            patch_weight=vision.patch_embed.proj.weight.detach().cpu();patch_bias=vision.patch_embed.proj.bias.detach().cpu()
            flat_weight=patch_weight.reshape(cfg.hidden_size,-1)
            assert flat_weight.shape==(1152,1536)and flat_weight.dtype==patch_bias.dtype==torch.float16
            flat_cpu=torch.nn.functional.linear(payload['pixel_values'],flat_weight,patch_bias)
            flat_repeat=torch.nn.functional.linear(payload['pixel_values'],flat_weight,patch_bias)
            assert previous.compare(flat_repeat,flat_cpu)['raw_bits_equal']
            r['actual_entry']['CPU_Half_Linear_vs_original_CPU_Half_Conv3d']=previous.compare(flat_cpu,cpu['patch'])
        assert set(inputs)==set(LEAVES)
        with torch.inference_mode():
            r['actual_nonlinear_leaf']=run_controls(vision,inputs,previous.compare,previous.tensor_info)
        assert all(r['actual_nonlinear_leaf'][name]for name in('all_native_repeats_bit_identical','all_cpu_half_repeats_bit_identical','all_npu_float_diagnostics_supported','all_inputs_unchanged','all_leaf_parameters_unchanged'))
        writer.write(r)
        patch=PatchBiasAdapter89(vision,Runtime5GM(0),ROOT/'bias_kernels')
        patch_input=pixels.reshape(8064,3,2,16,16)
        with torch.inference_mode():
            patch_control=patch.packing_control(patch_input)
            patch.set_enabled(True)
            patch_out=vision.patch_embed.proj(patch_input).reshape(8064,1152)
            patch_repeat=vision.patch_embed.proj(patch_input).reshape(8064,1152)
            assert previous.compare(patch_out,patch_repeat)['raw_bits_equal']
            r['actual_patch_onecast_GEMM']={'same_original_Half_weight_bias':True,'controls':patch_control,
                                          'custom_vs_original_CPU_Half_Conv3d':previous.compare(patch_out,cpu['patch']),
                                          'custom_vs_CPU_Half_flattenLinear':previous.compare(patch_out,flat_cpu)}
            patch.set_enabled(False)
        nonlinear=VisionNonlinearAdapter89(vision);nonlinear.set_modes(norm=False,gelu=False)
        rebase={'enabled':False,'calls':0};entry_npu=cpu['entry'].to('npu:0')
        assert previous.compare(entry_npu,cpu['entry'])['raw_bits_equal']
        entry_npu_sha=previous.tensor_info(entry_npu)['sha256']
        def rebase_pre(module,args,kwargs):
            if not rebase['enabled']:return None
            assert previous.compare(args[0],captured['entry'])['raw_bits_equal']
            rebase['calls']+=1
            return(entry_npu,)+tuple(args[1:]),kwargs
        rebase_handle=vision.blocks[0].register_forward_pre_hook(rebase_pre,with_kwargs=True)
        modes=(('native',False,False,False,False),('CPU_entry_only',True,False,False,False),
               ('CPU_entry_bias',True,True,False,False),('Float_norm_only',False,False,True,False),
               ('Float_GELU_only',False,False,False,True),('Float_norm_and_GELU',False,False,True,True),
               ('Float_norm_GELU_bias',False,True,True,True),('CPU_entry_norm_GELU_bias',True,True,True,True))
        modes=tuple(tuple(item)+(False,)for item in modes)+(
            ('patch_bias_only',False,False,False,False,True),
            ('patch_and_linear_bias',False,True,False,False,True),
            ('patch_norm_GELU_linearbias',False,True,True,True,True))
        for name,reset,use_bias,use_norm,use_gelu,use_patch in modes:
            patch.set_enabled(use_patch)
            bias.set_enabled(use_bias);nonlinear.set_modes(norm=use_norm,gelu=use_gelu);rebase['enabled']=reset
            layers=[]
            def layer_post(i):
                def hook(module,args,out):
                    info=previous.tensor_info(out);assert info['finite'];layers.append({'layer':i,'output':info})
                return hook
            for i,layer in enumerate(vision.blocks):handles.append(layer.register_forward_hook(layer_post(i)))
            try:
                with torch.inference_mode():merged,ds=vision(pixels,grids)
            finally:
                for h in handles:h.remove()
                handles.clear()
            assert len(layers)==27
            case={'patch_onecast_GEMM':use_patch,'CPU_entry_reset':reset,'norm_Float_diagnostic':use_norm,'GELU_Float_diagnostic':use_gelu,
                  'onecast_bias':use_bias,'merged_vs_CPU':previous.compare(merged,oracle['merged']),
                  'DeepStack_vs_CPU':[previous.compare(a,b)for a,b in zip(ds,oracle['deepstack'])],
                  'raw27_layers':layers,'warm_s':[],'repeat_bits':[]}
            with torch.inference_mode():
                for _ in range(3):
                    torch.npu.synchronize();start=time.perf_counter();out,out_ds=vision(pixels,grids);torch.npu.synchronize()
                    case['warm_s'].append(time.perf_counter()-start)
                    repeated={'merged':previous.compare(out,merged),'deepstack':[previous.compare(a,b)for a,b in zip(out_ds,ds)]}
                    assert repeated['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in repeated['deepstack'])
                    case['repeat_bits'].append(repeated)
            case['median_resident_vision_s']=statistics.median(case['warm_s'])
            case['time_not_full_encoder_and_CPU_reset_extra_control_not_speed_candidate']=reset
            case['nonlinear']=nonlinear.snapshot();case['bias']=bias.snapshot();r['cases'][name]=case;writer.write(r)
        patch.set_enabled(False);bias.set_enabled(False);nonlinear.set_modes(norm=False,gelu=False);rebase['enabled']=False
        with torch.inference_mode():out,ds=vision(pixels,grids)
        r['native_recovery']={'merged':previous.compare(out,native_merged),'deepstack':[previous.compare(a,b)for a,b in zip(ds,native_ds)]}
        assert r['native_recovery']['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in r['native_recovery']['deepstack'])
        r['original_parameters_unchanged']=before==previous.parameters_sha(vision);assert r['original_parameters_unchanged']
        r['pixels_unchanged']=pixels_sha==previous.tensor_info(pixels)['sha256'];assert r['pixels_unchanged']
        r['CPU_entry_NPU_transfer_and_all_calls_unchanged']=entry_npu_sha==previous.tensor_info(entry_npu)['sha256'];assert r['CPU_entry_NPU_transfer_and_all_calls_unchanged']
        r['rebase_calls']=rebase['calls']
        r['CPU_reference_SHA_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_SHA_unchanged']
        target=ROOT/'CPUEntryAndActualLeaves89.pt';assert not target.exists()
        torch.save({'CPU_entry_reference':cpu,'native_entry':{k:x.cpu()for k,x in captured.items()},
                    'actual_leaf_inputs':{k:x.cpu()for k,x in inputs.items()}},target)
        r['frozen_inputs_SHA256']=previous.file_sha(target);r['completed']=True
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        for h in handles:h.remove()
        if rebase_handle is not None:rebase_handle.remove()
        if patch is not None:r['patch_final']=patch.snapshot();patch.uninstall()
        if nonlinear is not None:r['nonlinear_final']=nonlinear.snapshot();nonlinear.uninstall()
        if bias is not None:r['bias_final']=bias.snapshot();bias.uninstall()
        writer.write(r)


if __name__=='__main__':main()
