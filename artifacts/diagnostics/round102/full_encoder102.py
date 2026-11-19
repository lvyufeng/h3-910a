#!/usr/bin/env python3
"""Actual TP8 FP16 full encoder: single-launch BC128 buffered-reuse Float vision, native/precise text A/B."""
from pathlib import Path
import json
import os
import time
import statistics
ROOT=Path('/tmp/h3_round102_full01')


def main():
    import torch
    import torch_npu
    import best_text87 as previous
    from evidence_io102 import EvidenceWriter
    from safetensors import safe_open
    from runtime5gm83 import Runtime5GM
    from attention_engine85 import AccurateAttention85
    from vision_adapter87 import VisionBiasAdapter87
    from patch_bias89 import PatchBiasAdapter89
    from erf_gelu_fixed90 import ErfGeluAdapter90
    from geometry_cache91 import VisionGeometryCache91
    from flash_engine101 import FlashVisionAttention101
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
    writer=EvidenceWriter(ROOT/('tp8_rank%d.json'%rank))
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,
       'source_SHA256':previous.file_sha(Path(__file__)),'full_encoder_precision_accepted':False,
       'whole_goal_complete':False,'numerical_tolerance':None,'cases':{},
       'FP32_row_adapter_not_installed':True,'all_linear_model_weights_Half':True,
       'real_full_encoder_warm_not_CPU_condition_time':True,
       'R89_redundant_Float_nonlinears_not_installed':True,
       'R101_BC128_Float_vision_attention_opt_in':True,'CPU_text_RoPE_cache_not_installed':True}
    kv=None;attention=None;vision_attention=None;bias=None;patch=None;erf=None;geometry=None;group=None;dist_ready=False;handles=[]
    try:
        torch.set_num_threads(2);torch.npu.set_device(rank);previous.ROOT=ROOT
        assert previous.file_sha(previous.__file__)=='be449afe470a593d5381bcf8c08a1c0c3654010e1790351b59d8dae88d909409'
        assert os.environ.get('VLLM_OMNI_910A_TE_DTYPE')=='half'
        assert all(os.environ.get(name,'0')=='0'for name in('VLLM_OMNI_910A_TEXT_FP32_LINEAR','VLLM_OMNI_910A_TEXT_PAD_KV','VLLM_OMNI_910A_VISION_FP32'))
        candidate=previous.load_candidate(r);previous.install_fp16_f32_math_attention(candidate,r)
        from kv_padding81 import install
        kv=install(candidate)
        proof=previous.load_cpu_gate(r)
        assert previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert previous.file_sha(previous.FIXTURE)==previous.FIXTURE_SHA256
        assert previous.file_sha(previous.MODEL/'model.safetensors.index.json')==previous.CHECKPOINT_INDEX_SHA256
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        payload=torch.load(previous.FIXTURE,map_location='cpu',weights_only=True)
        assert payload['input_ids'].shape==(2043,)and payload['pixel_values'].shape==(8064,1536)
        assert payload['pixel_values'].dtype==torch.float16 and payload['image_grid_thw'].tolist()==[[1,48,84],[1,48,84]]
        from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
        from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel,init_distributed_environment,initialize_model_parallel
        init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl');dist_ready=True
        initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
        group=GroupCoordinator([list(range(size))],rank,'hccl')
        encoder=candidate.MiniMaxH3Qwen3VLEncoder(str(previous.MODEL),device=torch.device('npu',rank),load_model=True,encoder_group=group)
        encoder.eval();index=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map']
        def tensors():
            for shard in sorted(set(index.values())):
                keys=[name for name,s in index.items()if s==shard and encoder._map_weight_name(name)is not None]
                if not keys:continue
                with safe_open(str(previous.MODEL/shard),framework='pt',device='cpu')as h:
                    for name in keys:yield name,h.get_tensor(name)
        loaded=encoder.load_weights(tensors());assert loaded==set(dict(encoder.named_parameters()))
        assert len(encoder.text_model.layers)==50 and not any(name=='norm'for name,_ in encoder.text_model.named_children())
        assert candidate.MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER==50
        assert all(p.dtype==torch.float16 for p in encoder.parameters())
        for name,p in encoder.vision.named_parameters():
            assert previous.tensor_info(p)['sha256']==proof['parameter_conversion']['visual.'+name]['reference_sha256']
        encoder.to(torch.device('npu',rank));kv.prime(encoder.text_model)
        attention=AccurateAttention85(candidate,Runtime5GM(rank),ROOT/'attention_kernels');attention.prepare();attention.set_enabled(False)
        bias=VisionBiasAdapter87(candidate,Runtime5GM(rank),ROOT/'bias_kernels');bias.prime(encoder.vision);bias.set_enabled(False)
        patch=PatchBiasAdapter89(encoder.vision,Runtime5GM(rank),ROOT/'bias_kernels');patch.set_enabled(False)
        erf=ErfGeluAdapter90(encoder.vision,Runtime5GM(rank),ROOT/'erf_kernels');erf.prepare();erf.set_enabled(False)
        vision_attention=FlashVisionAttention101(candidate,Runtime5GM(rank),ROOT/'vision_attention_kernels')
        r['vision_attention_prepare']=vision_attention.prepare();vision_attention.set_enabled(False)
        expected_vision=json.loads((ROOT/'expected_vision101.json').read_text())
        geometry=VisionGeometryCache91(encoder.vision)
        prime_grid=payload['image_grid_thw'].to(torch.device('npu',rank),torch.long)
        r['geometry_prime']=geometry.prime(prime_grid);geometry.set_enabled(False)
        r['setup_excluded_from_warm']={'all_vision_weights_original_Half':True,'kernel_prebuild_and_layout_cache':True,
                                       'geometry_cache_only_positions_and_rotary_not_pixels_or_conditioning':True,
                                       'KV_padding':kv.snapshot(),'erf':erf.snapshot(),'vision_bias':bias.snapshot(),'patch':patch.snapshot()}
        param_before=previous.parameters_sha(encoder);input_before=previous.inputs_sha(payload)
        cpu_vision=torch.load(previous.CPU_ROOT/'FP16_vision.pt',map_location='cpu',weights_only=True)
        cpu_inputs=torch.load(previous.CPU_ROOT/'FP16_multimodal_inputs.pt',map_location='cpu',weights_only=True)
        gold=torch.load(previous.CPU_ROOT/'FP16_selected50.pt',map_location='cpu',weights_only=True)['pre_final_norm']
        gold_layers={i:torch.load(previous.CPU_ROOT/('FP16_layer%02d.pt'%i),map_location='cpu',weights_only=True)for i in previous.EXPECTED_CPU_LAYERS}
        cpu_layer_sha={row['layer']:row['output']['sha256']for row in proof['language_layers']}
        captured={}
        def equivalent(collect=False):
            device=torch.device('npu',rank);ids=payload['input_ids'].to(device,torch.long)[None]
            mask=torch.ones_like(ids);types=torch.zeros_like(ids,dtype=torch.int32)
            types.masked_fill_(ids==encoder.image_token_id,1);types.masked_fill_(ids==encoder.video_token_id,2)
            grid=payload['image_grid_thw'].to(device,torch.long)
            embeds=encoder.text_model.embed_tokens(ids)
            images,deepstack=encoder.vision(payload['pixel_values'].to(device,torch.float16),grid)
            images=images.to(embeds.device,embeds.dtype)
            imask,_=encoder._get_placeholder_mask(ids,inputs_embeds=embeds,image_features=images)
            embeds=embeds.masked_scatter(imask,images)
            vmask,ds=encoder._aggregate_deepstack(imask,None,deepstack,None)
            pos,_=candidate._get_rope_index(ids,mm_token_type_ids=types,image_grid_thw=grid,video_grid_thw=None,
                                          attention_mask=mask,spatial_merge_size=int(encoder.vision.config.spatial_merge_size))
            if collect:
                captured['vision']={'merged':images.detach().cpu(),'deepstack':[x.detach().cpu()for x in deepstack]}
                captured['text_inputs']={'inputs_embeds':embeds.detach().clone(),'position_ids':pos.detach().clone(),
                                         'visual_pos_masks':vmask.detach().clone(),'deepstack_visual_embeds':[x.detach().clone()for x in ds]}
            return encoder.text_model(embeds,pos,visual_pos_masks=vmask,deepstack_visual_embeds=ds).cpu()
        def stock():
            return encoder.encode_ids(payload['input_ids'],pixel_values=payload['pixel_values'],image_grid_thw=payload['image_grid_thw'])
        visual_mask=cpu_inputs['visual_pos_masks'].reshape(-1)
        flatgold=gold.reshape(-1,5120);highrange=flatgold.abs().amax(-1)>=512
        def subsets(value):
            flat=value.reshape(-1,5120);result={}
            for name,select in(('text_nonvisual',~visual_mask),('visual',visual_mask),('reference_abs_ge512',highrange),('reference_abs_lt512',~highrange)):
                if bool(select.any()):result[name]=previous.compare(flat[select],flatgold[select])
            return result
        outputs={};layer_hashes={}
        modes=(('native',False,False,False),
               ('corrected_vision_native_text',True,False,False),
               ('corrected_bc128_vision_native_text',True,False,True),
               ('corrected_bc128_vision_precise_text',True,True,True))
        for name,fix,precise,accurate_vision in modes:
            bias.set_enabled(fix);patch.set_enabled(fix);erf.set_enabled(fix)
            attention.set_enabled(precise);geometry.set_enabled(True);vision_attention.set_enabled(accurate_vision)
            captured.clear();layers={};vislayers={}
            def layer_post(i,kind):
                def hook(module,args,out):
                    (layers if kind=='text'else vislayers)[i]=out.detach().cpu().contiguous()
                return hook
            for i,m in enumerate(encoder.text_model.layers):handles.append(m.register_forward_hook(layer_post(i,'text')))
            for i,m in enumerate(encoder.vision.blocks):handles.append(m.register_forward_hook(layer_post(i,'vision')))
            try:
                with torch.inference_mode():first=equivalent(collect=True)
            finally:
                for h in handles:h.remove()
                handles.clear()
            assert len(layers)==50 and len(vislayers)==27
            case={'vision_fixes':fix,'precise_text_attention':precise,'geometry_cache':True,'accurate_streaming_vision_attention':accurate_vision,
                  'selected50_vs_CPU':previous.compare(first,gold),'token_subsets':subsets(first),
                  'vision_merged_vs_CPU':previous.compare(captured['vision']['merged'],cpu_vision['merged']),
                  'vision_DeepStack_vs_CPU':[previous.compare(a,b)for a,b in zip(captured['vision']['deepstack'],cpu_vision['deepstack'])],
                  'text_inputs_vs_CPU':previous.compare(captured['text_inputs']['inputs_embeds'],cpu_inputs['inputs_embeds']),
                  'position_ids_vs_CPU':previous.compare(captured['text_inputs']['position_ids'],cpu_inputs['position_ids']),
                  'visual_mask_vs_CPU':previous.compare(captured['text_inputs']['visual_pos_masks'],cpu_inputs['visual_pos_masks']),
                  'raw50_layers':[],'raw27_vision_layers':[],'warm_stock_s':[],'warm_output_SHA256':[]}
            assert case['selected50_vs_CPU']['finite']and case['position_ids_vs_CPU']['raw_bits_equal']and case['visual_mask_vs_CPU']['raw_bits_equal']
            for i in range(50):
                info=previous.tensor_info(layers[i]);assert info['finite']
                row={'layer':i,'output':info,'original_CPU_SHA256':cpu_layer_sha[i]}
                if i in gold_layers:row['vs_CPU']=previous.compare(layers[i],gold_layers[i])
                case['raw50_layers'].append(row)
            for i in range(27):
                info=previous.tensor_info(vislayers[i]);assert info['finite']
                case['raw27_vision_layers'].append({'layer':i,'output':info,'original_CPU_SHA256':proof['vision_layers'][i]['output']['sha256']})
            expected_name='current_patch_bias_erf_bc128_attention'if accurate_vision else ('current_patch_bias_erf'if fix else 'native')
            control=expected_vision[expected_name]
            assert case['vision_merged_vs_CPU']['npu']['sha256']==control['merged_SHA256']
            assert [x['output']['sha256']for x in case['raw27_vision_layers']]==control['raw27_SHA256']
            assert [x['npu']['sha256']for x in case['vision_DeepStack_vs_CPU']]==control['DeepStack_SHA256']
            case['matches_actual_R101_vision_merged_and_all27_boundary_SHA']=True
            case['vision_attention_snapshot']=vision_attention.snapshot()
            # Persist actual inference output outside all timed calls, never used
            # as a cached conditioning input. Future real DiT-conditioning CPU
            # controls can consume this without replaying or fabricating outputs.
            if rank==0:
                saved_path=ROOT/('NPU_selected50_'+name+'.pt')
                assert not saved_path.exists()
                torch.save({'case':name,'selected50':first.detach().contiguous(),
                            'input_ids':payload['input_ids'].clone(),
                            'visual_pos_masks':cpu_inputs['visual_pos_masks'].clone(),
                            'reference_receipt_SHA256':previous.file_sha(previous.CPU_ROOT/'reference.json')},saved_path)
                case['actual_NPU_output_saved_outside_timer']={'path':str(saved_path),'file_SHA256':previous.file_sha(saved_path)}
            layer_hashes[name]=[x['output']['sha256']for x in case['raw50_layers']]+[x['output']['sha256']for x in case['raw27_vision_layers']]
            with torch.inference_mode():repeat=equivalent(collect=False);stock_probe=stock()
            case['repeat_vs_first']=previous.compare(repeat,first);case['stock_vs_equivalent']=previous.compare(stock_probe,first)
            assert case['repeat_vs_first']['raw_bits_equal']and case['stock_vs_equivalent']['raw_bits_equal']
            stock_sha=previous.tensor_info(stock_probe)['sha256']
            for _ in range(5):
                torch.distributed.barrier()
                torch.npu.synchronize();start=time.perf_counter()
                with torch.inference_mode():warm=stock()
                torch.npu.synchronize();case['warm_stock_s'].append(time.perf_counter()-start)
                info=previous.tensor_info(warm);assert info['sha256']==stock_sha
                case['warm_output_SHA256'].append(info['sha256'])
            case['median_stock_s']=statistics.median(case['warm_stock_s'])
            case['warm_scope']='real private encode_ids includes pixels/newvision,all50text,HCCL/layout/packing,D2H;no hooks or CPUprecomputed condition'
            case['geometry_snapshot']=geometry.snapshot();case['attention_snapshot']=attention.snapshot()
            assert all(row['sha256_unchanged_since_prime']for row in case['geometry_snapshot']['cached'].values())
            r['cases'][name]=case;outputs[name]=first;writer.write(r)
            if name=='native':
                assert previous.tensor_info(first)['sha256']=='e34340b57f0a800cfd3c479a91dc7ea55ee86be934d07a87d91d296eb9678a5e'
                r['native_matches_R82_output_SHA']=True
            del layers,vislayers,repeat,stock_probe,warm
        # ExternalCPUtextconditions are accuracy controls only,NEVER above fullencoderwarm.
        original_inputs=captured['text_inputs'];device=torch.device('npu',rank)
        assert previous.compare(original_inputs['position_ids'],cpu_inputs['position_ids'])['raw_bits_equal']
        cpu_on_npu={'inputs_embeds':cpu_inputs['inputs_embeds'].reshape(original_inputs['inputs_embeds'].shape).to(device),
                    'deepstack':[x.to(device)for x in cpu_inputs['deepstack_visual_embeds']]}
        assert previous.compare(cpu_on_npu['inputs_embeds'],cpu_inputs['inputs_embeds'])['raw_bits_equal']
        r['CPU_condition_controls']={}
        for name,precise in(('native_attention',False),('precise_attention',True)):
            attention.set_enabled(precise)
            with torch.inference_mode():
                out=encoder.text_model(cpu_on_npu['inputs_embeds'].clone(),original_inputs['position_ids'],
                                       visual_pos_masks=original_inputs['visual_pos_masks'],deepstack_visual_embeds=[x.clone()for x in cpu_on_npu['deepstack']]).cpu()
                repeated=encoder.text_model(cpu_on_npu['inputs_embeds'].clone(),original_inputs['position_ids'],
                                            visual_pos_masks=original_inputs['visual_pos_masks'],deepstack_visual_embeds=[x.clone()for x in cpu_on_npu['deepstack']]).cpu()
            same=previous.compare(repeated,out);assert same['raw_bits_equal']
            r['CPU_condition_controls'][name]={'selected50_vs_CPU':previous.compare(out,gold),'repeat':same,'not_full_encoder_timing':True}
            if not precise:assert previous.tensor_info(out)['sha256']=='851a626fac934eb95ac9e370d6dd24f9ab936711ecef73d0961ed694fa7dc059'
        bias.set_enabled(False);patch.set_enabled(False);erf.set_enabled(False);attention.set_enabled(False);vision_attention.set_enabled(False);geometry.set_enabled(False)
        with torch.inference_mode():restored=stock()
        r['native_recovery']=previous.compare(restored,outputs['native']);assert r['native_recovery']['raw_bits_equal']
        r['geometry_final']=geometry.snapshot()
        assert all(row['sha256_unchanged_since_prime']for row in r['geometry_final']['cached'].values())
        r['parameter_and_fixture_SHA_unchanged']=param_before==previous.parameters_sha(encoder)and input_before==previous.inputs_sha(payload)
        assert r['parameter_and_fixture_SHA_unchanged']
        r['CPU_reference_SHA_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_SHA_unchanged'];r['uniform_Half_self_check']=encoder.r79_self_check()
        r['memory']={'allocated':int(torch.npu.memory_allocated(rank)),'max_allocated':int(torch.npu.max_memory_allocated(rank)),'reserved':int(torch.npu.memory_reserved(rank))}
        r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        for h in handles:h.remove()
        if vision_attention is not None:vision_attention.uninstall()
        if geometry is not None:geometry.uninstall()
        if erf is not None:erf.uninstall()
        if patch is not None:patch.uninstall()
        if bias is not None:bias.uninstall()
        if attention is not None:attention.uninstall()
        if kv is not None:kv.uninstall()
        if group is not None:group.destroy()
        if dist_ready:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        writer.write(r)


if __name__=='__main__':main()
