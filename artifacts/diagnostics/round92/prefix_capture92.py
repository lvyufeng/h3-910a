#!/usr/bin/env python3
"""Actual TP8 prefix0..24 captured bit-identical to R91;not a complete encoder timing."""
from pathlib import Path
import json
import os
ROOT=Path('/tmp/h3_round92_case01')


def main():
    import torch
    import torch_npu
    import best_text87 as previous
    from evidence_io92 import EvidenceWriter
    from safetensors import safe_open
    from runtime5gm83 import Runtime5GM
    from attention_engine85 import AccurateAttention85
    from vision_adapter87 import VisionBiasAdapter87
    from patch_bias89 import PatchBiasAdapter89
    from erf_gelu_fixed90 import ErfGeluAdapter90
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
    writer=EvidenceWriter(ROOT/('prefix_rank%d.json'%rank))
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,
       'source_SHA256':previous.file_sha(Path(__file__)),'full_encoder_precision_accepted':False,
       'whole_goal_complete':False,'numerical_tolerance':None,'cases':{},
       'FP32_row_adapter_not_installed':True,'all_linear_model_weights_Half':True,
       'not_full_encoder_timing':True,
       'R88_slow_vision_attention_and_R89_redundant_Float_nonlinears_not_installed':True}
    kv=None;attention=None;bias=None;patch=None;erf=None;group=None;dist_ready=False;handles=[]
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
        r['setup_excluded_from_warm']={'all_vision_weights_original_Half':True,'kernel_prebuild_and_layout_cache':True,
                                       'KV_padding':kv.snapshot(),'erf':erf.snapshot(),'vision_bias':bias.snapshot(),'patch':patch.snapshot()}
        param_before=previous.parameters_sha(encoder);input_before=previous.inputs_sha(payload)
        cpu_inputs=torch.load(previous.CPU_ROOT/'FP16_multimodal_inputs.pt',map_location='cpu',weights_only=True)
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
        expected=json.loads((ROOT/'expected_raw24_92.json').read_text())['cases']
        inputs={};frozen={}
        class PrefixCaptured(Exception):pass
        def raw24_post(module,args,out):
            assert out.shape==(1,2043,5120)and out.dtype==torch.float16
            frozen['state']=out.detach().cpu().contiguous()
            raise PrefixCaptured()
        hook=encoder.text_model.layers[24].register_forward_hook(raw24_post)
        try:
            for name,fix,precise in(('native',False,False),('corrected_vision_native_text',True,False),
                                   ('corrected_vision_precise_text_geometry_cache',True,True)):
                bias.set_enabled(fix);patch.set_enabled(fix);erf.set_enabled(fix)
                attention.set_enabled(precise);captured.clear()
                completed24=[]
                for _ in range(2):
                    frozen.clear()
                    try:
                        with torch.inference_mode():equivalent(collect=True)
                    except PrefixCaptured:pass
                    else:raise RuntimeError('raw24hookdidnotstoptheprefixbefore25')
                    state=frozen['state'];info=previous.tensor_info(state)
                    assert info['sha256']==expected[name]['sha256'],'actualprefixdoesnotreproduceR91raw24'
                    completed24.append(state)
                assert previous.compare(completed24[0],completed24[1])['raw_bits_equal']
                assert previous.compare(captured['text_inputs']['position_ids'],cpu_inputs['position_ids'])['raw_bits_equal']
                inputs[name]=completed24[0]
                r['cases'][name]={'raw24':previous.tensor_info(completed24[0]),'raw24_matches_R91':True,
                                 'repeat_bit_identical':True,'no_raw25to49_executed':True,
                                 'original_position_ids_equal_CPU_reference':True}
                writer.write(r)
        finally:hook.remove()
        if rank==0:
            p=ROOT/'NPU_raw24_inputs92.pt';assert not p.exists()
            torch.save({'inputs':inputs,'position_ids':cpu_inputs['position_ids'],
                        'source_raw24_SHAs':{name:row['sha256']for name,row in expected.items()},
                        'captured_source_SHA256':r['source_SHA256']},p)
            r['input_bundle_file_SHA256']=previous.file_sha(p)
        torch.distributed.barrier()
        r['parameter_and_fixture_SHA_unchanged']=param_before==previous.parameters_sha(encoder)and input_before==previous.inputs_sha(payload)
        assert r['parameter_and_fixture_SHA_unchanged']
        r['CPU_reference_SHA_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_SHA_unchanged'];r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        for h in handles:h.remove()
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
