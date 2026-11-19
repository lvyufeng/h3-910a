#!/usr/bin/env python3
"""Actual eight-rank original-Half complete vision: replicated versus true block TP8."""
from pathlib import Path
import json
import os
import statistics
import time
ROOT=Path(os.environ.get('R107_ROOT','/tmp/h3_round107_vision_tp01'))


def main():
    import torch
    import torch_npu
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    import full_row_fp32_fixed84 as previous
    from evidence_io107 import EvidenceWriter
    from runtime5gm83 import Runtime5GM
    from vision_adapter87 import VisionBiasAdapter87
    from patch_bias89 import PatchBiasAdapter89
    from erf_gelu_fixed90 import ErfGeluAdapter90
    from geometry_cache91 import VisionGeometryCache91
    from flash_engine101 import FlashVisionAttention101
    from head2_attention107 import Head2VisionAttention107
    from vision_gemm_backend107 import VisionGemmBackend107
    from vision_tp_adapter107 import VisionTPAdapter107
    from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
    writer=EvidenceWriter(ROOT/('vision_tp_rank%d.json'%rank))
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'cases':{},
       'not_full_encoder_timing':True,'no_FloatLinear_teacher_BFgoldcast_hidden_conditioning_or_fake_norm':True,
       'only27blocks_TP8_patch_mergers_norm_geometry_replicated':True}
    group=None;ready=False;tp=None;full=None;head2=None;geometry=None;bias=None;patch=None;erf=None;handles=[]
    try:
        torch.set_num_threads(2);torch.npu.set_device(rank);previous.ROOT=ROOT
        init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl');ready=True
        initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
        group=GroupCoordinator([list(range(size))],rank,'hccl')
        proof=previous.load_cpu_gate(r);candidate=previous.load_candidate(r)
        assert previous.file_sha(previous.FIXTURE)==previous.FIXTURE_SHA256
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),local_files_only=True).vision_config
        assert(cfg.hidden_size,cfg.intermediate_size,cfg.num_heads,cfg.depth)==(1152,4304,16,27)
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
        assert set(state)==wanted;vision.load_state_dict(state,strict=True,assign=True);del state
        vision.to(device=torch.device('npu',rank));vision.eval()
        vision.rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(cfg.hidden_size//cfg.num_heads//2)
        assert all(p.dtype==torch.float16 for p in vision.parameters())
        before=previous.parameters_sha(vision)
        payload=torch.load(previous.FIXTURE,map_location='cpu',weights_only=True)
        pixels=payload['pixel_values'].to(torch.device('npu',rank));grids=payload['image_grid_thw'].to(torch.device('npu',rank))
        assert pixels.shape==(8064,1536)and grids.tolist()==[[1,48,84],[1,48,84]]
        inputsha=previous.tensor_info(pixels)['sha256'];oracle=torch.load(previous.CPU_ROOT/'FP16_vision.pt',map_location='cpu',weights_only=True)
        bias=VisionBiasAdapter87(candidate,Runtime5GM(rank),ROOT/'bias_kernels');bias.prime(vision)
        patch=PatchBiasAdapter89(vision,Runtime5GM(rank),ROOT/'bias_kernels')
        erf=ErfGeluAdapter90(vision,Runtime5GM(rank),ROOT/'erf_kernels');erf.prepare()
        full=FlashVisionAttention101(candidate,Runtime5GM(rank),ROOT/'replicated_attention_kernels');r['full_attention_prepare']=full.prepare()
        head2=Head2VisionAttention107(candidate,Runtime5GM(rank),ROOT/'head2_kernels');r['head2_prepare']=head2.prepare()
        backend=VisionGemmBackend107(Runtime5GM(rank),ROOT/'tp_backend_kernels')
        tp=VisionTPAdapter107(candidate,group,backend,head2.forward,m=8064);r['TP_prime']=tp.prime(vision)
        geometry=VisionGeometryCache91(vision);r['geometry_prime']=geometry.prime(grids);geometry.set_enabled(True)
        expected=json.loads((ROOT/'expected_vision101.json').read_text());first_native=None
        cases=(('native',False,False,False,'native'),
               ('corrected_replica_native_attention',True,False,False,'current_patch_bias_erf'),
               ('corrected_replica_BC128',True,True,False,'current_patch_bias_erf_bc128_attention'),
               ('corrected_TP8_BC128',True,False,True,None))
        for name,fixed,accurate,use_tp,controlname in cases:
            bias.set_enabled(fixed);patch.set_enabled(fixed);erf.set_enabled(fixed);full.set_enabled(accurate)
            tp.set_enabled(use_tp);head2.set_enabled(use_tp);layers=[];before_stats=tp.snapshot()['stats']
            def hook_layer(i):
                def hook(module,args,out):
                    info=previous.tensor_info(out);assert info['finite'];layers.append({'layer':i,'output':info})
                return hook
            for i,block in enumerate(vision.blocks):handles.append(block.register_forward_hook(hook_layer(i)))
            try:
                with torch.inference_mode():merged,deepstack=vision(pixels,grids)
            finally:
                for handle in handles:handle.remove()
                handles.clear()
            assert len(layers)==27
            c={'fixes':fixed,'replicated_BC128_attention':accurate,'vision_TP8':use_tp,'geometry_cache':True,
               'merged_vs_CPU':previous.compare(merged,oracle['merged']),
               'DeepStack_vs_CPU':[previous.compare(a,b)for a,b in zip(deepstack,oracle['deepstack'])],
               'raw27_layers':layers,'warm_s':[],'TP_first_snapshot':tp.snapshot()}
            if use_tp:
                expected_counts=tp.snapshot()['expected_counts_per_forward'];after_stats=tp.snapshot()['stats']
                fields={'attention_calls':'enabled_attention_calls','mlp_calls':'enabled_mlp_calls'}
                delta={key:after_stats[fields.get(key,key)]-before_stats[fields.get(key,key)]for key in expected_counts if key!='primed_weight_modules'}
                # Priming is a persistent count, not108 new weights everyforward.
                delta['primed_weight_modules']=after_stats['primed_weight_modules']
                assert delta==expected_counts;c['actual_one_vision_TP_counts']=delta
            if controlname is not None:
                control=expected[controlname]
                assert c['merged_vs_CPU']['npu']['sha256']==control['merged_SHA256']
                assert [v['output']['sha256']for v in layers]==control['raw27_SHA256']
                assert [v['npu']['sha256']for v in c['DeepStack_vs_CPU']]==control['DeepStack_SHA256']
                c['matches_original_R101_all27_DS_merger_bits']=True
            with torch.inference_mode():again,again_ds=vision(pixels,grids)
            assert previous.compare(again,merged)['raw_bits_equal']and all(previous.compare(a,b)['raw_bits_equal']for a,b in zip(again_ds,deepstack))
            for _ in range(5):
                torch.distributed.barrier();torch.npu.synchronize();begin=time.perf_counter()
                with torch.inference_mode():out,out_ds=vision(pixels,grids)
                torch.npu.synchronize();c['warm_s'].append(time.perf_counter()-begin)
                assert previous.compare(out,merged)['raw_bits_equal']and all(previous.compare(a,b)['raw_bits_equal']for a,b in zip(out_ds,deepstack))
            c['median_s']=statistics.median(c['warm_s']);c['all_repeats_merged_and_DS_bits_equal']=True
            c['timing_scope']='actualresidentpixels, all27blocks, TPcolumn/FloatrowHCCL+packing+GELU+localH2, allmergers; no hooks/CPUprecomputedconditioning'
            c['TP_after_warm']=tp.snapshot();c['head2']=head2.snapshot();r['cases'][name]=c;writer.write(r)
            if name=='native':first_native=(merged.detach().cpu(),[x.detach().cpu()for x in deepstack])
        tp.set_enabled(False);head2.set_enabled(False);bias.set_enabled(False);patch.set_enabled(False);erf.set_enabled(False);full.set_enabled(False)
        with torch.inference_mode():restored,restored_ds=vision(pixels,grids)
        r['native_recovery']={'merged':previous.compare(restored,first_native[0]),'deepstack':[previous.compare(a,b)for a,b in zip(restored_ds,first_native[1])]}
        assert r['native_recovery']['merged']['raw_bits_equal']and all(v['raw_bits_equal']for v in r['native_recovery']['deepstack'])
        r['original_parameters_unchanged']=before==previous.parameters_sha(vision);assert r['original_parameters_unchanged']
        r['original_pixels_unchanged']=inputsha==previous.tensor_info(pixels)['sha256'];assert r['original_pixels_unchanged']
        r['original_CPU_gold_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9';assert r['original_CPU_gold_unchanged']
        r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        for handle in handles:handle.remove()
        if tp is not None:r['TP_final']=tp.snapshot();tp.uninstall()
        if head2 is not None:head2.uninstall()
        if geometry is not None:geometry.uninstall()
        if full is not None:full.uninstall()
        if erf is not None:erf.uninstall()
        if patch is not None:patch.uninstall()
        if bias is not None:bias.uninstall()
        if group is not None:group.destroy()
        if ready:destroy_model_parallel();torch.distributed.destroy_process_group()
        writer.write(r)


if __name__=='__main__':main()
