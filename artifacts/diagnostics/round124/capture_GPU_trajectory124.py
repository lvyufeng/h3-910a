#!/usr/bin/env python3
"""GPU real49-step H3 trajectory with exact replay inputs and five full-state snapshots.

Original full50-layer transformer, successful FP16-matrix/FP32-range-safe source,
TP2/USP4, actual saved GPU encoder activations, exact R123 noise/keyframe/sigma bundle.
The conditioning is explicitly reused for a numerical diagnostic, not a performance
cache. All model forwards execute on CUDA. CPU is weight storage/setup only. No
model weights are downloaded or relayed to the workstation, no BF16, no clipping,
no student/LoRA, and no VAE decode in this trajectory-capture run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--sampling-root',required=True)
    parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args()
    root=Path(args.root);sampling_root=Path(args.sampling_root)
    destination=root/'GPU_trajectory124'
    destination.mkdir(exist_ok=True)
    import torch
    from safetensors import safe_open
    from vllm_omni.diffusion.data import OmniDiffusionConfig,DiffusionParallelConfig,AttentionConfig,AttentionSpec,TransformerConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_distributed_env
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig,get_sp_plan_from_model
    from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
    from vllm_omni.diffusion.forward_context import set_forward_context,get_forward_context
    from h3_replay123.transformer import MiniMaxH3DiTModel
    from h3_replay123.denoise_loop import MiniMaxH3DenoiseBranch
    from h3_replay123.scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0,minimax_h3_euler_eta0_step
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank);device=torch.device(f'cuda:{local_rank}')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.set_num_threads(4)
    selected_steps=[0,10,25,40,48]
    report={'scope':__doc__,'completed':False,'rank':rank,'actual_DiT_evaluations':0,'full_transformer_layers':50,
            'no_CPU_model_inference':True,'GPU_encoder_conditioning_explicitly_reused':True,
            'no_local_model_weights':True,'precision_contract':'Successful GPU FP16 GEMM inputs/weights with original FP32 projections, FP32 residual/RMS and wide-output row GEMMs',
            'parallel_config':{'TP':2,'USP':4,'world':8},'selected_zero_based_steps':selected_steps,
            'CPU_layerwise_offload_is_diagnostic_not_speed_benchmark':True,'steps':[],'snapshots':{}}
    receipt=destination/f'trajectory_rank{rank}.json';assert not receipt.exists()
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def summary(x):return {'shape':list(x.shape),'dtype':str(x.dtype),'finite':bool(torch.isfinite(x).all()),'max_abs':float(x.abs().amax())}
    publish()
    try:
        start=time.perf_counter()
        manifest=json.loads((sampling_root/'same_initial_sampling123.json').read_text())
        bundle_path=sampling_root/'same_initial_sampling123.pt'
        assert hashlib.sha256(bundle_path.read_bytes()).hexdigest()==manifest['payload_sha256']
        bundle=torch.load(bundle_path,map_location='cpu',weights_only=True)
        report['sampling_bundle_sha256']=manifest['payload_sha256']
        report['transformer_source_sha256']=hashlib.sha256((sampling_root/'h3_replay123/transformer.py').read_bytes()).hexdigest()
        assert report['transformer_source_sha256']=='7cf0c53e6d9199b7944d9026c2bdc1120b1e32084cffe266c434d2d6449b8029'
        init_distributed_environment(world_size=8,rank=rank,local_rank=local_rank,distributed_init_method='env://',backend='nccl')
        initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='nccl')
        architecture=json.loads((Path(args.checkpoint)/'config.json').read_text());assert architecture['num_layers']==50
        pc=DiffusionParallelConfig(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1)
        od=OmniDiffusionConfig(model=str(Path(args.checkpoint).parent),model_class_name='MiniMaxH3Pipeline',dtype=torch.float16,
                              tf_model_config=TransformerConfig.from_dict(architecture),parallel_config=pc,
                              diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA')),num_gpus=8)
        with set_current_diffusion_config(od),set_forward_context(omni_diffusion_config=od),torch.device('cpu'):
            model=MiniMaxH3DiTModel(od,quant_config=None)
        names={name for name,p in model.named_parameters()}|{name for name,p in model.named_buffers()}
        index=json.loads((Path(args.checkpoint)/'model.safetensors.index.json').read_text())['weight_map']
        assert names<=set(index)
        def weights():
            for filename in sorted({index[name] for name in names}):
                with safe_open(str(Path(args.checkpoint)/filename),framework='pt',device='cpu') as f:
                    for name in sorted(names):
                        if index[name]==filename:yield name,f.get_tensor(name)
        loaded=model.load_weights(weights());assert set(loaded)==names
        model.post_load_weights();model.eval()
        for name,module in model.named_children():
            if name!='blocks':module.to(device)
        def bring_block_to_GPU(module,inputs):
            module.to(device)
        for block in model.blocks:
            block.register_forward_pre_hook(bring_block_to_GPU)
            def offload(module,inputs,output):
                module.to('cpu')
                return output
            block.register_forward_hook(offload)
        report['setup_s_not_benchmark']=time.perf_counter()-start
        branch=MiniMaxH3DenoiseBranch(packed=bundle['packed'],text_embeddings=bundle['text_embeddings'],token_tags=bundle['token_tags'],device=device)
        video=bundle['initial_video_rows_with_keyframes'].to(device).clone()
        audio=bundle['initial_audio_rows'].to(device).clone()
        target_cpu=torch.nonzero(bundle['packed']['update_mask'],as_tuple=False).flatten().long()
        anchor_cpu=torch.nonzero(~bundle['packed']['update_mask'],as_tuple=False).flatten().long()
        target=target_cpu.to(device);anchor=anchor_cpu.to(device)
        anchors=bundle['keyframe_noise_aug_rows'].to(device)
        sv=bundle['sigmas_video'].tolist();sa=bundle['sigmas_audio'].tolist()
        assert len(sv)==len(sa)==50
        with set_current_diffusion_config(od),set_forward_context(omni_diffusion_config=od),torch.inference_mode():
            apply_sequence_parallel(model,SequenceParallelConfig(ulysses_degree=4,ring_degree=1),get_sp_plan_from_model(model))
            get_forward_context().sp_plan_hooks_applied=True
            for step in range(49):
                torch.cuda.synchronize();start_step=time.perf_counter()
                tv,ta=1.-sv[step],1.-sa[step]
                fk=branch.forward_kwargs(video_rows=video,audio_rows=audio,t_video=tv,t_audio=ta,
                                          imgvid_cond_timestep=max(tv,0.999),audio_ref_cond_timestep=max(ta,1.))
                vv,av=model(**fk)
                torch.cuda.synchronize()
                vs,aas=summary(vv),summary(av)
                assert vs['finite'] and aas['finite'],step
                if step in selected_steps and rank==0:
                    payload={'step_index':step,'video_rows':video.cpu().contiguous(),'audio_rows':audio.cpu().contiguous(),
                             'video_velocity':vv.cpu().contiguous(),'audio_velocity':av.cpu().contiguous(),
                             'sigma_video':torch.tensor([sv[step],sv[step+1]],dtype=torch.float32),
                             'sigma_audio':torch.tensor([sa[step],sa[step+1]],dtype=torch.float32)}
                    path=destination/f'snapshot_step{step:02d}.pt'
                    with path.open('xb') as f:torch.save(payload,f)
                    sr={'step_index':step,'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'contains_model_weights':False,'video_sigma':sv[step],'audio_sigma':sa[step],
                        'video_input_sha256_raw':tensor_sha(video),'audio_input_sha256_raw':tensor_sha(audio),
                        'video_velocity_sha256_raw':tensor_sha(vv),'audio_velocity_sha256_raw':tensor_sha(av)}
                    (destination/f'snapshot_step{step:02d}.json').write_text(json.dumps(sr,indent=2)+'\n')
                    report['snapshots'][str(step)]=sr
                vx=video.index_select(0,target)
                x0v=minimax_h3_rf_v_to_x0(vx,vv.index_select(0,target),torch.tensor(tv,dtype=torch.float32,device=device))
                newv=minimax_h3_euler_eta0_step(vx,x0v,sigma_curr=sv[step],sigma_next=sv[step+1])
                video=video.clone().index_copy(0,target,newv);video.index_copy_(0,anchor,anchors)
                x0a=minimax_h3_rf_v_to_x0(audio,av,torch.tensor(ta,dtype=torch.float32,device=device))
                audio=minimax_h3_euler_eta0_step(audio,x0a,sigma_curr=sa[step],sigma_next=sa[step+1])
                torch.cuda.synchronize()
                assert torch.isfinite(video).all() and torch.isfinite(audio).all()
                report['actual_DiT_evaluations']+=1
                elapsed=time.perf_counter()-start_step
                record={'step_index':step,'video_sigma':sv[step],'audio_sigma':sa[step],'video_velocity':vs,'audio_velocity':aas,'elapsed_s_includes_offload_capture_not_benchmark':elapsed}
                report['steps'].append(record);publish()
                if rank==0:print(json.dumps({'trajectory_step_completed':step,'evaluations':step+1,'snapshot_saved':step in selected_steps,'elapsed_s':elapsed}),flush=True)
        assert report['actual_DiT_evaluations']==49
        if rank==0:
            assert torch.equal(video.cpu().index_select(0,anchor_cpu).view(torch.uint8),bundle['keyframe_noise_aug_rows'].view(torch.uint8))
            path=destination/'final_state.pt'
            with path.open('xb') as f:torch.save({'video_rows':video.cpu(),'audio_rows':audio.cpu(),'target_video_row_indices':target_cpu},f)
            report['final_state_file']=str(path);report['final_state_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            report['anchors_pinned_raw_bit_identical']=True
        report['completed']=True;publish()
        if rank==0:print(json.dumps({'GPU_trajectory_completed':True,'actual_evaluations':49,'selected_snapshots':selected_steps}),flush=True)
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if torch.distributed.is_initialized():destroy_distributed_env()


if __name__=='__main__':main()
