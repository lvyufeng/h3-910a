#!/usr/bin/env python3
"""Same-input NPU H3 checks at five real GPU-trajectory sigma positions.

Explicitly teacher-forced diagnostic input snapshots, not a production conditioning
cache. Uses the original full50-layer transformer and the R123 verified FP16 matrix /
FP32 range contract with patch/time residual guards. All actual model compute is on
owned 910A. Full video/audio prediction tensors are compared, not sampled errors.
Engineering acceptance: relativeRMS <=1% per modality, target0.5%; no zero-error demand.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--steps',default='0,10,25,40,48')
    args=parser.parse_args()
    root=Path(args.root)
    steps=[int(x) for x in args.steps.split(',')]
    assert steps and set(steps)<={0,10,25,40,48}
    destination=root/'NPU_sigma_checks124';destination.mkdir(exist_ok=True)
    import torch
    import torch_npu
    from safetensors import safe_open
    from vllm_omni.diffusion.data import OmniDiffusionConfig,DiffusionParallelConfig,AttentionConfig,AttentionSpec,TransformerConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_distributed_env
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig,get_sp_plan_from_model
    from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
    from vllm_omni.diffusion.forward_context import set_forward_context,get_forward_context
    from h3_replay123.transformer import MiniMaxH3DiTModel
    from h3_replay123.denoise_loop import MiniMaxH3DenoiseBranch
    from fp32_patch_residual_guard123 import install_patch_projection_guard123
    from fp32_time_residual_guard123 import install_time_projection_guard123
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    torch.set_num_threads(4);device=torch.device(f'npu:{local_rank}')
    receipt=destination/f'sigma_checks_rank{rank}.json';assert not receipt.exists()
    report={'scope':__doc__,'completed':False,'rank':rank,'selected_steps':steps,'same_input_gate_limit_relative_RMS':0.01,
            'target_relative_RMS':0.005,'zero_error_required':False,'CPU_model_inference':False,'same_input_diagnostic_not_full_generation':True,
            'full_prediction_tensors_compared_on_rank0':True,'full_transformer_layers':50,'checks':[]}
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    def metric(candidate,reference):
        candidate=candidate.cpu().float();reference=reference.cpu().float()
        assert candidate.shape==reference.shape
        a=candidate.double();b=reference.double();d=a-b
        nrms=float(d.square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-30))
        brms=float(b.square().mean().sqrt())
        dot=float((a*b).sum())
        cosine=dot/max(float(a.square().sum().sqrt()*b.square().sum().sqrt()),1e-30)
        return {'relative_RMS':nrms,'RMSE':float(d.square().mean().sqrt()),'GPU_RMS':brms,
                'max_abs_error':float(d.abs().max()),'p99_abs_error':float(torch.quantile(d.abs().flatten(),0.99)),
                'max_abs_error_over_GPU_RMS':float(d.abs().max())/max(brms,1e-30),'cosine_similarity':cosine,
                'finite':bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all()),
                'accepted_under_1percent_gate':nrms<=0.01,'meets_0point5percent_target':nrms<=0.005,'shape':list(a.shape)}
    token=(root/'activation_transfer_token124.txt').read_text().strip()
    def fetch_snapshot(step):
        local=root/f'GPU_snapshot_step{step:02d}.pt';metadata=root/f'GPU_snapshot_step{step:02d}.json'
        if local.is_file() and metadata.is_file():
            sr=json.loads(metadata.read_text());assert hashlib.sha256(local.read_bytes()).hexdigest()==sr['sha256'];return sr
        deadline=time.monotonic()+7200
        while True:
            try:
                request=urllib.request.Request(f'http://100.64.122.53:8126/snapshot_step{step:02d}.json',headers={'X-H3-Activation-Token':token})
                with urllib.request.urlopen(request,timeout=30) as response:sr=json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code!=404:raise
                if time.monotonic()>deadline:raise TimeoutError('GPU trajectory snapshot not ready within the bounded wait') from exc
                threading.Event().wait(15)
        assert sr['step_index']==step and not sr['contains_model_weights']
        request=urllib.request.Request(f'http://100.64.122.53:8126/snapshot_step{step:02d}.pt',headers={'X-H3-Activation-Token':token})
        with urllib.request.urlopen(request,timeout=120) as response:data=response.read(40*1024**2)
        assert len(data)==sr['bytes'] and hashlib.sha256(data).hexdigest()==sr['sha256']
        assert not local.exists() and not metadata.exists()
        with local.open('xb') as f:f.write(data)
        metadata.write_text(json.dumps(sr,indent=2)+'\n')
        print(json.dumps({'GPU_snapshot_transferred_directly_to_NPU':step,'bytes':len(data),'hash_verified':True}),flush=True)
        return sr
    def wait_for_local_snapshot(step):
        # Do not leave seven peers inside an HCCL barrier while the GPU
        # advances to a later sigma. Wait on filesystem notifications first.
        import ctypes,select
        metadata=root/f'GPU_snapshot_step{step:02d}.json'
        local=root/f'GPU_snapshot_step{step:02d}.pt'
        libc=ctypes.CDLL(None,use_errno=True)
        libc.inotify_init1.argtypes=[ctypes.c_int];libc.inotify_init1.restype=ctypes.c_int
        libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];libc.inotify_add_watch.restype=ctypes.c_int
        fd=libc.inotify_init1(os.O_NONBLOCK)
        if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        try:
            if libc.inotify_add_watch(fd,str(root).encode(),0x00000100|0x00000008)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
            deadline=time.monotonic()+7200
            while True:
                try:
                    if local.is_file() and metadata.is_file():
                        sr=json.loads(metadata.read_text())
                        if local.stat().st_size==sr['bytes']:return
                except json.JSONDecodeError:pass
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Completed local GPU snapshot did not arrive')
                readable,_,_=select.select([fd],[],[],remaining)
                if readable:os.read(fd,65536)
        finally:os.close(fd)
    publish()
    try:
        initial_root=Path('/tmp/h3_align12301')
        manifest=json.loads((initial_root/'same_initial_sampling123.json').read_text())
        bundle_path=initial_root/'same_initial_sampling123.pt'
        assert hashlib.sha256(bundle_path.read_bytes()).hexdigest()==manifest['payload_sha256']
        bundle=torch.load(bundle_path,map_location='cpu',weights_only=True)
        report['initial_sampling_bundle_sha256']=manifest['payload_sha256']
        init_distributed_environment(world_size=8,rank=rank,local_rank=local_rank,distributed_init_method='env://',backend='hccl')
        initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='hccl')
        pc=DiffusionParallelConfig(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1)
        arch=json.loads((Path(args.checkpoint)/'config.json').read_text());assert arch['num_layers']==50
        od=OmniDiffusionConfig(model=str(Path(args.checkpoint).parent),model_class_name='MiniMaxH3Pipeline',dtype=torch.float16,
                              tf_model_config=TransformerConfig.from_dict(arch),parallel_config=pc,
                              diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA')),num_gpus=8)
        with set_current_diffusion_config(od),set_forward_context(omni_diffusion_config=od),torch.device('cpu'):
            model=MiniMaxH3DiTModel(od,quant_config=None)
        selected={name for name,p in model.named_parameters()}|{name for name,p in model.named_buffers()}
        index=json.loads((Path(args.checkpoint)/'model.safetensors.index.json').read_text())['weight_map']
        def weights():
            for filename in sorted({index[name] for name in selected}):
                with safe_open(str(Path(args.checkpoint)/filename),framework='pt',device='cpu') as f:
                    for name in sorted(selected):
                        if index[name]==filename:yield name,f.get_tensor(name)
        loaded=model.load_weights(weights());assert set(loaded)==selected
        model.post_load_weights();model.eval()
        precision_modules=install_patch_projection_guard123(model)+install_time_projection_guard123(model)
        report['precision_guard_modules']=precision_modules
        for name,module in model.named_children():
            if name!='blocks':module.to(device)
        def onload(module,inputs):module.to(device)
        def offload(module,inputs,output):module.to('cpu');return output
        for block in model.blocks:
            block.register_forward_pre_hook(onload);block.register_forward_hook(offload)
        branch=MiniMaxH3DenoiseBranch(packed=bundle['packed'],text_embeddings=bundle['text_embeddings'],token_tags=bundle['token_tags'],device=torch.device('cpu'))
        target=torch.nonzero(bundle['packed']['update_mask'],as_tuple=False).flatten().long()
        def move(value):
            if isinstance(value,torch.Tensor):return value.to(device)
            if isinstance(value,dict):return {key:move(v) for key,v in value.items()}
            return value
        with set_current_diffusion_config(od),set_forward_context(omni_diffusion_config=od),torch.inference_mode():
            apply_sequence_parallel(model,SequenceParallelConfig(ulysses_degree=4,ring_degree=1),get_sp_plan_from_model(model))
            get_forward_context().sp_plan_hooks_applied=True
            for step in steps:
                if rank==0:sr=fetch_snapshot(step)
                else:wait_for_local_snapshot(step)
                torch.distributed.barrier()
                snapshot_path=root/f'GPU_snapshot_step{step:02d}.pt'
                snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=True)
                assert snapshot['step_index']==step
                sv,sa=snapshot['sigma_video'].tolist(),snapshot['sigma_audio'].tolist()
                assert sv==bundle['sigmas_video'][step:step+2].tolist() and sa==bundle['sigmas_audio'][step:step+2].tolist()
                tv,ta=1.-sv[0],1.-sa[0]
                fk=branch.forward_kwargs(video_rows=snapshot['video_rows'],audio_rows=snapshot['audio_rows'],t_video=tv,t_audio=ta,
                                          imgvid_cond_timestep=max(tv,0.999),audio_ref_cond_timestep=max(ta,1.))
                fk=move(fk)
                torch.npu.synchronize();start=time.perf_counter()
                vv,av=model(**fk);torch.npu.synchronize()
                finite=bool(torch.isfinite(vv).all() and torch.isfinite(av).all())
                assert finite,step
                row={'step_index':step,'sigma_video':sv[0],'sigma_audio':sa[0],'model_outputs_finite':finite,
                     'elapsed_s_includes_CPU_offload_not_benchmark':time.perf_counter()-start}
                accepted=True
                if rank==0:
                    row['video_target_velocity']=metric(vv.cpu().index_select(0,target),snapshot['video_velocity'].index_select(0,target))
                    row['audio_velocity']=metric(av.cpu(),snapshot['audio_velocity'])
                    accepted=row['video_target_velocity']['accepted_under_1percent_gate'] and row['audio_velocity']['accepted_under_1percent_gate']
                    path=destination/f'NPU_prediction_step{step:02d}.pt'
                    with path.open('xb') as f:torch.save({'step_index':step,'video_velocity':vv.cpu(),'audio_velocity':av.cpu()},f)
                    row['snapshot_sha256']=sr['sha256'];row['accepted_under_1percent_gate']=accepted
                    print(json.dumps({'same_input_NPU_check':row},ensure_ascii=False),flush=True)
                flag=torch.tensor([int(accepted)],dtype=torch.int32,device=device)
                torch.distributed.broadcast(flag,src=0)
                report['checks'].append(row);publish()
                if not bool(flag.item()):
                    raise RuntimeError(f'Same-input DiT step{step} exceeded the agreed1% gate; full generation is not authorized by this gate')
        report['completed']=True
        report['all_selected_sigma_gates_accepted']=True
        publish()
        if rank==0:print(json.dumps({'NPU_representative_sigma_checks_completed':True,'accepted_steps':steps,'limit_relativeRMS':0.01}),flush=True)
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if torch.distributed.is_initialized():destroy_distributed_env()


if __name__=='__main__':main()
