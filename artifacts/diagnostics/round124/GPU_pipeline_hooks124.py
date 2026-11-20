#!/usr/bin/env python3
"""Capture actual GPU H3 model inputs/outputs on the known-successful vLLM-Omni path.

Live GPU encoder and tiled native keyframe VAE, original checkpoint and verified
range-safe FP16 transformer, actual engine offloader. No diagnostic reconstruction
of conditioning, CPU model baseline, student or LoRA. Five complete snapshots and
all49 actual forwards are recorded. Stop intentionally before VAE output decoding;
this is a numerical trajectory capture, not a claimed newly generated video.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time


def install():
    import torch
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
    from vllm.utils.torch_utils import set_default_torch_dtype
    root=Path(os.environ['H3_GPU_PIPELINE_CAPTURE_ROOT124'])
    out=root/'GPU_trajectory124';out.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    original_factory=vae_module._load_remote_component
    def FP32_factory(*args,**kwargs):
        with set_default_torch_dtype(torch.float32):return original_factory(*args,**kwargs)
    vae_module._load_remote_component=FP32_factory
    def rank():return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get('RANK','-1'))
    def sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def CPU(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().contiguous()
        if isinstance(value,dict):return {name:CPU(x) for name,x in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(CPU(x) for x in value)
        return value
    original_forward=MiniMaxH3DiTModel.forward
    def forward(self,**kwargs):
        step=getattr(self,'_trajectory_index124',0)
        started=time.perf_counter();current_rank=rank()
        receipt=out/f'pipeline_trajectory_rank{current_rank}.json'
        report=getattr(self,'_trajectory_report124',None)
        if report is None:
            report={'scope':__doc__,'completed':False,'rank':current_rank,'actual_DiT_evaluations':0,'known_successful_vllm_omni_pipeline_used':True,
                    'live_GPU_encoder_and_tiled_VAE_conditioning':True,'model_weights_downloaded_or_relayed_to_local':False,'steps':[],'snapshots':{}}
            object.__setattr__(self,'_trajectory_report124',report)
            report['parameter_dtype_counts']={str(dtype):sum(p.numel() for p in self.parameters() if p.dtype==dtype) for dtype in {p.dtype for p in self.parameters()}}
        if step==0 and current_rank==0:
            static={name:CPU(value) for name,value in kwargs.items() if name not in ['x','audio_x','unique_timesteps','inverse_indices']}
            path=out/'GPU_snapshot_context124.pt'
            with path.open('xb') as f:torch.save({'static_forward_kwargs':static},f)
            metadata={'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'contains_model_weights':False,'contains_actual_GPU_encoder_activations':True,'use_scope':'Explicit same-input diagnostic replay only, never hidden NPU generation conditioning'}
            (out/'GPU_snapshot_context124.json').write_text(json.dumps(metadata,indent=2)+'\n')
            report['context']=metadata
        vv,av=original_forward(self,**kwargs)
        finite=bool(torch.isfinite(vv).all() and torch.isfinite(av).all())
        if not finite:raise FloatingPointError(f'Known-successful GPU pipeline produced nonfinite prediction at step{step}')
        report['steps'].append({'step_index':step,'video_velocity_max_abs':float(vv.float().abs().max()),'audio_velocity_max_abs':float(av.float().abs().max()),
                                'all_predictions_finite':True,'elapsed_s_includes_engine_offload_and_capture':time.perf_counter()-started})
        if step in [0,10,25,40,48] and current_rank==0:
            img_pos=kwargs['img_pos_info']['position_ids'];audio_pos=kwargs['audio_pos_info']['position_ids']
            video_rows=kwargs['x'][0].index_select(0,img_pos).detach().cpu().contiguous()
            audio_rows=kwargs['audio_x'][0].index_select(0,audio_pos).detach().cpu().contiguous()
            sv=__import__('vllm_omni.diffusion.models.minimax_h3.time_request',fromlist=['minimax_h3_time_shift_sigmas']).minimax_h3_time_shift_sigmas(num_steps=50,shift_scale=12.)
            sa=__import__('vllm_omni.diffusion.models.minimax_h3.time_request',fromlist=['minimax_h3_time_shift_sigmas']).minimax_h3_time_shift_sigmas(num_steps=50,shift_scale=3.)
            payload={'step_index':step,'video_rows':video_rows,'audio_rows':audio_rows,'video_velocity':CPU(vv),'audio_velocity':CPU(av),
                     'unique_timesteps':CPU(kwargs['unique_timesteps']),'inverse_indices':CPU(kwargs['inverse_indices']),
                     'sigma_video':torch.tensor(sv[step:step+2],dtype=torch.float32),'sigma_audio':torch.tensor(sa[step:step+2],dtype=torch.float32)}
            path=out/f'snapshot_step{step:02d}.pt'
            with path.open('xb') as f:torch.save(payload,f)
            metadata={'step_index':step,'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'contains_model_weights':False,'context_sha256':report['context']['sha256'],'actual_known_successful_pipeline_inputs':True,
                      'video_sigma':float(sv[step]),'audio_sigma':float(sa[step]),'video_input_sha256_raw':sha(video_rows),
                      'audio_input_sha256_raw':sha(audio_rows),'video_velocity_sha256_raw':sha(vv),'audio_velocity_sha256_raw':sha(av)}
            (out/f'snapshot_step{step:02d}.json').write_text(json.dumps(metadata,indent=2)+'\n');report['snapshots'][str(step)]=metadata
        report['actual_DiT_evaluations']+=1;self._trajectory_index124=step+1
        if step==48:report['completed']=True
        receipt.write_text(json.dumps(report,indent=2)+'\n')
        return vv,av
    MiniMaxH3DiTModel.forward=forward
    def stop_before_decode(self,*args,**kwargs):
        current_rank=rank();report=getattr(self.transformer,'_trajectory_report124',{})
        assert report.get('completed') and report.get('actual_DiT_evaluations')==49
        (out/f'decode_intentionally_skipped_rank{current_rank}.json').write_text(json.dumps({'trajectory_complete':True,'decode_executed':False,'intentional_stop_not_generation_success':True})+'\n')
        raise RuntimeError('H3_GPU_TRAJECTORY124_COMPLETE_DECODE_INTENTIONALLY_SKIPPED')
    pipeline.MiniMaxH3Pipeline.decode=stop_before_decode
    (root/f'GPU_bootstrap_pid{os.getpid()}124.json').write_text(json.dumps({'scope':__doc__,'installed':True,'pid':os.getpid(),'no_shared_source_edits':True},indent=2)+'\n')
