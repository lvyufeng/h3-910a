"""R127 request-scoped original R124+FA integration. Observes, never optimizes math."""
from __future__ import annotations
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from runtime_context127 import event,flush,request_index,request_directory,begin_request,end_request

FIXTURES=Path('/tmp/h3_align12401')


def install():
    import torch
    import torch_npu
    from h3_replay123.transformer import MiniMaxH3DiTModel as VerifiedDiT,MiniMaxH3Attention
    from fp32_patch_residual_guard123 import install_patch_projection_guard123
    from fp32_time_residual_guard123 import install_time_projection_guard123
    from npu_half_nan_compat124 import ieee_nonfinite_masks,half_nan_scope
    from vllm.utils.torch_utils import set_default_torch_dtype
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline
    import vllm_omni.diffusion.models.minimax_h3.vae as vae
    import observe_runtime127 as observe
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    for name in ('H3_R126_VAE_OPERAND_CACHE','VLLM_OMNI_910A_FAST_RESIDUAL','VLLM_OMNI_910A_FAST_ACT'):
        assert os.environ.get(name,'0')!='1',('Unapproved candidate enabled',name)
    factory=vae._load_remote_component
    def original_FP32_factory(*args,**kwargs):
        with set_default_torch_dtype(torch.float32):result=factory(*args,**kwargs)
        assert all(p.dtype==torch.float32 for p in result.parameters() if p.is_floating_point())
        return result
    vae._load_remote_component=original_FP32_factory
    def preserve_original_parameters(decoder,*,device):
        assert all(p.dtype==torch.float32 for p in decoder.parameters() if p.is_floating_point())
        event('native_VAE_FP32_parameter_rewrite_disabled',fused_VAE_installer_return=False)
        return False
    vae.install_h3_vae_optimizations=preserve_original_parameters
    spec=importlib.util.spec_from_file_location('h3_live_encoder127',FIXTURES/'NPU_encoder_FP16_124.py')
    encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
    manifest=json.loads((FIXTURES/'isolated_integration_sources124.json').read_text());assert manifest['completed']
    def stats(x):
        cpu=x.detach().cpu().float()
        return {'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'finite':bool(torch.isfinite(cpu).all()),
                'max_abs':float(cpu.abs().amax()),'D2H_finiteness_audit_not_CPU_model_inference':True}
    class OriginalGuardedDiT127(VerifiedDiT):
        def __init__(self,od_config,quant_config=None,*,diffusers_weights=False):
            if diffusers_weights or quant_config is not None:raise ValueError('Original unquantized checkpoint only')
            super().__init__(od_config,quant_config=None)
            self.diffusers_weights=False;self._request_steps127=0;self._total_steps127=0;self._checks127=0
            self._precision_modules127=install_patch_projection_guard123(self)+install_time_projection_guard123(self)
            for name,module in self.named_modules():
                if not (name.endswith('.qkv_proj') or name.endswith('.mlp.fc1') or name.endswith('.adaln_proj.linear')):continue
                def check(layer,inputs,result,name=name):
                    tensor=result[0] if isinstance(result,tuple) else result
                    if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float16:return
                    self._checks127+=1
                    nan,posinf,neginf=ieee_nonfinite_masks(tensor)
                    nonfinite=int((nan|posinf|neginf).sum(dtype=torch.int32))
                    boundary=int((tensor.abs()==65504).sum(dtype=torch.int32))
                    if nonfinite or boundary:
                        event('half_matrix_range_warning',module=name,nonfinite=nonfinite,boundary=boundary)
                        raise FloatingPointError('Half range failed at '+name)
                module.register_forward_hook(check)
            event('matched_transformer_constructed',main_layers=len(self.blocks),source_sha256=manifest['source_digests']['transformer.py'],
                precision_modules=self._precision_modules127,parameter_dtype_counts={str(d):sum(p.numel() for p in self.parameters() if p.dtype==d) for d in {p.dtype for p in self.parameters()}})
        def forward(self,**kwargs):
            assert request_index() in (1,2,3)
            kwargs.pop('video_token_layout',None)
            assert int(kwargs.get('packed_seq_params',{}).get('num_requests',1))==1
            step=self._request_steps127;assert 0<=step<49
            start=time.perf_counter();before=self._checks127
            event('dit_step_begin',step_index=step,total_step_index=self._total_steps127,
                  unique_timesteps=kwargs['unique_timesteps'].detach().cpu().tolist())
            result=super().forward(**kwargs);count=self._checks127-before;assert count==155
            video,audio=map(stats,result)
            assert video['finite'] and audio['finite']
            event('dit_step_end',step_index=step,total_step_index=self._total_steps127,
                elapsed_s_includes_offload_and_D2H_audit=time.perf_counter()-start,
                actual_Half_matrix_modules_this_step=count,total_checks=self._checks127,video=video,audio=audio,
                full50_main_layers_executed=True)
            self._request_steps127+=1;self._total_steps127+=1
            return result
    pipeline.MiniMaxH3DiTModel=OriginalGuardedDiT127
    pipeline.MiniMaxH3Attention=MiniMaxH3Attention
    pipeline.MiniMaxH3Qwen3VLEncoder=encoder.MiniMaxH3Qwen3VLEncoder
    original_noise=pipeline.MiniMaxH3Pipeline._initial_noise
    def exact_noise(self,*,seed,latent_t,latent_h,latent_w,audio_t):
        assert (seed,latent_t,latent_h,latent_w,audio_t)==(1234,37,48,84,207)
        bundle=torch.load('/tmp/h3_align12301/same_initial_sampling123.pt',map_location='cpu',weights_only=True)
        event('exact_initial_noise_used',source_sha256='8a560c430b3048dd8f4ad5c7ef7d5a603cef7c4f52d6fe56aa4173e42120e3b9',teacher_hidden_used=False)
        return bundle['initial_video_target_rows'].clone(),bundle['initial_audio_rows'].clone()
    pipeline.MiniMaxH3Pipeline._initial_noise=exact_noise
    original_build=pipeline.MiniMaxH3Pipeline._build_denoise_inputs
    def build(self,*args,**kwargs):
        result=original_build(self,*args,**kwargs)
        bundle=torch.load('/tmp/h3_align12301/same_initial_sampling123.pt',map_location='cpu',weights_only=True)
        for name in ('sigmas_video','sigmas_audio'):
            assert len(result[name])==50
            assert torch.equal(torch.tensor(result[name],dtype=torch.float32).view(torch.uint8),bundle[name].view(torch.uint8))
        event('denoise_inputs_ready',sigma_points=50,packed_rows=result['branch'].seq_len,own_NPU_conditioning=True)
        return result
    pipeline.MiniMaxH3Pipeline._build_denoise_inputs=build
    original_encode=encoder.MiniMaxH3Qwen3VLEncoder._encode
    def encode(self,*args,**kwargs):
        result=original_encode(self,*args,**kwargs)
        assert result.dtype==torch.float16
        info=stats(result);assert info['finite'];event('live_NPU_encoder_hidden',**info)
        if getattr(self.encoder_group,'rank_in_group',0)==0:
            cpu=result.detach().cpu().contiguous()
            path=request_directory()/f'live_encoder_pid{os.getpid()}127.pt'
            with path.open('xb') as f:torch.save({'hidden_states':cpu},f)
            event('live_NPU_encoder_hidden_saved',path=str(path),sha256_raw=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
        return result
    encoder.MiniMaxH3Qwen3VLEncoder._encode=encode
    original_decode=pipeline.MiniMaxH3Pipeline.decode
    def decode(self,video_latent,audio_latent,**kwargs):
        assert self.transformer._request_steps127==49
        counts={}
        for label in ('video_vae','audio_vae'):
            component=getattr(self,label)
            assert all(p.dtype==torch.float32 for p in component.parameters() if p.is_floating_point())
            counts[label]=sum(p.numel() for p in component.parameters())
        if torch.distributed.get_rank()==0:
            path=request_directory()/f'final_latents_pid{os.getpid()}127.pt'
            vc=video_latent.detach().cpu().contiguous();ac=audio_latent.detach().cpu().contiguous()
            assert torch.isfinite(vc).all() and torch.isfinite(ac).all()
            with path.open('xb') as f:torch.save({'video_latent':vc,'audio_latent':ac,'actual_DiT_evaluations':49},f)
            event('final_NPU_latents_saved',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        event('native_NPU_VAE_decode_begin',original_FP32_parameter_counts=counts)
        start=time.perf_counter()
        with half_nan_scope(enabled=True) as counter:result=original_decode(self,video_latent,audio_latent,**kwargs)
        event('native_NPU_VAE_decode_end',elapsed_s=time.perf_counter()-start,exact_nan_compat_calls=counter['calls'],operand_cache_enabled=False)
        return result
    pipeline.MiniMaxH3Pipeline.decode=decode
    from diagnostic_rpc135 import install_rpc
    install_rpc()
    observe.install(torch)
    if os.environ.get('H3_R127_INIT_DIAGNOSTICS')=='1':
        from init_diagnostics127 import install as install_init_diagnostics
        install_init_diagnostics()
    # Stage host timings and consumer-stream events; drain only after a whole request.
    def stage(cls,name,label):
        original=getattr(cls,name)
        @functools.wraps(original)
        def run(self,*args,**kwargs):
            if not request_index():return original(self,*args,**kwargs)
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record();start=time.perf_counter();facts=observe.stream_facts(torch)
            event('stage_begin',stage=label,streams=facts,memory=observe.memory(torch))
            try:return original(self,*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-start;end.record()
                event('stage_end',stage=label,host_elapsed_s=elapsed,memory=observe.memory(torch))
                observe._DEVICE_EVENTS.append(({'stage':label},begin,end))
        setattr(cls,name,run)
    for name in ('_prepare_request_inputs','encode_prompt','diffuse','decode'):
        stage(pipeline.MiniMaxH3Pipeline,name,'pipeline.'+name)
    stage(vae.MiniMaxH3VideoVAE,'decode_latent','video_vae.decode_latent')
    stage(vae.MiniMaxH3AudioVAE,'decode_latent','audio_vae.decode_latent')
    original_forward=pipeline.MiniMaxH3Pipeline.forward
    def forward(self,request):
        index=getattr(self,'_runtime_request127',0)+1;assert index<=3
        self._runtime_request127=index;self.transformer._request_steps127=0
        begin_request(index);observe.reset_request_observations()
        before_checks=self.transformer._checks127
        torch.npu.reset_peak_memory_stats()
        event('native_request_begin',index=index,streams=observe.stream_facts(torch),
              pipeline_id=id(self),transformer_id=id(self.transformer),state=observe.describe_pipeline(self,torch,numa=True))
        flush() # Small metadata checkpoint outside whole-step timing.
        start=time.perf_counter();success=False
        try:
            result=original_forward(self,request)
            torch.npu.synchronize() # Request boundary only, after all native work.
            assert self.transformer._request_steps127==49 and self.transformer._checks127-before_checks==49*155
            event('native_request_end',host_elapsed_s=time.perf_counter()-start,
                actual_steps=self.transformer._request_steps127,total_steps=self.transformer._total_steps127,
                checks_this_request=self.transformer._checks127-before_checks,
                state=observe.describe_pipeline(self,torch,numa=True),prefetch=observe.prefetch_summary(),
                stage_device_events=observe.drain_device_events(),native_outputs_returned_unchanged=True)
            success=True
            return result
        except BaseException as exc:
            event('native_request_failure',type=type(exc).__name__,message=str(exc));raise
        finally:end_request(success=success)
    pipeline.MiniMaxH3Pipeline.forward=forward
    event('R127_original_math_installed',safety_original_R124=True,strict_checks=155,
        operand_cache_enabled=False,geometry_cache_enabled=False,range_selector_candidate_enabled=False,
        shared_stream_changed=False,model_arithmetic_changed=False)
    flush()
