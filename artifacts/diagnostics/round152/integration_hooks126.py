#!/usr/bin/env python3
"""Worker-local NPU H3 full-generation precision integration and honest audit.

Original native checkpoint, live NPU encoder and VAEs. Exact GPU-exported initial
noise only is reused; no GPU hidden-state/keyframe-activation cache substitutes
NPU model computation. The verified GPU arithmetic transformer is injected into
the current NPU pipeline behind a single-native-request compatibility adapter.
No student, LoRA, VSA, BF16 fallback, clipping, fake norm or shared source edits.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_align12401')
GENERATION_ROOT=Path(os.environ['H3_R126_GENERATION_OUT']).resolve()
assert GENERATION_ROOT.parent==Path('/tmp/h3_silu152') and GENERATION_ROOT.name.startswith('NPU_full_generation126')
assert GENERATION_ROOT.is_dir()


def install():
    import torch
    import torch_npu
    torch.npu.matmul.allow_hf32=False
    torch.npu.conv.allow_hf32=False
    from h3_replay123.transformer import MiniMaxH3DiTModel as VerifiedDiT,MiniMaxH3Attention
    from fp32_patch_residual_guard123 import install_patch_projection_guard123
    from fp32_time_residual_guard123 import install_time_projection_guard123
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline
    import vllm_omni.diffusion.models.minimax_h3.vae as vae_module
    from vllm.utils.torch_utils import set_default_torch_dtype
    original_remote_factory=vae_module._load_remote_component
    def original_FP32_VAE_factory(*args,**kwargs):
        with set_default_torch_dtype(torch.float32):
            model=original_remote_factory(*args,**kwargs)
        assert all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point())
        return model
    vae_module._load_remote_component=original_FP32_VAE_factory
    manifest=json.loads((ROOT/'isolated_integration_sources124.json').read_text())
    assert manifest['completed']
    encoder_path=ROOT/'NPU_encoder_FP16_124.py'
    spec=importlib.util.spec_from_file_location('h3_NPU_live_encoder124',encoder_path)
    encoder_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder_module)
    audit_path=GENERATION_ROOT/f'worker_alignment_pid{os.getpid()}126.jsonl'
    def event(name,**data):
        row={'event':name,'pid':os.getpid(),'rank':int(os.environ.get('LOCAL_RANK',os.environ.get('RANK','-1'))),'unix':time.time(),**data}
        with audit_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    if os.environ.get('H3_NPU_VAE_PRESERVE_FP32_PARAMS124')=='1':
        # The current native installer rewrites decoder Linear parameters to
        # Half. Keep the original FP32 parameter tensors as on the successful
        # GPU path; decode's normal autocast still supplies FP16 matrix work.
        def preserve_vae_parameters(decoder,*,device):
            assert all(p.dtype==torch.float32 for p in decoder.parameters() if p.is_floating_point())
            event('native_VAE_FP32_parameter_rewrite_disabled',GPU_autocast_matrix_contract_preserved=True)
            return False
        vae_module.install_h3_vae_optimizations=preserve_vae_parameters
    if os.environ.get('H3_NPU_CAPTURE_NATIVE_MEDIA124')=='1':
        from capture_NPU_media126 import install_capture
        install_capture(pipeline)
    def stats(x):
        x=x.detach()
        # Preserve raw bits during transfer before CPU finiteness checks.
        # Legacy910A float predicates failed the explicit NaN/Inf regression.
        cpu=x.cpu().float()
        return {'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'finite':bool(torch.isfinite(cpu).all()),
                'max_abs':float(cpu.abs().amax()),'finiteness_checked_on_CPU_activation_copy_not_CPU_model_inference':True}
    class MatchedRangeSafeDiT124(VerifiedDiT):
        def __init__(self,od_config,quant_config=None,*,diffusers_weights=False):
            if diffusers_weights or quant_config is not None:
                raise ValueError('Alignment supports only original native unquantized H3, not modular/student/LoRA checkpoints')
            super().__init__(od_config,quant_config=None)
            self._actual_step124=0
            self.diffusers_weights=False
            self._precision_modules124=install_patch_projection_guard123(self)+install_time_projection_guard123(self)
            self._range_checked_half_matrix_outputs124=0
            for name,module in self.named_modules():
                if not (name.endswith('.qkv_proj') or name.endswith('.mlp.fc1') or name.endswith('.adaln_proj.linear')):
                    continue
                def verify_half_matrix_output(layer,inputs,result,name=name):
                    tensor=result[0] if isinstance(result,tuple) else result
                    if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float16:return
                    self._range_checked_half_matrix_outputs124+=1
                    from equivalent_safety126 import half_safety_counts
                    nonfinite,boundary=half_safety_counts(tensor,fast=True)
                    nonfinite_count=int(nonfinite)
                    finite=nonfinite_count==0
                    boundary_count=int(boundary)
                    if not finite or boundary_count:
                        event('half_matrix_range_warning',module=name,finite=finite,half_limit_value_count=boundary_count)
                        raise FloatingPointError('Half matrix output reached nonfinite or saturation boundary at '+name)
                module.register_forward_hook(verify_half_matrix_output)
            event('matched_transformer_constructed',source_sha256=manifest['source_digests']['transformer.py'],precision_modules=self._precision_modules124,
                  parameter_dtype_counts={str(dtype):sum(p.numel() for p in self.parameters() if p.dtype==dtype) for dtype in {p.dtype for p in self.parameters()}},
                  main_layers=len(self.blocks),hidden_conditioning_cache=False)
        def forward(self,**kwargs):
            # Current pipeline adds geometry-only metadata that the pinned
            # verified single-document reference derives from the original
            # packed indices. It does not select VSA or alter valid-key masks.
            kwargs.pop('video_token_layout',None)
            if int(kwargs.get('packed_seq_params',{}).get('num_requests',1))!=1:
                raise ValueError('Alignment compatibility adapter does not allow co-batched/multi-document requests')
            step=self._actual_step124
            started=time.perf_counter()
            before_checks=self._range_checked_half_matrix_outputs124
            event('dit_step_begin',step_index=step,unique_timesteps=kwargs['unique_timesteps'].detach().cpu().tolist())
            vv,av=super().forward(**kwargs)
            actual_checks=self._range_checked_half_matrix_outputs124-before_checks
            assert actual_checks==155,('Full original50-layer forward did not execute all expected matrix modules',actual_checks)
            video=stats(vv);audio=stats(av)
            if not video['finite'] or not audio['finite']:raise FloatingPointError('Nonfinite actual NPU prediction')
            event('dit_step_end',step_index=step,video=video,audio=audio,elapsed_s_includes_offload=time.perf_counter()-started,
                  checked_Half_matrix_output_count=self._range_checked_half_matrix_outputs124,no_observed_Half_saturation_boundary=True,
                  actual_Half_matrix_modules_this_step=actual_checks,full50_main_layers_executed=True)
            self._actual_step124+=1
            return vv,av
    pipeline.MiniMaxH3DiTModel=MatchedRangeSafeDiT124
    pipeline.MiniMaxH3Attention=MiniMaxH3Attention
    pipeline.MiniMaxH3Qwen3VLEncoder=encoder_module.MiniMaxH3Qwen3VLEncoder
    native_initial=pipeline.MiniMaxH3Pipeline._initial_noise
    def exact_initial(self,*,seed,latent_t,latent_h,latent_w,audio_t):
        if (seed,latent_t,latent_h,latent_w,audio_t)!=(1234,37,48,84,207):
            raise ValueError('This exact-noise trial is scoped to seed1234 1344x768 124alignedframes')
        bundle=torch.load('/tmp/h3_align12301/same_initial_sampling123.pt',map_location='cpu',weights_only=True)
        event('exact_initial_noise_used',video_shape=list(bundle['initial_video_target_rows'].shape),audio_shape=list(bundle['initial_audio_rows'].shape),
              source_bundle_sha256='8a560c430b3048dd8f4ad5c7ef7d5a603cef7c4f52d6fe56aa4173e42120e3b9',GPU_encoder_activation_used=False,GPU_keyframe_latent_used=False)
        return bundle['initial_video_target_rows'].clone(),bundle['initial_audio_rows'].clone()
    pipeline.MiniMaxH3Pipeline._initial_noise=exact_initial
    original_encode=pipeline.MiniMaxH3Pipeline.encode_prompt
    def encode_prompt(self,*args,**kwargs):
        started=time.perf_counter();output=original_encode(self,*args,**kwargs)
        event('live_NPU_encode_prompt_completed',elapsed_s=time.perf_counter()-started,teacher_forcing_used=False)
        return output
    pipeline.MiniMaxH3Pipeline.encode_prompt=encode_prompt
    original_encoder_call=encoder_module.MiniMaxH3Qwen3VLEncoder._encode
    def live_encode(self,*args,**kwargs):
        result=original_encoder_call(self,*args,**kwargs)
        assert result.dtype==torch.float16
        row=stats(result);assert row['finite']
        event('live_NPU_encoder_hidden',**row)
        if getattr(self.encoder_group,'rank_in_group',0)==0:
            cpu=result.detach().cpu().contiguous()
            path=GENERATION_ROOT/f'live_NPU_encoder_hidden_pid{os.getpid()}126.pt'
            with path.open('xb') as f:torch.save({'hidden_states':cpu},f)
            event('live_NPU_encoder_hidden_saved',path=str(path),sha256_raw=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
        return result
    encoder_module.MiniMaxH3Qwen3VLEncoder._encode=live_encode
    original_build=pipeline.MiniMaxH3Pipeline._build_denoise_inputs
    def build(self,*args,**kwargs):
        output=original_build(self,*args,**kwargs)
        assert len(output['sigmas_video'])==len(output['sigmas_audio'])==50
        bundle=torch.load('/tmp/h3_align12301/same_initial_sampling123.pt',map_location='cpu',weights_only=True)
        for name,key in [('sigmas_video','sigmas_video'),('sigmas_audio','sigmas_audio')]:
            assert torch.equal(torch.tensor(output[name],dtype=torch.float32).view(torch.uint8),bundle[key].view(torch.uint8)),name
        event('denoise_inputs_ready',packed_rows=output['branch'].seq_len,sigma_points=50,expected_DiT_evaluations=49,
              own_NPU_conditioning=True,cond_anchor=stats(output['cond_anchor']) if output['cond_anchor'] is not None else None)
        return output
    pipeline.MiniMaxH3Pipeline._build_denoise_inputs=build
    if os.environ.get('H3_NPU_SAVE_FINAL_LATENTS124')=='1':
        from npu_half_nan_compat124 import half_nan_scope
        def wrap_decode(original):
            def decode_with_capture(self,video_latent,audio_latent,**kwargs):
                assert self.transformer._actual_step124==49
                if torch.distributed.get_rank()==0:
                    path=GENERATION_ROOT/f'final_NPU_latents_pid{os.getpid()}126.pt'
                    vc=video_latent.detach().cpu().contiguous();ac=audio_latent.detach().cpu().contiguous()
                    assert torch.isfinite(vc).all() and torch.isfinite(ac).all()
                    with path.open('xb') as f:torch.save({'video_latent':vc,'audio_latent':ac,'height':kwargs['height'],'width':kwargs['width'],'actual_DiT_evaluations':49,'contains_model_weights':False},f)
                    event('final_NPU_latents_saved',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          video_shape=list(vc.shape),audio_shape=list(ac.shape),actual_DiT_evaluations=49,contains_model_weights=False)
                counts={}
                for label in ['video_vae','audio_vae']:
                    module=getattr(self,label)
                    counts[label]={str(dtype):sum(p.numel() for p in module.parameters() if p.dtype==dtype) for dtype in {p.dtype for p in module.parameters()}}
                    assert all(p.dtype==torch.float32 for p in module.parameters() if p.is_floating_point()),label
                event('native_NPU_VAE_decode_begin',original_FP32_parameter_counts=counts,standard_FP16_video_autocast_retained=True)
                started=time.perf_counter()
                with half_nan_scope(enabled=os.environ.get('H3_NPU_NAN_COMPAT124')=='1') as counter:
                    result=original(self,video_latent,audio_latent,**kwargs)
                event('native_NPU_VAE_decode_end',elapsed_s=time.perf_counter()-started,exact_NanToNum_compatibility_calls=counter['calls'],BF16_fallback=False)
                return result
            return decode_with_capture
        pipeline.MiniMaxH3Pipeline.decode=wrap_decode(pipeline.MiniMaxH3Pipeline.decode)
        pipeline.MiniMaxH3Pipeline.decode_to_mp4=wrap_decode(pipeline.MiniMaxH3Pipeline.decode_to_mp4)
    # Operand copies are experiment-only; parameters/norm/residual/front/final
    # remain exactly the original FP32 tensors. Scope cache to decoder residency.
    if os.environ.get('H3_R126_VAE_OPERAND_CACHE')=='1':
        from autocast_operand_cache126 import AutocastOperandCache126
        native_decode_latent=vae_module.MiniMaxH3VideoVAE.decode_latent
        def decode_with_operand_cache(self,latent):
            assert all(p.dtype==torch.float32 for p in self.parameters() if p.is_floating_point())
            # Called only AFTER the native residency context has staged decoder.
            cache=AutocastOperandCache126(self.model.decoder)
            try:
                started=time.perf_counter();cache.prepare();prime_s=time.perf_counter()-started
                cache.set_enabled(True)
                event('R126_VAE_operand_cache_begin',cache=cache.snapshot(),prime_s=prime_s)
                result=native_decode_latent(self,latent)
                snapshot=cache.snapshot()
                assert snapshot['original_parameters_unchanged']
                event('R126_VAE_operand_cache_end',cache=snapshot,original_FP32_parameters_unchanged=True)
                return result
            finally:
                cache.close()
        vae_module.MiniMaxH3VideoVAE.decode_latent=decode_with_operand_cache
    event('R126_equivalent_safety_enabled',exhaustive_Half_words=65536,original_check_count=155,
          strict_stop_placement_preserved=True,model_arithmetic_modified=False,
          exact_device_scale_enabled=False,TP_residency_changed=False,geometry_cache_enabled=False)
    event('isolated_alignment_bootstrap_installed',source_manifest=manifest,CPU_model_inference=False,BF16_encoder_roundtrip=False,
          own_NPU_encoder_required=True,own_NPU_VAE_required=True,shared_runtime_modified=False)
