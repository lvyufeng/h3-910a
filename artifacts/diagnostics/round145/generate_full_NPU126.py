#!/usr/bin/env python3
"""Full original-checkpoint H3 generation through isolated vLLM-Omni on owned8x910A.

Regression of the accepted R124 + FA baseline after R126 exact operator gates; not a new precision acceptance. Live
NPU encoder and original FP32 native VAEs, same seed/noise/sigmas/keyframe PNGs,
FP16 matrices with tested FP32 range protections. No GPU encoder/keyframe-latent
teacher forcing, no CPU inference, no student/LoRA/BF16 fallback or clipping.
This is generation and quality validation, not the encoder/VAE/DiT speed target.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_hook145')
FIXTURES=Path('/tmp/h3_align12401')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-label',default='NPU_full_generation126')
    args=parser.parse_args()
    out=ROOT/args.output_label
    assert out.resolve().parent==ROOT and out.name.startswith('NPU_full_generation126')
    out.mkdir(exist_ok=False)
    os.environ['H3_R126_GENERATION_OUT']=str(out)
    from full_bootstrap126 import install_alignment
    install_alignment()
    import numpy as np
    import torch
    import torch_npu
    import av
    import soundfile as sf
    from PIL import Image,ImageDraw
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline
    from media_helpers124 import find_frames,find_audio,export_av,validate_av
    assert pipeline.MiniMaxH3DiTModel.__name__=='MatchedRangeSafeDiT124'
    from vllm_omni.diffusion.executor.multiproc_executor import _DLO_DP_WAVE_TIMEOUT_S
    assert _DLO_DP_WAVE_TIMEOUT_S>=7200
    nan_probe=json.loads((FIXTURES/'Half_nan_to_num_equivalent_probe124k.json').read_text())
    assert nan_probe['completed'] and len(nan_probe['checks'])==24
    assert all(x['bit_identical_to_native_CPU_operator'] for x in nan_probe['checks'])
    import pickle
    from capture_NPU_media126 import capture_NPU_post_process126
    assert pickle.loads(pickle.dumps(capture_NPU_post_process126)) is capture_NPU_post_process126
    gate=json.loads((FIXTURES/'NPU_sigma_checks124c/sigma_checks_rank0.json').read_text())
    assert gate['completed'] and gate['all_selected_sigma_gates_accepted']
    assert {row['step_index'] for row in gate['checks']}=={0,10,25,40,48}
    assert all(row['video_target_velocity']['accepted_under_1percent_gate'] and row['audio_velocity']['accepted_under_1percent_gate'] for row in gate['checks'])
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    result={'scope':__doc__,'completed':False,'actual_video_generated':False,'all_sigma_checks_passed':True,
            'own_NPU_encoder_used':True,'GPU_encoder_or_keyframe_latent_cache_used':False,'CPU_model_inference':False,
            'no_local_model_weights':True,'model_repo':'MiniMaxAI/MiniMax-H3','revision':'0919fb684102ef115126612f3fbb84d1027e7317',
            'checkpoint':'/tmp/h3_align12401/original_metadata124f/FL2VA','original_weights_checkpoint':'/home/mseco/models/MiniMax-H3/FL2VA','original_official_metadata_view_used':True,'resident_NPU_metadata_and_weights_unchanged':True,'seed':1234,'height':768,'width':1344,'requested_frames':120,'aligned_frames':124,
            'sigma_points':50,'actual_DiT_evaluations_expected':49,'task':'fl2va','native_audio':True,
            'precision':'FP16 matrix operands/weights; original FP32 front/time/final/VAE exceptions and FP32 residual/norm/wide-GEMM output',
            'no_BF16_roundtrip_fallback_or_clipping':True,'quality_accepted':False,'speed_targets_accepted':False,'CANN_optional_knowledge_bank_disabled':False,'private_IPC_authentication_preserved':True,'CANN_helpers_model_bootstrap_skipped':True,'generation_start_unix':time.time(),'actual_executor_rpc_timeout_s':_DLO_DP_WAVE_TIMEOUT_S,'native_NanToNum_compatibility_preflight_passed':True,'original_VAE_FP32_parameters_preserved':True,'quality_mode':'lossless','no_cache_dit_model_output_reuse':True}
    receipt=out/'generation_result126.json'
    def publish():receipt.write_text(json.dumps(result,indent=2)+'\n')
    publish();omni=None
    try:
        images=[]
        for name,x in [('first',320),('last',1024)]:
            image=Image.new('RGB',(1344,768),(200,205,212));draw=ImageDraw.Draw(image)
            draw.rectangle((0,640,1344,768),fill=(98,102,108));draw.ellipse((x-45,550,x+45,640),fill=(215,30,30))
            path=out/f'{name}.png';image.save(path);images.append(image)
            result[name+'_image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
                'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
                'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
                'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
        result['engine_kwargs']=kwargs;publish();started=time.perf_counter()
        omni=Omni(model=result['checkpoint'],**kwargs)
        result['engine_initialization_s']=time.perf_counter()-started;publish()
        sampling=OmniDiffusionSamplingParams(seed=1234,height=768,width=1344,num_frames=120,fps=24,frame_rate=24,
                    num_inference_steps=49,quality='lossless',output_type='np',return_frames=True,save_output=False,
                    extra_args={'task':'fl2va','duration_seconds':5,'audio_mode':'native','frame_indices':[0,-1]})
        request={'prompt':'a red ball bouncing from left to right on the floor','multi_modal_data':{'image':images}}
        started=time.perf_counter();outputs=omni.generate(request,sampling_params_list=sampling,use_tqdm=False)
        result['generation_wall_s']=time.perf_counter()-started
        assert len(outputs)==1 and outputs[0].finished and not outputs[0].error
        output=outputs[0];frames=find_frames(output.multimodal_output)
        if frames is None:frames=find_frames(output.images)
        audio=find_audio(output.multimodal_output)
        assert frames is not None and audio is not None
        assert frames.shape==(124,768,1344,3),frames.shape
        if frames.dtype!=np.uint8:
            assert np.isfinite(frames).all() and frames.min()>=0 and frames.max()<=1
            frames=np.rint(frames*255).astype(np.uint8)
        assert np.isfinite(audio).all() and audio.shape[0]==2
        audit=[json.loads(line) for p in out.glob('worker_alignment_pid*126.jsonl') for line in p.read_text().splitlines()]
        audit=[row for row in audit if row['unix']>=result['generation_start_unix']]
        ends=[row for row in audit if row['event']=='dit_step_end']
        pids={row['pid'] for row in ends}
        counts={str(pid):sum(row['pid']==pid for row in ends) for pid in pids}
        assert len(counts)==8 and set(counts.values())=={49},counts
        assert all(row['video']['finite'] and row['audio']['finite'] for row in ends)
        assert any(row['event']=='live_NPU_encoder_hidden_saved' for row in audit)
        result['actual_DiT_counts_by_worker_pid']=counts
        assert all(row['actual_Half_matrix_modules_this_step']==155 for row in ends)
        cache_events=[row for row in audit if row['event']=='R126_VAE_operand_cache_end']
        assert len(cache_events)==8 and all(row['cache']['cached_operand_call_count']>0 and row['cache']['original_parameters_unchanged'] for row in cache_events)
        result['VAE_operand_cache_runtime_events']=cache_events
        result['equivalent_strict_safety_all_8x49x155_checks_passed']=True
        result['new_combined_configuration_quality_accepted']=False
        video_path=out/'h3_910a_vllm_omni_fp16.mp4';export_av(frames,audio,video_path)
        wav_path=out/'h3_910a_vllm_omni_audio.wav';sf.write(str(wav_path),audio.T,32000,subtype='FLOAT')
        result['video_sha256']=hashlib.sha256(video_path.read_bytes()).hexdigest();result['audio_sha256']=hashlib.sha256(wav_path.read_bytes()).hexdigest()
        result['video_bytes']=video_path.stat().st_size;result['audio_bytes']=wav_path.stat().st_size
        result['media_verification']=validate_av(video_path)
        result['native_audio_peak']=float(np.abs(audio).max());result['native_audio_RMS']=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
        selected=[0,12,24,36,48,62,74,86,98,110,123]
        thumb_w,thumb_h=336,192
        sheet=Image.new('RGB',(4*thumb_w,3*(thumb_h+24)),(245,245,245));drawer=ImageDraw.Draw(sheet)
        for j,i in enumerate(selected):
            image=Image.fromarray(frames[i]);image.save(out/f'frame_{i:03d}.png')
            image=image.resize((thumb_w,thumb_h));x=(j%4)*thumb_w;y=(j//4)*(thumb_h+24);sheet.paste(image,(x,y));drawer.text((x+6,y+thumb_h+4),f'frame{i:03d}  {i/24:.2f}s',fill=(20,20,20))
        sheet.save(out/'contact_sheet.png')
        result['completed']=True;result['actual_video_generated']=True;publish()
        print(json.dumps({'NPU_full_generation_completed':True,'actual_DiT_steps_per_worker':49,'frames':124,'output_dir':str(out),'quality_acceptance_pending_visual_review':True}),flush=True)
    except BaseException as exc:result['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if omni is not None:omni.close()


if __name__=='__main__':main()
