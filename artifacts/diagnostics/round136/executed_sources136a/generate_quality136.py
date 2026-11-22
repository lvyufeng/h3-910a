#!/usr/bin/env python3
"""Approved full49 quality test: originalFalse -> globalTrue+AdaLN -> restoredFalse.

Recomputes own live encoder and original FP32 VAE under each pipeline flag. No
teacher tensors, output edits, adoption, tolerance or copy candidate. Arrays and
activation archives stay on the host; only bounded scalar evidence is collected.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_quality13601');SESSION=ROOT/'engine_quality136a'
LABELS={1:'r01_original_false',2:'r02_candidate_true',3:'r03_restored_false'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    assert SESSION.is_dir();receipt=SESSION/'lifecycle_result136.json';assert not receipt.exists()
    approval=json.loads((ROOT/'approval136.json').read_text());assert approval['selected']=='只做完整49步质量对照 (Recommended)' and not approval['numeric_baseline_adoption']
    os.environ['H3_R127_SESSION_OUT']=str(SESSION)
    from full_bootstrap127 import install_alignment
    install_alignment()
    import numpy as np
    import torch
    import torch_npu
    import soundfile as sf
    from PIL import Image,ImageDraw
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from media_helpers124 import export_av,validate_av
    from native_media136 import find_native_frames,find_native_audio
    from audit_completion127 import wait_complete
    from vllm_omni.diffusion.executor.multiproc_executor import _DLO_DP_WAVE_TIMEOUT_S
    assert _DLO_DP_WAVE_TIMEOUT_S>=7200
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
        'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
        'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
        'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
    report={'scope':__doc__,'completed':False,'requests':[],'engine_kwargs':kwargs,'control_RPCs':[],
        'model_revision':'0919fb684102ef115126612f3fbb84d1027e7317','model_weights':'/home/mseco/models/MiniMax-H3/FL2VA',
        'checkpoint':'/tmp/h3_align12401/original_metadata124f/FL2VA','single_Omni_instance':True,'approval':approval,
        'numeric_baseline_adopted':False,'tolerance_adopted':False,'copy_candidate_enabled':False,
        'candidate_single_complete49_request_not_repeatability_proof':True,'activation_and_raw_array_files_host_only':True}
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    def audit(index):
        return [r for path in SESSION.glob('worker_runtime_pid*127.jsonl') for line in path.read_text().splitlines() if (r:=json.loads(line))['request_index']==index]
    omni=None;prepared=False;finished=False;publish()
    def rpc(action,index=0):
        started=time.perf_counter();reply=omni.engine.collective_rpc('r136_original_candidate_quality_control',timeout=2400,args=(action,index),stage_ids=[0])
        report['control_RPCs'].append({'action':action,'index':index,'wall_s_outside_generation':time.perf_counter()-started,'reply':reply});publish()
    try:
        started=time.perf_counter();omni=Omni(model=report['checkpoint'],**kwargs);report['engine_initialization_s']=time.perf_counter()-started;publish()
        rpc('prepare');prepared=True
        workers=[json.loads(p.read_text()) for p in ROOT.glob('quality_worker_rank*136.json')]
        assert len(workers)==8 and sorted(r['rank'] for r in workers)==list(range(8)) and all(r['prepared'] for r in workers)
        expected_pids={r['pid'] for r in workers};assert len(expected_pids)==8;identities=None;first_image_SHA=None
        for index,label in LABELS.items():
            out=SESSION/label;assert out.is_dir();rpc('set',index)
            images=[];image_SHAs={}
            for name,x in (('first',320),('last',1024)):
                image=Image.new('RGB',(1344,768),(200,205,212));draw=ImageDraw.Draw(image)
                draw.rectangle((0,640,1344,768),fill=(98,102,108));draw.ellipse((x-45,550,x+45,640),fill=(215,30,30))
                image_path=out/(name+'.png');assert not image_path.exists();image.save(image_path);images.append(image);image_SHAs[name]=sha(image_path)
            if first_image_SHA is None:first_image_SHA=image_SHAs
            else:assert image_SHAs==first_image_SHA
            sampling=OmniDiffusionSamplingParams(seed=1234,height=768,width=1344,num_frames=120,fps=24,frame_rate=24,
                num_inference_steps=49,quality='lossless',output_type='np',return_frames=True,save_output=False,
                extra_args={'task':'fl2va','duration_seconds':5,'audio_mode':'native','frame_indices':[0,-1]})
            request={'prompt':'a red ball bouncing from left to right on the floor','multi_modal_data':{'image':images}}
            started=time.perf_counter();outputs=omni.generate(request,sampling_params_list=sampling,use_tqdm=False);wall=time.perf_counter()-started
            assert len(outputs)==1 and outputs[0].finished and not outputs[0].error
            frames=find_native_frames(outputs[0].multimodal_output,torch,np)
            if frames is None:frames=find_native_frames(outputs[0].images,torch,np)
            audio=find_native_audio(outputs[0].multimodal_output,torch,np)
            assert frames is not None and audio is not None and frames.shape==(124,768,1344,3) and frames.dtype==np.uint8
            assert audio.shape==(2,165600) and audio.dtype==np.float32 and bool(np.isfinite(audio).all())
            completion=wait_complete(out,index,timeout_s=120);rows=audit(index)
            ends=[r for r in rows if r['event']=='dit_step_end'];pids={r['pid'] for r in ends};assert pids==expected_pids
            assert len(ends)==49*8 and all(sorted(r['step_index'] for r in ends if r['pid']==pid)==list(range(49)) for pid in pids)
            assert all(r['actual_Half_matrix_modules_this_step']==155 and r['actual_FA_launches']==50 and r['actual_FA_errors']==0 for r in ends)
            assert sum(r['actual_Half_matrix_modules_this_step'] for r in ends)==60760
            assert all(r['actual_deterministic_flag']==(index==2) and r['actual_candidate_calls_this_step']==int(index==2) for r in ends)
            encoder_events=[r for r in rows if r['event']=='live_NPU_encoder_hidden'];assert len(encoder_events)==8 and {r['pid'] for r in encoder_events}==expected_pids
            assert all(r['actual_encoder_TP_size']==8 and r['encoder_group_ranks']==list(range(8)) for r in encoder_events)
            assert sorted(r['encoder_vocab_partition_rank'] for r in encoder_events)==list(range(8))
            inputs=[r for r in rows if r['event']=='denoise_inputs_ready'];assert len(inputs)==8 and all(r['sigma_points']==50 for r in inputs)
            starts=[r for r in rows if r['event']=='native_request_begin'];request_ends=[r for r in rows if r['event']=='native_request_end']
            assert len(starts)==len(request_ends)==8 and all(r['actual_deterministic_flag']==(index==2) for r in starts+request_ends)
            ids={r['pid']:(r['pipeline_id'],r['transformer_id']) for r in starts}
            if identities is None:identities=ids
            else:assert ids==identities
            frame_path=out/'nativeframes136.npy';audio_path=out/'audio136.npy'
            with frame_path.open('xb') as file:np.save(file,frames,allow_pickle=False)
            with audio_path.open('xb') as file:np.save(file,audio,allow_pickle=False)
            latents=list(out.glob('final_latents_pid*127.pt'));encoded=list(out.glob('live_encoder_pid*127.pt'));assert len(latents)==len(encoded)==1
            video_path=out/'h3_quality136.mp4';wav_path=out/'h3_quality136.wav';assert not video_path.exists() and not wav_path.exists()
            export_av(frames,audio,video_path);sf.write(str(wav_path),audio.T,32000,subtype='FLOAT')
            row={'completed':True,'index':index,'label':label,'actual_deterministic_flag':index==2,
                'generation_wall_s':wall,'worker_audit_wait_s_outside_generation':completion['audit_wait_s'],'worker_pids':sorted(pids),
                'worker_pipeline_ids':ids,'actual_DiT_evaluations':49,'sigma_boundaries':50,'strict_checks_allranks':60760,
                'actual_FA_launches':19600,'actual_FA_errors':0,'actual_candidate_calls':392 if index==2 else 0,
                'images_sha256':image_SHAs,'rawframes_sha256':sha(frame_path),'rawaudio_sha256':sha(audio_path),
                'final_latents_sha256':sha(latents[0]),'encoder_hidden_file_sha256':sha(encoded[0]),
                'native_encoder_logical_SHA':[r['sha256_raw'] for r in rows if r['event']=='live_NPU_encoder_hidden_saved'],
                'native_video_shape':list(frames.shape),'native_video_dtype':str(frames.dtype),'native_audio_shape':list(audio.shape),'native_audio_dtype':str(audio.dtype),
                'video_sha256':sha(video_path),'audio_sha256':sha(wav_path),'media_verification':validate_av(video_path),
                'native_stage_reload_confirmed':any(r['event']=='stager_transition' and r['method']=='_load_once' for r in rows),
                'raw_arrays_saved_outside_generation_timer':True,'candidate_changes_global_pipeline_flag_not_only_DiT':index==2,
                'preview_video_codec_lossy_x264_CRF18_yuv420p':True,'audio_metrics_use_native_NPY_not_AAC':True,
                'media_export_and_validation_outside_generation_timer':True}
            report['requests'].append(row);publish();print(json.dumps({'request_completed':label,'generation_wall_s':wall,'checks':60760}),flush=True)
            del outputs,frames,audio,images,request,sampling,rows,ends;gc.collect()
        rpc('finish');finished=True
        worker_reports=[json.loads(p.read_text()) for p in ROOT.glob('quality_worker_rank*136.json')]
        assert len(worker_reports)==8 and all(r['completed'] and r['candidate_method_restored'] and not r['final_runtime']['deterministic'] and not r['final_runtime']['warn_only'] for r in worker_reports)
        assert sum(r['strict_Half_checks'] for r in worker_reports)==182280 and sum(r['actual_candidate_calls'] for r in worker_reports)==392
        report.update(completed=True,strict_checks_allranks=182280,actual_FA_launches=58800,actual_FA_errors=0,
            actual_candidate_calls=392,actual_full49_requests=3,original_methods_and_false_warn_flags_restored=True)
        publish()
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)}
        try:publish()
        except BaseException:pass
        raise
    finally:
        if omni is not None:
            if prepared and not finished and not (ROOT/'cohort_failure128.json').exists():
                try:rpc('abort_restore')
                except BaseException as exc:
                    report['failed_request_restore_RPC_error']={'type':type(exc).__name__,'message':str(exc)}
                    try:publish()
                    except BaseException:pass
            omni.close()


if __name__=='__main__':main()
