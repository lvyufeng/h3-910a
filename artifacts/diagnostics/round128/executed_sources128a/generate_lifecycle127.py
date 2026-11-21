#!/usr/bin/env python3
"""R128 formal worker native-input A/B, then original49 baseline/candidate/restored.

Replay never supplies sampler outputs and its check/time scope is separate. Full
candidate generation is gated on exact leaves, real owner-stream support and a
clear >=1% gain. Encoder/FP32-VAE/TP/HCCL/NUMA/global shim remain native.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_overlap12801')
SESSION=ROOT/'engine_overlap128a'
LABELS={1:'r01_cold',2:'r02_warm',3:'r03_native_reload'}


def main():
    assert SESSION.is_dir() and not (SESSION/'lifecycle_result127.json').exists()
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
    from media_helpers124 import find_frames,find_audio,export_av,validate_av
    import av
    for name in ('libx264','aac'):av.codec.Codec(name,'w')
    from vllm_omni.diffusion.executor.multiproc_executor import _DLO_DP_WAVE_TIMEOUT_S
    assert _DLO_DP_WAVE_TIMEOUT_S>=7200
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
        'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
        'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
        'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
    report={'scope':__doc__,'completed':False,'requests':[],'engine_kwargs':kwargs,
        'model_revision':'0919fb684102ef115126612f3fbb84d1027e7317','checkpoint':'/tmp/h3_align12401/original_metadata124f/FL2VA',
        'model_weights':'/home/mseco/models/MiniMax-H3/FL2VA','single_Omni_instance':True,'original_R124_safety_predicates':True,
        'new_precision_policy_accepted':False,'shared_runtime_changed':False,'no_local_model_weights':True}
    receipt=SESSION/'lifecycle_result127.json'
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    publish();omni=None
    def audit(index):
        return [json.loads(line) for path in SESSION.glob('worker_runtime_pid*127.jsonl')
                for line in path.read_text().splitlines() if json.loads(line)['request_index']==index]
    try:
        start=time.perf_counter();omni=Omni(model=report['checkpoint'],**kwargs)
        report['engine_initialization_s']=time.perf_counter()-start;publish()
        first_pids=None;pipeline_ids=None
        for index,label in LABELS.items():
            out=SESSION/label;assert out.is_dir()
            images=[];image_shas={}
            for name,x in [('first',320),('last',1024)]:
                image=Image.new('RGB',(1344,768),(200,205,212));draw=ImageDraw.Draw(image)
                draw.rectangle((0,640,1344,768),fill=(98,102,108));draw.ellipse((x-45,550,x+45,640),fill=(215,30,30))
                path=out/(name+'.png');image.save(path);images.append(image)
                image_shas[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            sampling=OmniDiffusionSamplingParams(seed=1234,height=768,width=1344,num_frames=120,fps=24,frame_rate=24,
                num_inference_steps=49,quality='lossless',output_type='np',return_frames=True,save_output=False,
                extra_args={'task':'fl2va','duration_seconds':5,'audio_mode':'native','frame_indices':[0,-1]})
            request={'prompt':'a red ball bouncing from left to right on the floor','multi_modal_data':{'image':images}}
            start=time.perf_counter();outputs=omni.generate(request,sampling_params_list=sampling,use_tqdm=False)
            wall=time.perf_counter()-start
            assert len(outputs)==1 and outputs[0].finished and not outputs[0].error
            frames=find_frames(outputs[0].multimodal_output)
            if frames is None:frames=find_frames(outputs[0].images)
            audio=find_audio(outputs[0].multimodal_output)
            assert frames is not None and audio is not None and frames.shape==(124,768,1344,3)
            assert np.isfinite(audio).all() and audio.shape==(2,165600)
            assert frames.dtype==np.uint8,'Native pipeline should already quantize; do not alter generated pixels'
            from audit_completion127 import wait_complete
            completion=wait_complete(out,index,timeout_s=120)
            rows=audit(index);ends=[e for e in rows if e['event']=='dit_step_end']
            pids={e['pid'] for e in ends};assert len(pids)==8 and sorted(pids)==completion['pids']
            assert all(sorted(e['step_index'] for e in ends if e['pid']==pid)==list(range(49)) for pid in pids)
            assert all(e['actual_Half_matrix_modules_this_step']==155 for e in ends)
            assert sum(e['actual_Half_matrix_modules_this_step'] for e in ends)==60760
            assert len([e for e in rows if e['event']=='live_NPU_encoder_hidden'])==8
            assert any(e['event']=='live_NPU_encoder_hidden_saved' for e in rows)
            request_ends=[e for e in rows if e['event']=='native_request_end'];assert len(request_ends)==8
            starts=[e for e in rows if e['event']=='native_request_begin'];assert len(starts)==8
            ids={e['pid']:(e['pipeline_id'],e['transformer_id']) for e in starts}
            if first_pids is None:first_pids=pids;pipeline_ids=ids
            else:assert pids==first_pids and ids==pipeline_ids,'Engine worker/models were reconstructed'
            video_path=out/'h3_910a_original127.mp4';wav_path=out/'h3_910a_native127.wav'
            export_av(frames,audio,video_path);sf.write(str(wav_path),audio.T,32000,subtype='FLOAT')
            row={'index':index,'label':label,'generation_wall_s':wall,'worker_audit_wait_s_outside_generation':completion['audit_wait_s'],'worker_pids':sorted(pids),
                'worker_pipeline_ids':ids,'strict_Half_checks_passed':60760,'steps_per_rank':49,'images_sha256':image_shas,
                'video_sha256':hashlib.sha256(video_path.read_bytes()).hexdigest(),
                'audio_sha256':hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                'native_audio_RMS':float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                'media_verification':validate_av(video_path),'original_live_encoder_executed':True,
                'new_operand_cache_enabled':False,'fresh_worker_processes':index==1,
                'native_stage_reload_confirmed':any(e['event']=='stager_transition' and e['method']=='_load_once' for e in rows),
                'only_DiT50_native_copy_candidate':index==2,'wall_includes_nonsampling_same_input_replay':index==1,
                'third_request_does_not_clear_CANN_compilation_cache':index==3}
            report['requests'].append(row);publish()
            print(json.dumps({'request_completed':label,'generation_wall_s':wall,'strict_checks':60760,'same_workers':True}),flush=True)
            del outputs,frames,audio,images,request,sampling;gc.collect()
            if index==1:
                gate_files=list(ROOT.glob('formal_replay_rank*128.json'));assert len(gate_files)==8
                gates=[json.loads(path.read_text()) for path in gate_files]
                assert sorted(r['rank'] for r in gates)==list(range(8)) and all(r['completed'] and r['restored'] for r in gates)
                report['formal_replay_gate_results']=[{'rank':r['rank'],'result':r['result'],'allowed':r['candidate_formal_generation_allowed'],
                    'all_bit_gates':r['all8ranks_bit_gates_passed'],'gain_fraction':r['candidate_gain_vs_restored_fraction']} for r in gates]
                publish()
                if not all(r['candidate_formal_generation_allowed'] for r in gates):
                    report['candidate_49_generation_not_launched']='Same-input bit/support/speed gate declined candidate; native generation preserved'
                    break
        assert len(report['requests']) in (1,3)
        report['completed']=True;report['actual_full_49_step_requests']=len(report['requests'])
        report['total_strict_checks_passed']=60760*len(report['requests']);report['worker_group_reused']=True;publish()
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if omni is not None:omni.close()


if __name__=='__main__':main()
