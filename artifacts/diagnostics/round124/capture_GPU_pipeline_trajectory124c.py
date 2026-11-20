#!/usr/bin/env python3
"""Use the known-successful vLLM-Omni GPU execution path to capture a real49-step trajectory.

Native original checkpoint, live GPU encoder and tiled FP32 keyframe VAE. Five
actual forward-input/output snapshots, no CPU model baseline, no local weights.
The run intentionally ends before output-VAE decoding after49 complete forwards;
it is not presented as a new successful generated video.
"""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/home/mseco/h3_align12401/pipeline124c')


def main():
    import torch
    from PIL import Image,ImageDraw
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    transformer=Path('/home/mseco/h3_gpu12201/vllm-omni-fl2va-release/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py')
    assert hashlib.sha256(transformer.read_bytes()).hexdigest()=='7cf0c53e6d9199b7944d9026c2bdc1120b1e32084cffe266c434d2d6449b8029'
    assert list(ROOT.glob('GPU_bootstrap_pid*124.json'))
    receipt=ROOT/'GPU_pipeline_trajectory_result124c.json';assert not receipt.exists()
    report={'scope':__doc__,'completed':False,'known_successful_vllm_omni_commit':'1c2a81f6d84aea4fff53bd2f894c2a287c237245',
            'checkpoint':'/home/mseco/h3_gpu12201/checkpoint/FL2VA','revision':'0919fb684102ef115126612f3fbb84d1027e7317',
            'live_GPU_encoder_and_tiled_VAE':True,'new_video_generated':False,'decoder_intentionally_skipped':True,'no_CPU_model_inference':True,
            'no_local_model_weights':True,'expected_actual_DiT_evaluations':49,'snapshots':[0,10,25,40,48]}
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    publish();omni=None
    try:
        images=[]
        for name,x in [('first',320),('last',1024)]:
            image=Image.new('RGB',(1344,768),(200,205,212));draw=ImageDraw.Draw(image)
            draw.rectangle((0,640,1344,768),fill=(98,102,108));draw.ellipse((x-45,550,x+45,640),fill=(215,30,30))
            path=ROOT/f'{name}.png';image.save(path);images.append(image)
            report[name+'_image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
                'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
                'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
                'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
        started=time.perf_counter();omni=Omni(model=report['checkpoint'],**kwargs)
        report['engine_initialization_s']=time.perf_counter()-started;publish()
        sampling=OmniDiffusionSamplingParams(seed=1234,height=768,width=1344,num_frames=120,fps=24,frame_rate=24,num_inference_steps=50,
                output_type='np',return_frames=True,save_output=False,
                extra_args={'task':'fl2va','duration_seconds':5,'audio_mode':'native','frame_indices':[0,-1]})
        started=time.perf_counter()
        try:
            outputs=omni.generate({'prompt':'a red ball bouncing from left to right on the floor','multi_modal_data':{'image':images}},sampling_params_list=sampling,use_tqdm=False)
            report['engine_returned_after_intentional_stop']=True
            report['engine_output_errors']=[str(out.error) for out in outputs]
        except Exception as exc:
            report['engine_stop_exception']={'type':type(exc).__name__,'message':str(exc)}
        report['trajectory_wall_s']=time.perf_counter()-started
        rows=list((ROOT/'GPU_trajectory124').glob('pipeline_trajectory_rank*.json'))
        assert len(rows)==8
        states=[json.loads(path.read_text()) for path in rows]
        assert all(x['completed'] and x['actual_DiT_evaluations']==49 for x in states)
        rank0=next(row for row in states if row['rank']==0)
        assert set(rank0['snapshots'])=={'0','10','25','40','48'}
        assert len(list((ROOT/'GPU_trajectory124').glob('decode_intentionally_skipped_rank*.json')))==8
        report['completed']=True;report['actual_DiT_evaluations_by_rank']={str(row['rank']):row['actual_DiT_evaluations'] for row in states}
        report['all49_predictions_all8ranks_finite']=True;publish()
        print(json.dumps({'known_successful_GPU_pipeline_trajectory_captured':True,'actual_steps':49,'all5_snapshots_ready':True,'no_new_video_claimed':True}),flush=True)
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if omni is not None:omni.close()


if __name__=='__main__':main()
