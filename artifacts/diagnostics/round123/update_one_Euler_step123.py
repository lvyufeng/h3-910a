#!/usr/bin/env python3
"""Apply one actual GPU/NPU RF-v/Euler-eta0 update to the full first-step prediction.

Uses the exact successful GPU sampler source and exported FP32 sigma bit values.
No model computation is repeated; no CPU model inference. Integer index_copy keeps
keyframe anchors pinned and avoids the known 910A Boolean IndexPutV2 failure.
This is the first step only, not a full render or multi-step precision acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--backend',choices=['cuda','npu'],required=True)
    parser.add_argument('--forward-label',required=True)
    args=parser.parse_args()
    import torch
    if args.backend=='npu':
        import torch_npu
        torch.npu.set_device(0)
    else:
        torch.cuda.set_device(0)
    from h3_replay123.scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0, minimax_h3_euler_eta0_step
    root=Path(args.root)
    source=root/args.forward_label
    destination=root/(args.forward_label+'_Euler1')
    assert not destination.exists();destination.mkdir()
    receipt=json.loads((source/'full_first_step_rank0.json').read_text())
    assert receipt['completed'] and receipt['DiT_forward_count']==1
    bundle=torch.load(root/'same_initial_sampling123.pt',map_location='cpu',weights_only=True)
    velocities={name:torch.load(source/f'{name}_velocity_all_rows_full_rank0.pt',map_location='cpu',weights_only=True)['value'] for name in ['video','audio']}
    for name,value in velocities.items():
        expected=receipt['captures'][name+'_velocity_all_rows']['full_tensor_raw_sha256']
        assert hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==expected
    device=torch.device(args.backend+':0')
    video=bundle['initial_video_rows_with_keyframes'].to(device)
    audio=bundle['initial_audio_rows'].to(device)
    target_cpu=torch.nonzero(bundle['packed']['update_mask'],as_tuple=False).flatten().long()
    anchor_cpu=torch.nonzero(~bundle['packed']['update_mask'],as_tuple=False).flatten().long()
    target=target_cpu.to(device);anchor=anchor_cpu.to(device)
    video_v=velocities['video'].to(device).index_select(0,target)
    audio_v=velocities['audio'].to(device)
    sv0,sv1=[float(x) for x in bundle['sigmas_video'][:2]]
    sa0,sa1=[float(x) for x in bundle['sigmas_audio'][:2]]
    with torch.inference_mode():
        video_target=video.index_select(0,target)
        x0v=minimax_h3_rf_v_to_x0(video_target,video_v,torch.tensor(1.-sv0,dtype=torch.float32,device=device))
        next_video=minimax_h3_euler_eta0_step(video_target,x0v,sigma_curr=sv0,sigma_next=sv1)
        video=video.clone().index_copy(0,target,next_video)
        anchors=bundle['keyframe_noise_aug_rows'].to(device)
        video.index_copy_(0,anchor,anchors)
        x0a=minimax_h3_rf_v_to_x0(audio,audio_v,torch.tensor(1.-sa0,dtype=torch.float32,device=device))
        next_audio=minimax_h3_euler_eta0_step(audio,x0a,sigma_curr=sa0,sigma_next=sa1)
        getattr(torch,args.backend).synchronize()
    assert video.device.type==args.backend and next_audio.device.type==args.backend
    video_cpu=video.cpu().contiguous();audio_cpu=next_audio.cpu().contiguous()
    assert torch.equal(video_cpu.index_select(0,anchor_cpu).view(torch.uint8),bundle['keyframe_noise_aug_rows'].view(torch.uint8))
    assert torch.isfinite(video_cpu).all() and torch.isfinite(audio_cpu).all()
    state={'next_video_rows_with_pinned_keyframes':video_cpu,'next_audio_rows':audio_cpu}
    path=destination/'next_state_full.pt'
    with path.open('xb') as file:torch.save(state,file)
    target_video=video_cpu.index_select(0,target_cpu)
    selected=torch.linspace(0,len(target_video)-1,512,dtype=torch.float64).round().long().unique()
    sample={'video_target_rows':target_video.index_select(0,selected),'video_target_row_indices':selected,
            'video_target_full_shape':list(target_video.shape),'audio_all_rows':audio_cpu}
    sample_path=destination/'next_state_sample.pt'
    with sample_path.open('xb') as file:torch.save(sample,file)
    report={'scope':__doc__,'completed':True,'backend':args.backend,'step_count':1,
            'real_full50_layer_prediction_used':True,'sampler_source_sha256':hashlib.sha256((root/'h3_replay123/scheduling_minimax_h3_euler_ancestral.py').read_bytes()).hexdigest(),
            'CPU_model_inference_executed':False,'video_sigma_current':sv0,'video_sigma_next':sv1,
            'audio_sigma_current':sa0,'audio_sigma_next':sa1,'all2016_keyframe_anchor_rows_reset_raw_bit_identical':True,
            'full_video_audio_state_finite':True,'next_video_shape':list(video_cpu.shape),'next_audio_shape':list(audio_cpu.shape),
            'full_state_path':str(path),'full_state_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'sample_path':str(sample_path),'sample_sha256':hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            'sample_video_rows':len(selected),'all414_audio_rows_present':True,'multi_step_precision_accepted':False}
    (destination/'Euler1_receipt123.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
