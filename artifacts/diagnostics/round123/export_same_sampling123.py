#!/usr/bin/env python3
"""Build exact shared sampling inputs for GPU/910A replay from the successful request.

Runs ON the owned RTX4090. It reconstructs the successful CPU RNG/sigma contract
and GPU-encodes the two original keyframes once; it is not a CPU inference baseline
and does not rerun the 49-step model. The old generation did not store its clean or
noised keyframe latents, so bit identity to those unstored old tensors is NOT proved.
The exported replay bundle itself is identical between the new GPU and NPU runs.
Reuses the actual saved live GPU encoder output as explicitly teacher-forced
conditioning for diagnostics, never as a hidden performance cache. No model weights
go to the local workstation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/home/mseco/h3_align12301')
BASE = Path('/home/mseco/h3_gpu12201')


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def main():
    import numpy as np
    import torch
    from PIL import Image
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER, minimax_h3_time_shift_sigmas
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    manifest_path = ROOT / 'same_initial_sampling123.json'
    assert not manifest_path.exists()
    result = {'scope': __doc__, 'completed': False, 'no_local_weight_download_or_relay': True,
              'successful_full_GPU_request_receipt': str(BASE / 'results/wide_output_fp16/generation_result122.json'),
              'teacher_forced_GPU_encoder_used_only_for_diagnostics': True, 'new_full_generation_executed': False,
              'same_checkpoint_revision': '0919fb684102ef115126612f3fbb84d1027e7317', 'started_unix': time.time()}
    def publish():
        manifest_path.write_text(json.dumps(result, indent=2) + '\n')
    publish()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        assert 'RTX 4090' in torch.cuda.get_device_name(0)
        complete = json.loads((BASE / 'results/wide_output_fp16/generation_result122.json').read_text())
        assert complete['completed'] and complete['actual_dit_evaluations_by_rank'] == {str(i): 49 for i in range(8)}
        capture_path = BASE / 'results/wide_output_fp16/vllm_fp16_encoder_hidden122.pt'
        capture = torch.load(capture_path, map_location='cpu', weights_only=True)
        hidden = capture['hidden_states'].contiguous()
        text_tags = capture['token_tags'].contiguous()
        assert hidden.shape == (2043, 5120) and hidden.dtype == torch.float16 and torch.isfinite(hidden).all()
        frame_count = MINIMAX_H3_SHAPE_PLANNER.align_frame_count(120)
        latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(frame_count)
        audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(frame_count / 24)
        assert (frame_count, latent_t, audio_t) == (124, 37, 207)
        vg = torch.Generator(device='cpu').manual_seed(1234)
        video_latent_noise = torch.randn((1, 24, latent_t, 48, 84), generator=vg, dtype=torch.float32)
        video_target_rows = minimax_h3_patchify_video_latent(video_latent_noise, patch_size=(1, 2, 2)).contiguous()
        ag = torch.Generator(device='cpu').manual_seed(1234)
        audio_rows = torch.randn((audio_t * 2, 32), generator=ag, dtype=torch.float32)
        sv = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=12.0)
        sa = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=3.0)
        assert len(sv) == len(sa) == 50 and sv[0] == sa[0] == 1.0 and sv[-1] == sa[-1] == 0.0
        packed = minimax_h3_packed_sequence(text_len=len(hidden), latent_t=latent_t, latent_h=48, latent_w=84,
                    audio_t=audio_t, include_keyframe_cond=True, keyframe_frame_indices=[0, -1], frame_count=frame_count)
        token_tags = packed['token_tags'].clone()
        token_tags[packed['text_pos']] = text_tags
        update_mask = packed['update_mask']
        images = []
        image_receipts = {}
        for name in ['first', 'last']:
            path = BASE / 'results/wide_output_fp16' / f'{name}.png'
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == complete['input_images'][name]['sha256']
            target = ROOT / f'{name}.png'
            with target.open('xb') as f:
                f.write(data)
            images.append(Image.open(path).convert('RGB'))
            image_receipts[name] = {'path': str(target), 'sha256': hashlib.sha256(data).hexdigest()}
        vae = MiniMaxH3VideoVAE(str(BASE / 'checkpoint/FL2VA/video_vae'), device=torch.device('cuda:0'))
        assert all(p.dtype == torch.float32 for p in vae.parameters())
        with torch.inference_mode():
            clean_rows = torch.cat([vae.encode_image(image).cpu() for image in images], dim=0).float().contiguous()
        del vae
        torch.cuda.empty_cache()
        visual_shapes = [(1, 48, 84), (1, 48, 84)]
        visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(clean_rows, condition_shapes=visual_shapes,
                    target_latent_t=latent_t, imgvid_cond_num_frames=2, seed=1234, noise_aug=0.999).cpu().contiguous()
        assert visual_anchor.shape == (2016, 96)
        initial_video_rows = torch.zeros((len(packed['img_pos']), 96), dtype=torch.float32)
        initial_video_rows[update_mask] = video_target_rows
        initial_video_rows[~update_mask] = visual_anchor
        payload = {'text_embeddings': hidden, 'text_tags': text_tags, 'packed': packed, 'token_tags': token_tags,
                   'video_latent_noise': video_latent_noise, 'initial_video_target_rows': video_target_rows,
                   'initial_video_rows_with_keyframes': initial_video_rows, 'initial_audio_rows': audio_rows,
                   'keyframe_clean_rows_GPU_FP32': clean_rows, 'keyframe_noise_aug_rows': visual_anchor,
                   'sigmas_video': torch.tensor(sv, dtype=torch.float32), 'sigmas_audio': torch.tensor(sa, dtype=torch.float32)}
        path = ROOT / 'same_initial_sampling123.pt'
        with path.open('xb') as f:
            torch.save(payload, f)
        result.update({'completed': True, 'torch_version': torch.__version__, 'GPU': torch.cuda.get_device_name(0),
                       'shape': {'frames': frame_count, 'latent_t': latent_t, 'latent_h': 48, 'latent_w': 84,
                                 'audio_t_per_channel': audio_t, 'text_rows': len(hidden), 'packed_rows': int(len(token_tags))},
                       'sigma_video_FP32_bits': np.asarray(sv, dtype=np.float32).view(np.uint32).tolist(),
                       'sigma_audio_FP32_bits': np.asarray(sa, dtype=np.float32).view(np.uint32).tolist(),
                       'sigma_points': 50, 'actual_evaluations': 49, 'images': image_receipts,
                       'tensor_identities': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256_raw_tensor_bytes': tensor_sha(value)}
                                            for key, value in payload.items() if isinstance(value, torch.Tensor)},
                       'payload_file': str(path), 'payload_bytes': path.stat().st_size,
                       'payload_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'finished_unix': time.time()})
        publish()
        print(json.dumps({key: value for key, value in result.items() if key not in ['scope', 'tensor_identities', 'sigma_video_FP32_bits', 'sigma_audio_FP32_bits']}, indent=2), flush=True)
    except BaseException as exc:
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise


if __name__ == '__main__':
    main()
