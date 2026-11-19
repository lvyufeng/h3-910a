#!/usr/bin/env python3
"""Complete original-checkpoint H3 via vLLM-Omni with explicit range-safe FP16 matrices.

Encoder and main/refiner matrix inputs and weights are FP16. Native FP32 projections
and VAE weights are retained. Measured range fixes: FP32 conditioning projection,
residual streams/RMS accumulation, SwiGLU point math with exact power-of-two scaling,
and native FP16-input/weight GEMM with FP32 MLP output/TP reduction. No BF16 fallback,
clipping, LoRA/student weights, fake norm, or cached conditioning. This is not a pure
Half or a lossless-BF16 claim. TP2/USP4 on 8x4090 reduces replicated-token overhead;
original FP16 trials and their exact configs/failures remain separately preserved.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/mseco/h3_gpu12201')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def summarize(value, depth=0):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return {'type': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, (bytes, bytearray)):
        return {'type': 'bytes', 'size': len(value)}
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    if depth > 5:
        return {'type': type(value).__name__}
    if isinstance(value, dict):
        return {str(k): summarize(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(x, depth + 1) for x in value[:5]]
    return {'type': type(value).__name__}


def find_frames(value):
    import av
    import numpy as np
    import torch
    from PIL import Image
    if isinstance(value, (bytes, bytearray)) and bytes(value)[4:8] == b'ftyp':
        import io
        with av.open(io.BytesIO(bytes(value))) as c:
            return np.stack([x.to_ndarray(format='rgb24') for x in c.decode(video=0)])
    if isinstance(value, (torch.Tensor, np.ndarray)):
        a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        if a.ndim == 5 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 4 and a.shape[-1] == 3:
            return a
        if a.ndim == 4 and a.shape[0] == 3:
            return a.transpose(1, 2, 3, 0)
        if a.ndim == 4 and a.shape[1] == 3:
            return a.transpose(0, 2, 3, 1)
        return None
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Image.Image):
        return np.stack([np.asarray(x.convert('RGB')) for x in value])
    if isinstance(value, dict):
        for k in ['video', 'videos', 'frames', 'video_frames', 'output', 'data', 'media']:
            if k in value:
                result = find_frames(value[k])
                if result is not None:
                    return result
        for x in value.values():
            result = find_frames(x)
            if result is not None:
                return result
    if isinstance(value, (list, tuple)):
        for x in value:
            result = find_frames(x)
            if result is not None:
                return result
    return None


def find_audio(value):
    import numpy as np
    import torch
    if isinstance(value, dict):
        for k in ['audio', 'waveform', 'audio_output']:
            if k in value and isinstance(value[k], (np.ndarray, torch.Tensor)):
                x = value[k]
                x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                if x.ndim == 3 and x.shape[0] == 1:
                    x = x[0]
                if x.ndim == 2 and x.shape[0] == 2:
                    return x.astype(np.float32, copy=False)
        for x in value.values():
            result = find_audio(x)
            if result is not None:
                return result
    return None


def export_av(frames, audio, path, fps=24, sample_rate=32000):
    import av
    import numpy as np
    with av.open(str(path), mode='w') as container:
        video = container.add_stream('libx264', rate=fps)
        video.width = frames.shape[2]
        video.height = frames.shape[1]
        video.pix_fmt = 'yuv420p'
        video.options = {'crf': '18', 'preset': 'medium'}
        sound = container.add_stream('aac', rate=sample_rate)
        sound.layout = 'stereo'
        sound.bit_rate = 192000
        for i, x in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(x), format='rgb24')
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        for i in range(0, audio.shape[1], 1024):
            chunk = np.ascontiguousarray(audio[:, i:i + 1024], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout='stereo')
            frame.sample_rate = sample_rate
            frame.pts = i
            frame.time_base = Fraction(1, sample_rate)
            for packet in sound.encode(frame):
                container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)


def validate_av(path):
    import av
    with av.open(str(path)) as c:
        assert len(c.streams.video) == 1 and len(c.streams.audio) == 1
        v = c.streams.video[0]
        a = c.streams.audio[0]
        metadata = {'video_codec': v.codec_context.name, 'audio_codec': a.codec_context.name,
                    'width': v.codec_context.width, 'height': v.codec_context.height,
                    'fps': str(v.average_rate), 'sample_rate': a.codec_context.sample_rate,
                    'audio_channels': a.codec_context.channels}
        frames = sum(1 for _ in c.decode(video=0))
    with av.open(str(path)) as c:
        samples = sum(x.samples for x in c.decode(audio=0))
    assert frames == 124 and samples > 160000
    metadata['decoded_video_frames'] = frames
    metadata['decoded_audio_samples'] = samples
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='range_safe_fp16')
    args = parser.parse_args()
    if args.case not in {'range_safe_fp16'}:
        raise ValueError('Unrecognized case')
    destination = ROOT / 'results' / args.case
    destination.mkdir(exist_ok=False)
    os.environ['H3_GPU_RESULT_ROOT'] = str(destination)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import soundfile as sf
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    import importlib.metadata as metadata

    report = {
        'scope': __doc__, 'completed': False, 'actual_generated_video': False,
        'FP16_quality_accepted': False, 'case': args.case,
        'versions': {k: metadata.version(k) for k in ['torch', 'vllm', 'vllm-omni', 'transformers', 'diffusers']},
        'model_repo': 'MiniMaxAI/MiniMax-H3', 'model_revision': '0919fb684102ef115126612f3fbb84d1027e7317',
        'vllm_omni_commit': '1c2a81f6d84aea4fff53bd2f894c2a287c237245',
        'checkpoint_root': str(ROOT / 'checkpoint/FL2VA'),
        'request': {'prompt': 'a red ball bouncing from left to right on the floor', 'seed': 1234,
                    'width': 1344, 'height': 768, 'nominal_seconds': 5, 'fps': 24,
                    'requested_frames': 120, 'expected_aligned_frames': 124, 'dit_evaluations': 49, 'task': 'fl2va'},
        'live_encoder_used': True, 'old_conditioning_cache_loaded': False,
        'mixed_precision': True, 'original_direct_Half_trial_passed': False,
        'main_DiT_50_block_matrix_inputs_and_weights_FP16': True, 'main_DiT_residual_streams_FP32': True, 'MLP_down_projection_outputs_and_TP_reduction_FP32': True, 'refiner_attention_and_MLP_FP16': True,
        'condition_projection_and_refiner_residual_RMS_FP32': True,
        'FP32_intentional_exceptions': ['time/latent/final-output projections', 'original video/audio VAE weights',
                                       'upstream normalization/softmax/rotary accumulation'],
    }
    receipt = destination / 'generation_result122.json'
    def publish():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    publish()
    omni = None
    try:
        manifest = json.loads((ROOT / 'checkpoint_verified122.json').read_text())
        assert manifest['completed'] and manifest['revision'] == report['model_revision']
        adaptation = json.loads((ROOT / 'fp16_source_adaptation122b.json').read_text())
        assert adaptation['completed']
        stability = json.loads((ROOT / 'MLP_Half_input_FP32_output_fix122.json').read_text())
        executed_transformer = ROOT / 'vllm-omni-fl2va-release/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'
        assert file_sha(executed_transformer) == stability['modified_sha256']
        report['MLP_Half_input_FP32_output_fix'] = stability
        report['main_residual_stability_fix'] = json.loads((ROOT / 'main_residual_FP32_stability_fix122.json').read_text())
        report['conditioning_stability_fix'] = json.loads((ROOT / 'conditioning_FP32_stability_fix122.json').read_text())
        report['FP32_intentional_exceptions'].append('condition_proj, main/refiner residual/RMS, SwiGLU point math and MLP down-GEMM output/TP reduction; all large GEMM inputs and weights remain FP16')
        report['checkpoint_verified_files'] = len(manifest['files'])
        report['checkpoint_verified_total_bytes'] = manifest['expected_total_bytes']
        report['dtype_adaptation'] = adaptation
        assert torch.cuda.device_count() == 8
        assert all('RTX 4090' in torch.cuda.get_device_name(i) for i in range(8))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        report['GPU_names'] = [torch.cuda.get_device_name(i) for i in range(8)]
        images = []
        inputs = {}
        for name, x in [('first', 320), ('last', 1024)]:
            image = Image.new('RGB', (1344, 768), (200, 205, 212))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 640, 1344, 768), fill=(98, 102, 108))
            draw.ellipse((x - 45, 550, x + 45, 640), fill=(215, 30, 30))
            image_path = destination / f'{name}.png'
            image.save(image_path)
            inputs[name] = {'file': str(image_path), 'sha256': file_sha(image_path)}
            images.append(image)
        report['input_images'] = inputs
        kwargs = {'trust_remote_code': True, 'task_type': 'fl2va', 'num_gpus': 8,
                  'tensor_parallel_size': 2, 'text_encoder_tp_size': 8,
                  'ulysses_degree': 4, 'ring_degree': 1, 'dtype': 'half', 'enforce_eager': True,
                  'enable_layerwise_offload': True,
                  'vae_use_tiling': True, 'vae_parallel_mode': 'tile', 'vae_patch_parallel_size': 8,
                  'diffusion_attention_backend': 'TORCH_SDPA', 'stage_init_timeout': 2400, 'init_timeout': 2400}
        report['engine_kwargs'] = kwargs
        publish()
        start = time.perf_counter()
        omni = Omni(model=str(ROOT / 'checkpoint/FL2VA'), **kwargs)
        report['engine_initialization_s'] = time.perf_counter() - start
        publish()
        report['sigma_schedule_contract'] = {'sampling_num_inference_steps': 50, 'sigma_points': 50, 'actual_dit_evaluations_expected': 49, 'reason': 'Pinned upstream version counts sigma points; equivalent 50 boundaries to the newer 49-evaluation API.'}
        sampling = OmniDiffusionSamplingParams(seed=1234, height=768, width=1344, num_frames=120,
                    fps=24, frame_rate=24, num_inference_steps=50, output_type='np', return_frames=True,
                    save_output=False, extra_args={'task': 'fl2va', 'duration_seconds': 5, 'audio_mode': 'native', 'frame_indices': [0, -1]})
        request = {'prompt': report['request']['prompt'], 'multi_modal_data': {'image': images}}
        start = time.perf_counter()
        outputs = omni.generate(request, sampling_params_list=sampling, use_tqdm=False)
        report['generation_wall_s_including_live_encoder'] = time.perf_counter() - start
        assert len(outputs) == 1 and outputs[0].finished and not outputs[0].error
        out = outputs[0]
        report['metrics'] = summarize(out.metrics)
        report['stage_durations'] = summarize(getattr(out, 'stage_durations', {}))
        frames = find_frames(out.multimodal_output)
        if frames is None:
            frames = find_frames(out.images)
        audio = find_audio(out.multimodal_output)
        if frames is None or audio is None:
            report['output_schema'] = summarize(out.multimodal_output)
            raise ValueError('Complete generation requires both actual video frames and native audio')
        assert frames.shape == (124, 768, 1344, 3), frames.shape
        if frames.dtype != np.uint8:
            assert np.isfinite(frames).all()
            assert frames.min() >= 0 and frames.max() <= 1, (frames.min(), frames.max())
            frames = np.rint(frames * 255).astype(np.uint8)
        assert np.isfinite(audio).all() and audio.shape[0] == 2
        worker_rows = [json.loads(line) for path in destination.glob('worker_audit_rank*.jsonl') for line in path.read_text().splitlines()]
        step_counts = {str(i): sum(row['event'] == 'dit_step_end' and row['rank'] == i for row in worker_rows) for i in range(8)}
        report['actual_dit_evaluations_by_rank'] = step_counts
        assert all(count == 49 for count in step_counts.values()), step_counts
        report['video_raw_shape'] = list(frames.shape)
        report['video_raw_rgb_sha256'] = hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()
        report['audio_raw_shape'] = list(audio.shape)
        report['audio_raw_peak'] = float(np.abs(audio).max())
        report['actual_video_duration_s'] = len(frames) / 24
        report['actual_audio_duration_s'] = audio.shape[1] / 32000
        wav = destination / 'h3_vllm_omni_fp16_audio.wav'
        sf.write(str(wav), audio.T, 32000, subtype='FLOAT')
        mp4 = destination / 'h3_vllm_omni_fp16.mp4'
        export_av(frames, audio, mp4)
        report['media_validation'] = validate_av(mp4)
        report['files'] = {x.name: {'bytes': x.stat().st_size, 'sha256': file_sha(x)} for x in [wav, mp4]}
        indexes = [0, 12, 24, 36, 48, 62, 74, 86, 98, 110, 123]
        contact = Image.new('RGB', (336 * 4, 192 * 3), (255, 255, 255))
        for j, i in enumerate(indexes):
            img = Image.fromarray(frames[i])
            img.save(destination / f'frame_{i:03d}.png')
            thumb = img.resize((336, 192))
            contact.paste(thumb, ((j % 4) * 336, (j // 4) * 192))
        contact.save(destination / 'contact_sheet.png')
        report['actual_generated_video'] = True
        report['completed'] = True
        publish()
        print(json.dumps({'completed': True, 'case': args.case, 'video': str(mp4),
                          'wall_s': report['generation_wall_s_including_live_encoder'],
                          'FP16_quality_accepted': False}), flush=True)
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise
    finally:
        if omni is not None:
            omni.close()
            report['engine_close_called'] = True
            publish()


if __name__ == '__main__':
    main()
