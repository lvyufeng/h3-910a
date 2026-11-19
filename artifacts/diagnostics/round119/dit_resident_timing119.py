"""Ideal residency: DiT and VAE RESIDENT, only the text encoder offloaded, encoder still live.

This is the measurement every earlier per-step figure was missing. Round 76 timed the
same request with `text_encoder=False` and an injected precomputed conditioning payload,
so its 14.657 s/step covered DiT and VAE only. Here the encoder is resident and runs
inside each request, so the per-step numbers cover the whole stack at the goal's
baseline request (1344x768, 5 s, 24 fps, 124 frames, seed 1234, 49 steps, fl2va,
USP8 on 8x910A) using the official MiniMax-H3 50-block model in FP16.

Two identical requests are issued: the first absorbs TBE compilation and residency
warm-up, the second is the measured one. Timing is reported per step from the engine's
own stage durations, and the pipeline [H3PERF] lines are captured from stdout so the
prep/model/update split can be read for a WARM step rather than a cold one.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

ROOT = Path('/tmp/h3_round119b_dit_resident')
MODEL = '/tmp/h3_round53_native49_fp16_model'
REQUEST = {'prompt': 'a red ball bouncing from left to right on the floor',
           'width': 1344, 'height': 768, 'fps': 24, 'aligned_frames': 124,
           'seconds': 5, 'seed': 1234, 'num_inference_steps': 49, 'task': 'fl2va'}


def schema(value, depth=0):
    if depth > 4:
        return str(type(value))
    if isinstance(value, torch.Tensor):
        return {'type': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, (bytes, bytearray)):
        return {'type': 'bytes', 'length': len(value)}
    if isinstance(value, dict):
        return {str(k): schema(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {'type': str(type(value)), 'length': len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {'type': str(type(value))}


def find_frames(value):
    if isinstance(value, (bytes, bytearray)) and bytes(value)[4:8] == b'ftyp':
        import av
        with av.open(io.BytesIO(bytes(value))) as container:
            return np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    if isinstance(value, (str, Path)) and str(value).endswith('.mp4') and Path(value).is_file():
        import av
        with av.open(str(value)) as container:
            return np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    if isinstance(value, (np.ndarray, torch.Tensor)):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[-1] == 3:
            return array
        if array.ndim == 4 and array.shape[0] == 3:
            return array.transpose(1, 2, 3, 0)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Image.Image):
        return np.stack([np.asarray(v.convert('RGB')) for v in value])
    if isinstance(value, dict):
        for key in ('video', 'videos', 'frames', 'video_frames', 'output', 'data', 'media'):
            if key in value:
                found = find_frames(value[key])
                if found is not None:
                    return found
        for child in value.values():
            found = find_frames(child)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for child in value:
            found = find_frames(child)
            if found is not None:
                return found
    for key in ('frames', 'video_frames', 'data', 'media', 'path', 'encoded_video', 'video_bytes'):
        if hasattr(value, key):
            found = find_frames(getattr(value, key))
            if found is not None:
                return found
    return None


def write(path, payload):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def make_images(root):
    images = []
    for name, x in (('first', 320), ('last', 1024)):
        image = Image.new('RGB', (1344, 768), (200, 205, 212))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 640, 1344, 768), fill=(98, 102, 108))
        draw.ellipse((x - 45, 550, x + 45, 640), fill=(215, 30, 30))
        images.append(image)
    return images


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    destination = ROOT / 'dit_resident_timing119b.json'
    r = {
        'scope': __doc__,
        'completed': False,
        'goal_complete': False,
        'quality_pass': False,
        'numerical_tolerance': None,
        'model': MODEL,
        'model_basis': 'official undistilled MiniMax-H3 50-block encoders/VAEs + official 50-block FP16 transformer',
        'request': REQUEST,
        'encoder_in_generation_engine': True,
        'conditioning_is_precomputed_for_this_request': False,
        'live_full_encoder_end_to_end_performance_proven': False,
        'cases': [],
    }
    write(destination, r)
    images = make_images(ROOT)
    omni = None
    try:
        kwargs = {
            'trust_remote_code': True, 'task_type': 'fl2va', 'num_gpus': 8,
            'ulysses_degree': 8, 'ring_degree': 1, 'text_encoder_tp_size': 8,
            'vae_parallel_mode': 'tile', 'vae_use_tiling': True, 'vae_patch_parallel_size': 8,
            'diffusion_attention_backend': 'TORCH_SDPA',
            'stage_init_timeout': 2400, 'init_timeout': 2400, 'dtype': 'half',
            # Keep the DiT resident: offload ONLY the text encoder. The legacy
            # boolean must stay False because the public config replaces it.
            'enable_distributed_layerwise_offload': False,
            'diffusion_offload_config': {'mode': 'layer', 'components': ['text_encoder']},
            'model_loaded': {'transformer': True, 'vae': True, 'text_encoder': True, 'vae_encoder': True},
        }
        r['engine_kwargs'] = kwargs
        start = time.perf_counter()
        omni = Omni(model=MODEL, **kwargs)
        r['engine_startup_seconds_excluded'] = time.perf_counter() - start
        write(destination, r)

        sampling = OmniDiffusionSamplingParams(
            seed=1234, height=768, width=1344, num_frames=120, fps=24, frame_rate=24,
            num_inference_steps=49, output_type='np', return_frames=True, save_output=False,
            extra_args={'task': 'fl2va', 'duration_seconds': 5, 'audio_mode': 'native'})
        # No additional_information: the resident encoder must run inside each request.
        request = {'prompt': REQUEST['prompt'], 'multi_modal_data': {'image': images}}

        # One 49-step request at the goal's exact configuration. The pipeline prints
        # per-step progress and [H3PERF] lines; steps 1-4 absorb TBE compilation, so the
        # warm cost is read from the later steps rather than from a second request.
        for label in ('measured_49steps',):
            start = time.perf_counter()
            outputs = omni.generate(request, sampling_params_list=sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            assert len(outputs) == 1 and outputs[0].finished and (not outputs[0].error)
            out = outputs[0]
            row = {
                'case': label,
                'request_wall_s_with_live_encoder': elapsed,
                'request_id': out.request_id,
                'metrics': schema(out.metrics),
                'stage_durations': schema(out.stage_durations),
                'multimodal_output_schema': schema(out.multimodal_output),
                'custom_output_schema': schema(out.custom_output),
            }
            frames = find_frames(out.multimodal_output) or find_frames(out.images)
            if frames is None:
                row['frames_present'] = False
            else:
                assert frames.shape == (124, 768, 1344, 3), frames.shape
                row['frames_present'] = True
                row['aligned_frames'] = int(frames.shape[0])
                row['raw_RGB_frame_sha256'] = hashlib.sha256(memoryview(np.ascontiguousarray(frames))).hexdigest()
                row['frame_min'] = int(frames.min())
                row['frame_max'] = int(frames.max())
                if label == 'measured_49steps':
                    for index in (0, 24, 48, 74, 98, 123):
                        Image.fromarray(frames[index].astype(np.uint8)).save(ROOT / ('measured_frame%03d.png' % index))
            r['cases'].append(row)
            write(destination, r)
            print(json.dumps({'case': label, 'request_wall_s_with_live_encoder': elapsed,
                              'frames_present': row['frames_present']}), flush=True)
        r['completed'] = True
    except BaseException as exc:
        r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        if omni is not None:
            omni.close()
            r['engine_close_called'] = True
        write(destination, r)


if __name__ == '__main__':
    main()