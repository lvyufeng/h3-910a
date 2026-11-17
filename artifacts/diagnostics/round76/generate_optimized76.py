"""Actual 49-step FL2VA videos using freshly optimized and independent-HF conditioning.

The optimized complete encoder is freshly recomputed in a separate owned8NPU
process before this run. Both video requests then use the supported EXTERNAL
TEXT CONDITIONING path,text_encoder=False and local VAE media encode. DiT and
VAE execute live. Request wall does NOT include live encoder/residency cost and
is not the full resident5.384s benchmark or official lossless comparison.
"""
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import prepare_encoder_inputs
from vllm_omni.model_executor.models.minimax_h3.preprocessing import build_minimax_h3_presentation
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3TextConditioning
from evidence_io76 import EvidenceWriter
ROOT = Path('/tmp/h3_round76_generation01')
MODEL = '/tmp/h3_round53_native49_fp16_model'
ENC = Path('/tmp/h3_round76_encoder01')
HF = Path('/tmp/h3_round74_hf01')

def raw(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()

def schema(value, depth=0):
    if depth > 5:
        return str(type(value))
    if isinstance(value, torch.Tensor):
        return {'type': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, Image.Image):
        return {'type': 'PIL', 'size': list(value.size)}
    if isinstance(value, (bytes, bytearray)):
        return {'type': 'bytes', 'length': len(value)}
    if isinstance(value, dict):
        return {str(k): schema(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {'type': str(type(value)), 'length': len(value), 'first': [schema(v, depth + 1) for v in value[:2]]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {'type': str(type(value)), 'attributes': list(getattr(value, '__dict__', {}))[:32]}

def find_frames(value):
    if isinstance(value, (bytes, bytearray)) and bytes(value)[4:8] == b'ftyp':
        import av
        with av.open(io.BytesIO(bytes(value))) as container:
            return np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)])
    if isinstance(value, (str, Path)) and str(value).endswith('.mp4') and Path(value).is_file():
        import av
        with av.open(str(value)) as container:
            return np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)])
    if isinstance(value, (np.ndarray, torch.Tensor)):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[-1] == 3:
            return array
        if array.ndim == 4 and array.shape[0] == 3:
            return array.transpose(1, 2, 3, 0)
        if array.ndim == 4 and array.shape[1] == 3:
            return array.transpose(0, 2, 3, 1)
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

def save_video(frames, path):
    import av
    assert frames.shape == (124, 768, 1344, 3)
    if frames.dtype != np.uint8:
        assert np.isfinite(frames).all()
        if frames.min() >= 0 and frames.max() <= 1:
            frames = (frames * 255).round().astype(np.uint8)
        else:
            raise ValueError('Unknown video pixel range:' + str((frames.min(), frames.max(), frames.dtype)))
    if path.exists():
        raise RuntimeError('Refuse overwriting generated video')
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width = 1344
        stream.height = 768
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18'}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert path.stat().st_size < 64 * 1024 * 1024
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'decoded_frame_count_expected': 124, 'encoding': 'libx264/yuv420p/crf18/24fps;video-onlyexport,audio separately retained in output schema'}

def main():
    writer = EvidenceWriter(ROOT / 'generation_result.json')
    r = {'scope': __doc__, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': [], 'native_bf16_compute_used': False, 'DiT_runtime_dtype': 'float16', 'encoder_in_generation_engine': False, 'conditioning_is_precomputed_for_this_request': True, 'live_full_encoder_end_to_end_performance_proven': False, 'official_goal_complete': False}
    omni = None
    import signal
    def stop(signum, frame):
        raise SystemExit('Generation interrupted:' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        payload = torch.load(ENC / 'conditioning.pt', map_location='cpu', weights_only=True)
        encoder_receipt = json.loads((ENC / 'encoder_rank0.json').read_text())
        assert hashlib.sha256((ENC / 'conditioning.pt').read_bytes()).hexdigest() == encoder_receipt['conditioning_file_sha256']
        optimized = payload['hidden_states_fp16']
        assert optimized.shape == (2043, 5120) and optimized.dtype == torch.float16
        assert raw(payload['hidden_states_bf16']) == '67ef196e33bcaa495420da1ebc07d1ff6dcaf89ddbaf75a350940ed8ae210eee'
        original_inputs = payload['actual_inputs']
        images = []
        for name, x in (('first', 320), ('last', 1024)):
            image = Image.new('RGB', (1344, 768), (200, 205, 212))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 640, 1344, 768), fill=(98, 102, 108))
            draw.ellipse((x - 45, 550, x + 45, 640), fill=(215, 30, 30))
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            data = buffer.getvalue()
            path = ROOT / (name + '.png')
            with path.open('xb') as f:
                f.write(data)
            images.append(image)
        prompt = 'a red ball bouncing from left to right on the floor'

        def sampling():
            return OmniDiffusionSamplingParams(seed=1234, height=768, width=1344, num_frames=120, fps=24, frame_rate=24, num_inference_steps=49, output_type='np', return_frames=True, save_output=False, extra_args={'task': 'fl2va', 'duration_seconds': 5, 'audio_mode': 'native'})
        raw_prompt = {'prompt': prompt, 'multi_modal_data': {'image': images}}
        prepared = prepare_encoder_inputs(raw_prompt, sampling(), task='fl2va')
        base = Path(MODEL) / 'FL2VA'
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(base), subfolder='tokenizer', local_files_only=True)
        processor = Qwen3VLProcessor.from_pretrained(str(base), subfolder='processor', local_files_only=True)
        pixels = processor.image_processor(images=prepared.images, return_tensors='pt')
        ids, tags = build_minimax_h3_presentation(tokenizer, prompt=prepared.prompt, task=prepared.media.task, condition_labels=prepared.condition_labels, image_grid_thw=pixels['image_grid_thw'], video_grid_thw=None, video_timestamps=prepared.video_timestamps, merge_size=int(processor.image_processor.merge_size))
        assert torch.equal(ids, original_inputs['input_ids']) and torch.equal(pixels['image_grid_thw'], original_inputs['image_grid_thw'])
        assert raw(pixels['pixel_values'].to(torch.float16)) == raw(original_inputs['pixel_values'])
        assert prepared.media.num_frames == 124 and tags.shape == (2043,)
        r['actual_inputs_match_current_encoder_fixture'] = True
        r['request'] = {'prompt': prompt, 'seconds': 5, 'width': 1344, 'height': 768, 'fps': 24, 'aligned_frames': 124, 'seed': 1234, 'num_inference_steps': 49, 'task': 'fl2va'}
        r['actual_input_ids_sha256'] = raw(ids)
        r['actual_tags_sha256'] = raw(tags)
        r['actual_FP16_transport_pixels_sha256'] = raw(original_inputs['pixel_values'])
        r['fresh_optimized_encoder_receipt'] = encoder_receipt['sources']
        writer.write(r)
        hf_reference = torch.load(HF / 'hf_selected50.pt', map_location='cpu', weights_only=True)['pre_final_norm'][0].to(torch.float16).contiguous()
        r['FP16_consumed_hidden_comparison'] = {'experimental_sha256': raw(optimized), 'HF_reference_sha256': raw(hf_reference), 'finite': bool(torch.isfinite(optimized.float()).all() and torch.isfinite(hf_reference.float()).all()), 'relative_rms': float((optimized.double() - hf_reference.double()).square().mean().sqrt() / hf_reference.double().square().mean().sqrt())}
        kwargs = {'trust_remote_code': True, 'task_type': 'fl2va', 'num_gpus': 8, 'usp': 8, 'ring': 1, 'text_encoder_tp_size': 8, 'vae_parallel_mode': 'tile', 'vae_use_tiling': True, 'vae_patch_parallel_size': 8, 'diffusion_attention_backend': 'TORCH_SDPA', 'stage_init_timeout': 2400, 'init_timeout': 2400, 'dtype': 'half', 'enable_distributed_layerwise_offload': True, 'model_loaded': {'transformer': True, 'vae': True, 'text_encoder': False, 'vae_encoder': True}}
        r['engine_kwargs'] = kwargs
        start = time.perf_counter()
        omni = Omni(model=MODEL, **kwargs)
        r['engine_startup_seconds_excluded'] = time.perf_counter() - start
        writer.write(r)
        for label, hidden in (('optimized', optimized), ('HF_reference_conditioning', hf_reference)):
            condition = MiniMaxH3TextConditioning.from_payload({'hidden_states': hidden.contiguous(), 'token_tags': tags.contiguous()}).to_payload()
            request = {'prompt': prompt, 'multi_modal_data': {'image': images}, 'additional_information': {'text_encoder_output': condition}}
            start = time.perf_counter()
            outputs = omni.generate(request, sampling_params_list=sampling(), use_tqdm=False)
            elapsed = time.perf_counter() - start
            assert len(outputs) == 1 and outputs[0].finished and (not outputs[0].error)
            out = outputs[0]
            row = {'case': label, 'request_wall_s_excluding_live_encoder': elapsed, 'request_id': out.request_id, 'metrics': schema(out.metrics), 'stage_durations': schema(out.stage_durations), 'multimodal_output_schema': schema(out.multimodal_output), 'custom_output_schema': schema(out.custom_output)}
            frames = find_frames(out.multimodal_output)
            if frames is None:
                frames = find_frames(out.images)
            if frames is None:
                r['unhandled_output_schema'] = row
                writer.write(r)
                raise ValueError('Generated output has no recognized decoded video frames;preserve schema for exact media handling')
            if frames.dtype != np.uint8:
                if np.isfinite(frames).all() and frames.min() >= 0 and frames.max() <= 1:
                    frames = (frames * 255).round().astype(np.uint8)
                else:
                    raise ValueError('Unknown decoded video pixel format/range')
            row['raw_RGB_frame_sha256'] = hashlib.sha256(memoryview(frames)).hexdigest()
            row['video'] = save_video(frames, ROOT / (label + '.mp4'))
            for index in (0, 12, 24, 36, 48, 62, 74, 86, 98, 110, 123):
                Image.fromarray(frames[index].astype(np.uint8)).save(ROOT / (label + f'_frame{index:03d}.png'))
            r['cases'].append(row)
            writer.write(r)
            print(json.dumps({'case': label, 'request_wall_s': elapsed, 'video': row['video']}), flush=True)
        r['completed'] = True
    except BaseException as exc:
        r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        if omni is not None:
            omni.close()
            r['engine_close_called'] = True
        writer.write(r)
if __name__ == '__main__':
    main()
