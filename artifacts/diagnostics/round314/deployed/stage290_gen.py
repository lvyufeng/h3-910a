"""Minimal full-generation run with stage timing (encoder / video VAE / audio VAE)."""
import json
import time
import stageprobe290
stageprobe290.install()

import torch
import numpy as np
from PIL import Image, ImageDraw

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

out = {'scope': 'stage290', 'completed': False}

images = []
for name, x in [('first', 320), ('last', 1024)]:
    image = Image.new('RGB', (1344, 768), (200, 205, 212))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 640, 1344, 768), fill=(98, 102, 108))
    draw.ellipse((x - 45, 550, x + 45, 640), fill=(215, 30, 30))
    images.append(image)

kwargs = {'trust_remote_code': True, 'task_type': 'fl2va', 'num_gpus': 8, 'tensor_parallel_size': 2,
          'text_encoder_tp_size': 8, 'ulysses_degree': 4, 'ring_degree': 1, 'dtype': 'half',
          'enforce_eager': True, 'enable_layerwise_offload': True, 'vae_use_tiling': True,
          'vae_parallel_mode': 'tile', 'vae_patch_parallel_size': 8,
          'diffusion_attention_backend': 'TORCH_SDPA', 'stage_init_timeout': 2400, 'init_timeout': 2400}

t0 = time.perf_counter()
omni = Omni(model='/tmp/h3_align12401/original_metadata124f/FL2VA', **kwargs)
out['engine_init_s'] = time.perf_counter() - t0
print('[STAGE] engine_init %.1fs' % out['engine_init_s'], flush=True)

sampling = OmniDiffusionSamplingParams(
    seed=1234, height=768, width=1344, num_frames=120, fps=24, frame_rate=24,
    num_inference_steps=49, quality='lossless', output_type='np', return_frames=True,
    save_output=False,
    extra_args={'task': 'fl2va', 'duration_seconds': 5, 'audio_mode': 'native', 'frame_indices': [0, -1]})
request = {'prompt': 'a red ball bouncing from left to right on the floor', 'multi_modal_data': {'image': images}}

t0 = time.perf_counter()
outputs = omni.generate(request, sampling_params_list=sampling, use_tqdm=False)
out['generation_wall_s'] = time.perf_counter() - t0
out['completed'] = bool(outputs and outputs[0].finished and not outputs[0].error)
out['stage_times'] = dict(stageprobe290.T)
with open('/tmp/h3_r221/stage290_result.json', 'w') as f:
    json.dump(out, f, indent=2)
print('[STAGE] RESULT', json.dumps(out), flush=True)
