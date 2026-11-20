#!/usr/bin/env python3
"""Hash existing checkpoint values used by conditioning and the first two blocks.

Only metadata/digests leave each owned host. No model weights are copied locally,
no model inference is executed on CPU, and no Half-to-BF16 roundtrip is performed.
Canonical values use the exact successful GPU storage contract before TP slicing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

FP32 = frozenset([
    'condition_proj.weight', 'condition_proj.bias', 'video_patch_proj.weight', 'video_patch_proj.bias',
    'audio_patch_proj.weight', 'audio_patch_proj.bias', 'time_embedder.proj_in.weight', 'time_embedder.proj_in.bias',
    'time_embedder.proj_out.weight', 'time_embedder.proj_out.bias', 'final_layer.video_out.weight',
    'final_layer.video_out.bias', 'final_layer.audio_out.weight', 'final_layer.audio_out.bias', 'rope.inv_freq',
])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint)
    out = Path(args.output)
    assert not out.exists()
    index_path = checkpoint / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    selected = sorted(name for name in index if not name.startswith('blocks.') or name.startswith(('blocks.0.', 'blocks.1.')))
    result = {'scope': __doc__, 'completed': False, 'checkpoint': str(checkpoint),
              'config_sha256': hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest(),
              'no_local_weight_download': True, 'CPU_model_inference_executed': False,
              'canonical_precision_contract': 'FP16 matrices/norm parameters; explicit original FP32 front/time/final/rope exception set',
              'selected_names_count': len(selected), 'parameters': {}}
    started = time.perf_counter()
    for filename in sorted({index[name] for name in selected}):
        with safe_open(str(checkpoint / filename), framework='pt', device='cpu') as file:
            for name in selected:
                if index[name] != filename:
                    continue
                value = file.get_tensor(name)
                canonical = value.to(torch.float32 if name in FP32 else torch.float16).contiguous()
                result['parameters'][name] = {'shape': list(value.shape), 'stored_dtype': str(value.dtype),
                                             'canonical_dtype': str(canonical.dtype),
                                             'canonical_sha256_raw_tensor_bytes': hashlib.sha256(canonical.view(torch.uint8).numpy().tobytes()).hexdigest()}
                del canonical, value
    result['completed'] = True
    result['hash_elapsed_s_not_model_inference'] = time.perf_counter() - started
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'completed': True, 'parameters_hashed': len(result['parameters']),
                      'config_sha256': result['config_sha256'], 'elapsed_s': result['hash_elapsed_s_not_model_inference']}), flush=True)


if __name__ == '__main__':
    main()
