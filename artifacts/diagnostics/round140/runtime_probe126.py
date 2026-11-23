#!/usr/bin/env python3
"""Collect R126 runtime capabilities and source SHAs without loading weights."""
from __future__ import annotations
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

ROOT = Path('/tmp/h3_roofline140')
MODULES = {
    'linear': 'vllm.model_executor.layers.linear',
    'parallel_state': 'vllm_omni.diffusion.distributed.parallel_state',
    'vae': 'vllm_omni.diffusion.models.minimax_h3.vae',
    'vae_dispatch': 'vllm_omni.diffusion.models.minimax_h3.ops.dispatch',
    'vae_ops': 'vllm_omni.diffusion.models.minimax_h3.ops.vae',
    'pipeline': 'vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3',
    'layerwise_backend': 'vllm_omni.diffusion.offloader.layerwise_backend',
    'offload_utils': 'vllm_omni.diffusion.offloader.utils',
    'activation': 'vllm_omni.diffusion.layers.activation',
    'qk_rope': 'vllm_omni.diffusion.layers.fused_qk_norm_rope',
    'indexed_modulation': 'vllm_omni.diffusion.layers.indexed_modulation',
    'attention_910a': 'vllm_omni.diffusion.attention.backends.h3_910a_attn',
    'fa910a': 'vllm_omni.diffusion.attention.backends.fa910a',
}

def schema(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
    if isinstance(value, dict):
        return {str(k): schema(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [schema(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(type(value))

def main():
    import torch
    import torch_npu
    report = {'scope': __doc__, 'completed': False, 'weights_loaded': False,
              'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__,
              'modules': {}, 'fixtures': {}}
    modules = {}
    for label, name in MODULES.items():
        try:
            module = importlib.import_module(name)
            path = Path(inspect.getfile(module)).resolve()
            assert path.suffix == '.py' and str(path).startswith('/home/mseco/')
            data = path.read_bytes()
            saved = ROOT / 'native_sources' / (label + '.py')
            if saved.exists():
                assert saved.read_bytes() == data, ('source changed during preflight', str(saved))
            else:
                saved.write_bytes(data)
            report['modules'][label] = {'import_name': name, 'path': str(path),
                'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                'snapshot': str(saved)}
            modules[label] = module
        except ImportError as exc:
            report['modules'][label] = {'import_name': name, 'unavailable': str(exc)}
            print(json.dumps({'module_import_failure': label, 'reason': str(exc)}), flush=True)
        (ROOT/'runtime_module_imports126.json').write_text(json.dumps(report, indent=2)+'\n')
    dispatch = modules.get('vae_dispatch')
    resolved = dispatch.resolve_h3_vae_operators(torch.device('npu:0')) if dispatch else None
    report['VAE_NPU_dispatch_imported'] = dispatch is not None
    report['VAE_NPU_dispatch_has_operators'] = resolved is not None
    report['VAE_NPU_dispatch_result_type'] = str(type(resolved))
    if resolved is not None:
        report['VAE_NPU_dispatch_operator_names'] = {
            k: getattr(v, '__module__', '') + '.' + getattr(v, '__name__', str(type(v)))
            for k, v in vars(resolved).items()}
    old = Path('/tmp/h3_align12401')
    for path in [old/'final_NPU_latents_pid2331981124.pt', old/'GPU_corrected_pipeline_context124.pt']:
        report['fixtures'][str(path)] = schema(torch.load(path, map_location='cpu', weights_only=True))
    if 'vae' in modules:
        report['VAE_adapter_signatures'] = {
            name: str(inspect.signature(getattr(modules['vae'], name)))
            for name in ['MiniMaxH3VideoVAE', 'MiniMaxH3AudioVAE']}
        report['VAE_video_methods'] = {
            name: str(inspect.signature(getattr(modules['vae'].MiniMaxH3VideoVAE, name)))
            for name in ['decode', 'enable_tiling', 'enable_slicing', 'load_to_device', 'offload_to_cpu']
            if hasattr(modules['vae'].MiniMaxH3VideoVAE, name)}
    report['completed'] = True
    destination = ROOT/'runtime_preflight126.json'
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
