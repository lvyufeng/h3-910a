#!/usr/bin/env python3
"""Matched TP2 conditioning/refiner-only replay of the exact successful GPU input.

This is an explicitly teacher-forced diagnostic, not a generation/performance pass:
main-DiT blocks are not constructed, and the saved live GPU encoder output is the
same input for both devices. Actual successful H3 source, original checkpoint values,
FP32 range-safe conditioning/residuals and FP16 matrix contracts are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--backend', choices=['cuda', 'npu'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--label', required=True)
    args = p.parse_args()
    root = Path(args.root)
    destination = root / args.label
    destination.mkdir(exist_ok=True)
    import torch
    if args.backend == 'npu':
        import torch_npu
    from safetensors import safe_open
    from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionParallelConfig, AttentionConfig, AttentionSpec
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_distributed_env
    from h3_replay123.transformer import MiniMaxH3DiTModel
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'{args.backend}:{local_rank}')
    if args.backend == 'cuda':
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    else:
        torch.npu.set_device(local_rank)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(4)
    report = {'scope': __doc__, 'completed': False, 'backend': args.backend, 'rank': rank,
              'actual_Generation_executed': False, 'diagnostic_only_no_backbone_blocks': True,
              'same_GPU_teacher_conditioning_input': True, 'shared_runtime_modified': False,
              'numeric_tolerance': None, 'precision_accepted': False, 'captures': {}}
    receipt = destination / f'refiner_replay_rank{rank}.json'
    if receipt.exists():
        raise RuntimeError('Refuse to overwrite a prior replay receipt')
    def publish():
        receipt.write_text(json.dumps(report, indent=2) + '\n')
    def synchronize():
        getattr(torch, args.backend).synchronize()
    def record(name, value):
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            return
        x = value.detach().cpu().contiguous()
        finite = bool(torch.isfinite(x.float()).all())
        info = {'shape': list(x.shape), 'dtype': str(x.dtype), 'finite': finite,
                'max_abs': float(x.float().abs().max()),
                'sha256_raw_tensor_bytes': hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
        if rank in [0, 1]:
            path = destination / f'{name}_rank{rank}.pt'
            if not path.exists():
                with path.open('xb') as f:
                    torch.save({'value': x}, f)
            info['path'] = str(path)
        report['captures'][name] = info
        publish()
        if not finite:
            raise FloatingPointError('Nonfinite stage output: ' + name)
    publish()
    try:
        source = json.loads((root / 'isolated_replay_source123.json').read_text())
        report['source_contract'] = source
        bundle_path = root / 'same_initial_sampling123.pt'
        manifest = json.loads((root / 'same_initial_sampling123.json').read_text())
        assert manifest['completed'] and hashlib.sha256(bundle_path.read_bytes()).hexdigest() == manifest['payload_sha256']
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
        report['sampling_payload_sha256'] = manifest['payload_sha256']
        hidden = bundle['text_embeddings']
        assert hidden.shape == (2043, 5120) and hidden.dtype == torch.float16
        report['actual_input_sha256'] = hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        report['text_tag_counts'] = {str(int(tag)): int((bundle['text_tags'] == tag).sum()) for tag in torch.unique(bundle['text_tags'])}
        init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank, distributed_init_method='env://', backend='nccl' if args.backend == 'cuda' else 'hccl')
        initialize_model_parallel(tensor_parallel_size=2, sequence_parallel_size=4, ulysses_degree=4, ring_degree=1,
                                  backend='nccl' if args.backend == 'cuda' else 'hccl')
        arch = json.loads((Path(args.checkpoint) / 'config.json').read_text())
        assert arch.get('num_layers', 50) == 50
        arch = {**arch, 'num_layers': 0}
        pc = DiffusionParallelConfig(tensor_parallel_size=2, ulysses_degree=4, ring_degree=1, sequence_parallel_size=4)
        ac = AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA' if args.backend == 'cuda' else 'TORCH_SDPA'))
        od = OmniDiffusionConfig(model=str(Path(args.checkpoint).parent), model_class_name='MiniMaxH3Pipeline', dtype=torch.float16,
                                 tf_model_config=arch, parallel_config=pc, diffusion_attention_config=ac, num_gpus=8)
        with set_current_diffusion_config(od), torch.device('cpu'):
            model = MiniMaxH3DiTModel(od, quant_config=None)
        assert len(model.blocks) == 0 and len(model.token_refiner.blocks) == 2
        params = dict(model.named_parameters())
        params.update(dict(model.named_buffers()))
        index = json.loads((Path(args.checkpoint) / 'model.safetensors.index.json').read_text())['weight_map']
        needed = set(params)
        missing = needed - set(index)
        assert not missing, missing
        files = sorted({index[name] for name in needed})
        def weights():
            for filename in files:
                with safe_open(str(Path(args.checkpoint) / filename), framework='pt', device='cpu') as f:
                    for name in sorted(needed):
                        if index[name] == filename:
                            yield name, f.get_tensor(name)
        loaded = model.load_weights(weights())
        assert set(loaded) == needed, (needed - set(loaded), set(loaded) - needed)
        model.post_load_weights()
        model.eval().to(device)
        model.condition_proj.register_forward_hook(lambda module, values, result: record('condition_proj', result))
        for i, block in enumerate(model.token_refiner.blocks):
            block.attn.qkv_proj.register_forward_hook(lambda module, values, result, i=i: record(f'refiner_{i}_qkv', result))
            block.attn.register_forward_hook(lambda module, values, result, i=i: record(f'refiner_{i}_attention', result))
            block.mlp.register_forward_hook(lambda module, values, result, i=i: record(f'refiner_{i}_MLP', result))
            block.register_forward_hook(lambda module, values, result, i=i: record(f'refiner_{i}_residual', result))
        with set_current_diffusion_config(od), torch.inference_mode():
            synchronize()
            started = time.perf_counter()
            projected, _ = model.condition_proj(hidden.to(device=device, dtype=torch.float32))
            refined = model.token_refiner(projected, cu_seqlens=torch.tensor([0, 2043, 2043], dtype=torch.int32, device=device), max_seqlen=2043)
            synchronize()
            report['replay_elapsed_s_with_capture_io_not_performance'] = time.perf_counter() - started
            record('refiner_final', refined)
        report['completed'] = True
        publish()
        print(json.dumps({'backend': args.backend, 'rank': rank, 'completed': True,
                          'input_sha256': report['actual_input_sha256'], 'refiner_final': report['captures']['refiner_final']}, indent=2), flush=True)
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed_env()


if __name__ == '__main__':
    main()
