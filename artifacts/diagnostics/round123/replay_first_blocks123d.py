#!/usr/bin/env python3
"""Short exact-sampling first-two-DiT-block replay, not a full generation.

Uses the successful GPU FP16-matrix/FP32-range-safe source and 1344x768/124-frame
initial sampling bundle. All actual packed rows and the valid-key mask are retained.
The two-block architecture is an explicitly truncated diagnostic. Optional GPU
refiner teacher forcing isolates the main backbone from encoder/refiner drift.
No last-head prediction, full video, speed target or whole-model precision pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--backend', choices=['cuda', 'npu'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--teacher-refiner', default=None)
    args = parser.parse_args()
    root = Path(args.root)
    destination = root / args.label
    destination.mkdir(exist_ok=True)
    import torch
    if args.backend == 'npu':
        import torch_npu
    from safetensors import safe_open
    from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionParallelConfig, AttentionConfig, AttentionSpec, TransformerConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_distributed_env, get_sequence_parallel_rank
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig, get_sp_plan_from_model
    from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
    from vllm_omni.diffusion.forward_context import set_forward_context, get_forward_context
    from h3_replay123.transformer import MiniMaxH3DiTModel
    from h3_replay123.denoise_loop import MiniMaxH3DenoiseBranch
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
              'full_generation_executed': False, 'main_backbone_layers_replayed': [0, 1],
              'GPU_refiner_teacher_forcing': bool(args.teacher_refiner),
              'speed_or_whole_precision_accepted': False, 'captures': {}}
    receipt = destination / f'first_blocks_rank{rank}.json'
    assert not receipt.exists()
    def publish():
        receipt.write_text(json.dumps(report, indent=2) + '\n')
    def synchronize():
        getattr(torch, args.backend).synchronize()
    def save(name, value, sample=False):
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            return
        cpu = value.detach().cpu().contiguous()
        info = {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
                'finite': bool(torch.isfinite(cpu.float()).all()), 'max_abs': float(cpu.float().abs().max()),
                'sha256_raw_tensor_bytes': hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}
        if rank % 2 == 0:
            path = destination / f'{name}_sample_rank{rank}.pt'
            if cpu.ndim == 2 and cpu.shape[0] > 64:
                row_ids = torch.linspace(0, cpu.shape[0] - 1, 64, dtype=torch.float64).round().long().unique()
                values = cpu.index_select(0, row_ids)
                info['sampled_rows_only'] = True
            else:
                row_ids = torch.arange(cpu.shape[0])
                values = cpu
                info['sampled_rows_only'] = False
            with path.open('xb') as f:
                torch.save({'value': values, 'row_indices': row_ids, 'full_shape': list(cpu.shape)}, f)
            info['path'] = str(path)
            info['sample_shape'] = list(values.shape)
            info['error_metric_scope'] = 'selected complete-channel rows; full-output finite/range/hash checks, not full-array error'
        report['captures'][name] = info
        publish()
        if not info['finite']:
            raise FloatingPointError('First-block diagnostic produced nonfinite values at ' + name)
    publish()
    try:
        sampling = json.loads((root / 'same_initial_sampling123.json').read_text())
        bundle_path = root / 'same_initial_sampling123.pt'
        assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == sampling['payload_sha256']
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
        report['exact_sampling_payload_sha256'] = sampling['payload_sha256']
        report['actual_source_sha256'] = hashlib.sha256((root / 'h3_replay123/transformer.py').read_bytes()).hexdigest()
        init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank, distributed_init_method='env://', backend='nccl' if args.backend == 'cuda' else 'hccl')
        initialize_model_parallel(tensor_parallel_size=2, sequence_parallel_size=4, ulysses_degree=4, ring_degree=1, backend='nccl' if args.backend == 'cuda' else 'hccl')
        sp_rank = get_sequence_parallel_rank()
        report['sequence_parallel_rank'] = sp_rank
        architecture = json.loads((Path(args.checkpoint) / 'config.json').read_text())
        assert architecture.get('num_layers', 50) == 50
        architecture = {**architecture, 'num_layers': 2}
        pc = DiffusionParallelConfig(tensor_parallel_size=2, sequence_parallel_size=4, ulysses_degree=4, ring_degree=1)
        od = OmniDiffusionConfig(model=str(Path(args.checkpoint).parent), model_class_name='MiniMaxH3Pipeline', dtype=torch.float16,
                 tf_model_config=TransformerConfig.from_dict(architecture), parallel_config=pc,
                 diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA')), num_gpus=8)
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od), torch.device('cpu'):
            model = MiniMaxH3DiTModel(od, quant_config=None)
        report['actual_SP_attention_strategies'] = [type(block.attn.attention.parallel_strategy).__name__ for block in model.blocks]
        assert all(name=='UlyssesParallelAttention' for name in report['actual_SP_attention_strategies'])
        assert len(model.blocks) == 2
        params = dict(model.named_parameters());params.update(dict(model.named_buffers()))
        index = json.loads((Path(args.checkpoint) / 'model.safetensors.index.json').read_text())['weight_map']
        needed = set(params)
        assert needed <= set(index)
        def weights():
            for filename in sorted({index[n] for n in needed}):
                with safe_open(str(Path(args.checkpoint) / filename), framework='pt', device='cpu') as file:
                    for name in sorted(needed):
                        if index[name] == filename:
                            yield name, file.get_tensor(name)
        loaded = model.load_weights(weights());assert set(loaded) == needed
        model.post_load_weights();model.eval().to(device)
        if args.backend == 'npu':
            from fp32_patch_residual_guard123 import install_patch_projection_guard123
            report['private_precision_guard_modules'] = install_patch_projection_guard123(model)
            report['private_precision_guard_source_sha256'] = hashlib.sha256((root/'fp32_patch_residual_guard123.py').read_bytes()).hexdigest()
            report['original_FP32_patch_weight_tensors_retained'] = True
            report['each_patch_GEMM_actual_input_and_weight_dtype'] = 'torch.float16'
            report['patch_GEMM_actual_output_dtype'] = 'torch.float32'

        teacher = None
        if args.teacher_refiner:
            teacher = torch.load(args.teacher_refiner, map_location='cpu', weights_only=True)['value'].to(device)
            assert teacher.shape == (2043, 5376) and teacher.dtype == torch.float16
            report['teacher_refiner_raw_sha256'] = hashlib.sha256(teacher.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            def return_teacher(module, values, output):
                return teacher
            model.token_refiner.register_forward_hook(return_teacher)
        model.video_patch_proj.register_forward_hook(lambda module, values, output: save('video_patch_proj', output))
        model.audio_patch_proj.register_forward_hook(lambda module, values, output: save('audio_patch_proj', output))
        model.sp_prepare.register_forward_hook(lambda module, values, output: save('initial_packed_embeddings_before_SP', output[0]))
        for i, block in enumerate(model.blocks):
            block.register_forward_pre_hook(lambda module, values, i=i: save(f'block_{i:02d}_input_local_rows', values[0]))
            block.register_forward_hook(lambda module, values, output, i=i: save(f'block_{i:02d}_output_local_rows', output))
        branch = MiniMaxH3DenoiseBranch(packed=bundle['packed'], text_embeddings=bundle['text_embeddings'], token_tags=bundle['token_tags'], device=device)
        # Construct the exact branch kwargs on CPU once, then move the
        # tensors. This avoids 910A's unsupported Boolean IndexPutV2 for
        # timestep assignment without changing any bits or row ordering.
        branch = MiniMaxH3DenoiseBranch(packed=bundle['packed'], text_embeddings=bundle['text_embeddings'], token_tags=bundle['token_tags'], device=torch.device('cpu'))
        fk = branch.forward_kwargs(video_rows=bundle['initial_video_rows_with_keyframes'], audio_rows=bundle['initial_audio_rows'],
                 t_video=0., t_audio=0., imgvid_cond_timestep=0.999, audio_ref_cond_timestep=1.)
        def to_device(v):
            if isinstance(v, torch.Tensor): return v.to(device)
            if isinstance(v, dict): return {k:to_device(x) for k,x in v.items()}
            return v
        fk = to_device(fk)
        report['first_step_unique_timesteps'] = fk['unique_timesteps'].cpu().tolist()
        report['actual_attention_backends'] = [block.attn.attention.attn_backend.get_name() for block in model.blocks]
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od) as ctx, torch.inference_mode():
            apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=4, ring_degree=1), get_sp_plan_from_model(model))
            get_forward_context().sp_plan_hooks_applied = True
            synchronize();started = time.perf_counter()
            output = model(**fk)
            synchronize()
            report['diagnostic_elapsed_s_includes_IO_not_speed'] = time.perf_counter() - started
            report['truncated_final_output_shapes'] = [list(x.shape) for x in output]
        assert report['captures']['block_00_input_local_rows']['shape'][0] == 10448
        assert report['captures']['block_01_output_local_rows']['shape'][0] == 10448
        report['precision_guard_call_counts'] = {name: getattr(getattr(model,name).quant_method,'call_count',None) for name in report.get('private_precision_guard_modules',[])}
        report['completed'] = True
        publish()
        print(json.dumps({'backend': args.backend, 'rank': rank, 'SP_rank': sp_rank, 'completed': True,
                          'same_initial_payload_sha256': sampling['payload_sha256']}, indent=2), flush=True)
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed_env()


if __name__ == '__main__':
    main()
