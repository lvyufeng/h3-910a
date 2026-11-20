#!/usr/bin/env python3
"""One real full-50-layer H3 DiT first step from the exact shared sampling bundle.

This is an isolated vLLM-Omni H3 numerical diagnostic, not full generation or a speed
benchmark. The original checkpoint is read only on each accelerator host. Each
backbone block is loaded through the actual TP-aware weight loader immediately
before its forward, then discarded; CPU model inference is never used. GPU encoder
activations are explicitly teacher-forced. No VAE decode, no 49-step blind run, no
BF16, no clipping, and no local model-weight download or relay.
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
              'DiT_forward_count': 0, 'backbone_layers': 50, 'full_generation_executed': False,
              'GPU_encoder_teacher_forced': True, 'CPU_model_inference_executed': False,
              'shared_runtime_source_modified': False, 'numeric_tolerance': None,
              'whole_model_precision_or_speed_accepted': False,
              'weight_residency': 'front/refiner resident, original backbone weights streamed per block through actual TP loaders; not a production offloader benchmark',
              'captures': {}, 'blocks': {}}
    receipt = destination / f'full_first_step_rank{rank}.json'
    assert not receipt.exists()
    def publish():
        receipt.write_text(json.dumps(report, indent=2) + '\n')
    def synchronize():
        getattr(torch, args.backend).synchronize()
    def statistics(value):
        if value.device.type == 'npu':
            peak = float(value.float().abs().amax(dim=tuple(range(value.ndim))))
        else:
            peak = float(value.float().abs().amax())
        finite = bool(torch.isfinite(value).all())
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'finite': finite, 'max_abs': peak}
    def capture(name, value, full_file=False, sample_rows=64):
        if isinstance(value, tuple):
            value = value[0]
        info = statistics(value)
        if not info['finite']:
            report['captures'][name] = info
            publish()
            raise FloatingPointError('Nonfinite full-step activation at ' + name)
        if rank % 2 == 0:
            cpu = value.detach().cpu().contiguous()
            info['full_tensor_raw_sha256'] = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            if full_file:
                path = destination / f'{name}_full_rank{rank}.pt'
                with path.open('xb') as file:
                    torch.save({'value': cpu}, file)
                info['full_file'] = str(path)
            row_indices = torch.linspace(0, cpu.shape[0] - 1, min(sample_rows, cpu.shape[0]), dtype=torch.float64).round().long().unique()
            payload = {'value': cpu.index_select(0, row_indices), 'row_indices': row_indices, 'full_shape': list(cpu.shape)}
            path = destination / f'{name}_sample_rank{rank}.pt'
            with path.open('xb') as file:
                torch.save(payload, file)
            info['sample_file'] = str(path)
            info['sample_shape'] = list(payload['value'].shape)
            info['sample_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['captures'][name] = info
        publish()
    publish()
    try:
        started = time.perf_counter()
        manifest = json.loads((root / 'same_initial_sampling123.json').read_text())
        bundle_path = root / 'same_initial_sampling123.pt'
        assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == manifest['payload_sha256']
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
        report['exact_sampling_payload_sha256'] = manifest['payload_sha256']
        report['actual_transformer_source_sha256'] = hashlib.sha256((root / 'h3_replay123/transformer.py').read_bytes()).hexdigest()
        backend = 'nccl' if args.backend == 'cuda' else 'hccl'
        init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank, distributed_init_method='env://', backend=backend)
        initialize_model_parallel(tensor_parallel_size=2, sequence_parallel_size=4, ulysses_degree=4, ring_degree=1, backend=backend)
        sp_rank = get_sequence_parallel_rank()
        report['SP_rank'] = sp_rank
        architecture = json.loads((Path(args.checkpoint) / 'config.json').read_text())
        assert architecture['num_layers'] == 50
        pc = DiffusionParallelConfig(tensor_parallel_size=2, sequence_parallel_size=4, ulysses_degree=4, ring_degree=1)
        od = OmniDiffusionConfig(model=str(Path(args.checkpoint).parent), model_class_name='MiniMaxH3Pipeline', dtype=torch.float16,
                                tf_model_config=TransformerConfig.from_dict(architecture), parallel_config=pc,
                                diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA')), num_gpus=8)
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od), torch.device('cpu'):
            model = MiniMaxH3DiTModel(od, quant_config=None)
        assert len(model.blocks) == 50
        all_needed = {name for name, value in model.named_parameters()} | {name for name, value in model.named_buffers()}
        index = json.loads((Path(args.checkpoint) / 'model.safetensors.index.json').read_text())['weight_map']
        assert all_needed <= set(index)
        def weights(selected):
            for filename in sorted({index[name] for name in selected}):
                with safe_open(str(Path(args.checkpoint) / filename), framework='pt', device='cpu') as file:
                    for name in sorted(selected):
                        if index[name] == filename:
                            yield name, file.get_tensor(name)
        front_needed = {name for name in all_needed if not name.startswith('blocks.')}
        loaded_total = model.load_weights(weights(front_needed))
        assert set(loaded_total) == front_needed
        model.post_load_weights()
        model.eval()
        for name, module in model.named_children():
            if name != 'blocks':
                module.to(device)
        if args.backend == 'npu':
            from fp32_patch_residual_guard123 import install_patch_projection_guard123
            report['patch_precision_guard_modules'] = install_patch_projection_guard123(model)
            report['patch_precision_guard_source_sha256'] = hashlib.sha256((root / 'fp32_patch_residual_guard123.py').read_bytes()).hexdigest()
        report['actual_SP_attention_strategies'] = [type(model.blocks[i].attn.attention.parallel_strategy).__name__ for i in [0, 24, 49]]
        assert all(name == 'UlyssesParallelAttention' for name in report['actual_SP_attention_strategies'])
        report['attention_backends'] = [model.blocks[i].attn.attention.attn_backend.get_name() for i in [0, 24, 49]]
        for i, block in enumerate(model.blocks):
            block_names = {name for name in all_needed if name.startswith(f'blocks.{i}.')}
            def before(module, inputs, i=i, selected=block_names):
                report['blocks'][str(i)] = {'started': True, 'completed': False}
                publish()
                layer_started = time.perf_counter()
                loaded = model.load_weights(weights(selected))
                assert set(loaded) == selected
                loaded_total.update(loaded)
                module.to(device)
                report['blocks'][str(i)]['layer_start_counter'] = layer_started
                report['blocks'][str(i)]['parameter_names_loaded'] = sorted(loaded)
                if i in [0, 1, 24, 49]:
                    capture(f'block_{i:02d}_input_local_rows', inputs[0])
            def after(module, inputs, output, i=i):
                stats = statistics(output)
                report['blocks'][str(i)].update(stats)
                if not stats['finite']:
                    publish()
                    raise FloatingPointError(f'Nonfinite real full-model block {i}')
                if i in [0, 1, 24, 49]:
                    capture(f'block_{i:02d}_output_local_rows', output)
                # This diagnostic executes one DiT forward only. Discard each
                # completed block, avoiding 8 resident CPU checkpoint replicas.
                module.to_empty(device='meta')
                synchronize()
                layer_started = report['blocks'][str(i)].pop('layer_start_counter')
                report['blocks'][str(i)]['elapsed_s_includes_load_copy_stats_not_benchmark'] = time.perf_counter() - layer_started
                report['blocks'][str(i)]['completed'] = True
                publish()
            block.register_forward_pre_hook(before)
            block.register_forward_hook(after)
        branch = MiniMaxH3DenoiseBranch(packed=bundle['packed'], text_embeddings=bundle['text_embeddings'], token_tags=bundle['token_tags'], device=torch.device('cpu'))
        fk = branch.forward_kwargs(video_rows=bundle['initial_video_rows_with_keyframes'], audio_rows=bundle['initial_audio_rows'],
                                   t_video=0., t_audio=0., imgvid_cond_timestep=0.999, audio_ref_cond_timestep=1.)
        def to_device(value):
            if isinstance(value, torch.Tensor): return value.to(device)
            if isinstance(value, dict): return {name: to_device(v) for name, v in value.items()}
            return value
        fk = to_device(fk)
        report['first_step_unique_timesteps'] = fk['unique_timesteps'].cpu().tolist()
        report['front_setup_s_not_benchmark'] = time.perf_counter() - started
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od), torch.inference_mode():
            apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=4, ring_degree=1), get_sp_plan_from_model(model))
            get_forward_context().sp_plan_hooks_applied = True
            synchronize();forward_started = time.perf_counter()
            video_velocity, audio_velocity = model(**fk)
            synchronize()
            report['DiT_forward_count'] = 1
            report['full_first_step_s_includes_streaming_capture_not_performance'] = time.perf_counter() - forward_started
            assert set(loaded_total) == all_needed
            assert all(report['blocks'][str(i)]['completed'] for i in range(50))
            assert video_velocity.shape == (39312, 96) and audio_velocity.shape == (414, 32)
            capture('video_velocity_all_rows', video_velocity, full_file=(rank == 0), sample_rows=512)
            capture('audio_velocity_all_rows', audio_velocity, full_file=(rank == 0), sample_rows=414)
            target_positions = torch.nonzero(bundle['packed']['update_mask'], as_tuple=False).flatten().long().to(device)
            capture('video_velocity_target_rows', video_velocity.index_select(0, target_positions), sample_rows=512)
        report['all_parameter_names_loaded_count'] = len(loaded_total)
        report['peak_allocated_accelerator_GiB'] = getattr(torch, args.backend).max_memory_allocated(device) / 1024**3
        report['completed'] = True
        publish()
        print(json.dumps({'backend': args.backend, 'rank': rank, 'SP_rank': sp_rank, 'completed': True,
                          'real_DiT_layers': 50, 'real_DiT_forward_count': 1,
                          'all_block_states_finite': True, 'sampling_payload_sha256': manifest['payload_sha256']}, indent=2), flush=True)
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed_env()


if __name__ == '__main__':
    main()
