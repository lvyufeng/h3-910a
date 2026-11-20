#!/usr/bin/env python3
"""R125 candidate harness: replay identical model inputs on original 910A.

Baseline is the exact R124 executed graph (verified transformer + FP32
patch/time guards + TP2/USP4). Each candidate mode must reproduce the saved
R124 NPU predictions; bit-identical outputs are the only automatic pass. Any
measurable deviation is recorded and the candidate is stopped for user review,
even when it would still fall inside the earlier 1% GPU-comparison tolerance.

This harness loads the original local checkpoint only. No weights are downloaded,
copied or transferred; no CPU model inference is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/tmp/h3_accel12501')
BASELINE_ROOT = Path('/tmp/h3_align12401')
STEPS = [0, 10, 25, 40, 48]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', required=True,
                        choices=['baseline', 'resident', 'static_cache', 'profile', 'naive'])
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    out = ROOT / args.label
    assert out.parent == ROOT
    import torch
    import torch_npu
    if rank == 0:
        assert not out.exists()
        out.mkdir()
    from safetensors import safe_open
    from vllm_omni.diffusion.data import (
        OmniDiffusionConfig, DiffusionParallelConfig, AttentionConfig,
        AttentionSpec, TransformerConfig,
    )
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel, destroy_distributed_env,
    )
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig, get_sp_plan_from_model
    from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
    from vllm_omni.diffusion.forward_context import set_forward_context, get_forward_context
    from h3_replay123.transformer import MiniMaxH3DiTModel
    from h3_replay123.denoise_loop import MiniMaxH3DenoiseBranch
    from fp32_patch_residual_guard123 import install_patch_projection_guard123
    from fp32_time_residual_guard123 import install_time_projection_guard123

    torch.npu.set_device(local_rank)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(4)
    device = torch.device(f'npu:{local_rank}')
    receipt = out / f'candidate_rank{rank}.json'
    report = {
        'scope': __doc__, 'completed': False, 'mode': args.mode, 'rank': rank,
        'baseline_is_committed_R124': True, 'baseline_commit': 'f3f1328',
        'baseline_npu_predictions': str(BASELINE_ROOT / 'NPU_sigma_checks124c'),
        'bit_identical_is_the_only_automatic_lossless_pass': True,
        'no_model_download_or_weight_copy': True, 'CPU_model_inference': False,
        'full_transformer_layers': 50, 'selected_steps': STEPS,
        'matrix_precision_contract': 'FP16 GEMM operands with FP32 wide/out_dtype and preserved original FP32 front/time/final parameters',
        'checks': [],
    }
    if args.mode == 'resident':
        report['candidate_hypothesis'] = (
            'Whole-model residency is a scheduling change only: the same parameter values '
            'are already produced by the per-block pinned CPU staging, so outputs must be bit-identical.'
        )
    elif args.mode == 'baseline':
        report['candidate_hypothesis'] = 'Baseline determinism check: repeated identical execution must reproduce itself bit for bit.'

    def publish():
        receipt.write_text(json.dumps(report, indent=2) + '\n')

    def stats(x):
        cpu = x.detach().cpu()
        return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
                'finite': bool(torch.isfinite(cpu.float()).all()),
                'max_abs': float(cpu.float().abs().amax())}

    def compare(candidate, baseline):
        a = candidate.detach().cpu().contiguous()
        b = baseline.detach().cpu().contiguous()
        assert a.shape == b.shape and a.dtype == b.dtype
        same_bytes = bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        x = a.float().double()
        y = b.float().double()
        diff = x - y
        denom = float(y.square().sum().sqrt().clamp_min(1e-30))
        return {
            'shape': list(a.shape), 'dtype': str(a.dtype),
            'bit_identical': same_bytes,
            'differing_elements': int((a != b).sum()),
            'total_elements': int(a.numel()),
            'relative_RMS_vs_R124_NPU_baseline': float(diff.square().sum().sqrt()) / denom,
            'max_abs_difference': float(diff.abs().max()),
            'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
        }

    publish()
    try:
        bundle_manifest = json.loads((BASELINE_ROOT.parent / 'h3_align12301' / 'same_initial_sampling123.json').read_text())
        bundle_path = BASELINE_ROOT.parent / 'h3_align12301' / 'same_initial_sampling123.pt'
        assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == bundle_manifest['payload_sha256']
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
        report['sigma_layout_bundle_sha256'] = bundle_manifest['payload_sha256']

        init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank,
                                    distributed_init_method='env://', backend='hccl')
        torch.distributed.barrier()
        assert out.is_dir()
        initialize_model_parallel(tensor_parallel_size=2, sequence_parallel_size=4,
                                  ulysses_degree=4, ring_degree=1, backend='hccl')
        pc = DiffusionParallelConfig(tensor_parallel_size=2, sequence_parallel_size=4,
                                     ulysses_degree=4, ring_degree=1)
        arch = json.loads((Path(args.checkpoint) / 'config.json').read_text())
        assert arch['num_layers'] == 50
        od = OmniDiffusionConfig(model=str(Path(args.checkpoint).parent),
                                 model_class_name='MiniMaxH3Pipeline', dtype=torch.float16,
                                 tf_model_config=TransformerConfig.from_dict(arch), parallel_config=pc,
                                 diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA')),
                                 num_gpus=8)
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od), torch.device('cpu'):
            model = MiniMaxH3DiTModel(od, quant_config=None)
        selected = {name for name, _ in model.named_parameters()} | {name for name, _ in model.named_buffers()}
        index = json.loads((Path(args.checkpoint) / 'model.safetensors.index.json').read_text())['weight_map']

        def weights():
            for filename in sorted({index[name] for name in selected}):
                with safe_open(str(Path(args.checkpoint) / filename), framework='pt', device='cpu') as handle:
                    for name in sorted(selected):
                        if index[name] == filename:
                            yield name, handle.get_tensor(name)

        loaded = model.load_weights(weights())
        assert set(loaded) == selected
        model.post_load_weights()
        model.eval()
        report['precision_guard_modules'] = (
            install_patch_projection_guard123(model) + install_time_projection_guard123(model))
        report['parameter_elements_by_dtype'] = {
            str(dtype): sum(p.numel() for p in model.parameters() if p.dtype == dtype)
            for dtype in {p.dtype for p in model.parameters()}
        }

        def onload(module, inputs):
            module.to(device)

        def offload(module, inputs, output):
            module.to('cpu')
            return output

        if args.mode == 'naive':
            from r125_naive import install_naive
            report['naive_replaced_guards'] = install_naive(model)
            report['candidate_is_reference_only_not_accepted'] = True
        if args.mode in ('baseline', 'static_cache', 'profile', 'naive'):
            for name, module in model.named_children():
                if name != 'blocks':
                    module.to(device)
            for block in model.blocks:
                block.register_forward_pre_hook(onload)
                block.register_forward_hook(offload)
            report['per_block_cpu_offload_used'] = True
        else:
            torch.npu.synchronize()
            started = time.perf_counter()
            model.to(device)
            torch.npu.synchronize()
            report['whole_model_resident_move_s'] = time.perf_counter() - started
            report['per_block_cpu_offload_used'] = False
            report['resident_parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())

        branch = MiniMaxH3DenoiseBranch(packed=bundle['packed'], text_embeddings=bundle['text_embeddings'],
                                        token_tags=bundle['token_tags'], device=torch.device('cpu'))
        target = torch.nonzero(bundle['packed']['update_mask'], as_tuple=False).flatten().long()

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(v) for key, v in value.items()}
            return value

        baseline_dir = BASELINE_ROOT / 'NPU_sigma_checks124c'
        static_device_cache = None
        if args.mode == 'static_cache':
            static_device_cache = {}
            report['static_device_tensor_cache'] = True
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od), torch.inference_mode():
            apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=4, ring_degree=1),
                                    get_sp_plan_from_model(model))
            get_forward_context().sp_plan_hooks_applied = True
            profile = args.mode == 'profile'
            steps_to_run = [25] if profile else STEPS
            region_totals = {}
            hooks = []
            if profile:
                def timed(name):
                    state = {'start': None}
                    def pre(module, inputs):
                        torch.npu.synchronize(); state['start'] = time.perf_counter()
                    def post(module, inputs, output):
                        torch.npu.synchronize()
                        region_totals[name] = region_totals.get(name, 0.0) + (time.perf_counter() - state['start'])
                    return pre, post
                for index, block in enumerate(model.blocks):
                    for region, module in (('attn', block.attn), ('mlp', block.mlp), ('adaln', block.adaln_proj),
                                           ('norm1', block.norm1), ('norm2', block.norm2)):
                        pre, post = timed(f'{region}[{0 if index == 0 else (49 if index == 49 else "mid")}]')
                        hooks.append(module.register_forward_pre_hook(pre))
                        hooks.append(module.register_forward_hook(post))
                for name, module in (('token_refiner', getattr(model, 'token_refiner', None)),
                                     ('time_embedder', getattr(model, 'time_embedder', None)),
                                     ('final_layer', getattr(model, 'final_layer', None)),
                                     ('video_patch_proj', getattr(model, 'video_patch_proj', None)),
                                     ('audio_patch_proj', getattr(model, 'audio_patch_proj', None))):
                    if module is None:
                        continue
                    pre, post = timed(name)
                    hooks.append(module.register_forward_pre_hook(pre))
                    hooks.append(module.register_forward_hook(post))
            for step in steps_to_run:
                snapshot = torch.load(BASELINE_ROOT / f'GPU_corrected_pipeline_snapshot_step{step:02d}.pt',
                                      map_location='cpu', weights_only=True)
                assert snapshot['step_index'] == step
                sv, sa = snapshot['sigma_video'].tolist(), snapshot['sigma_audio'].tolist()
                assert sv == bundle['sigmas_video'][step:step + 2].tolist()
                assert sa == bundle['sigmas_audio'][step:step + 2].tolist()
                tv, ta = 1. - sv[0], 1. - sa[0]
                fk = branch.forward_kwargs(video_rows=snapshot['video_rows'], audio_rows=snapshot['audio_rows'],
                                           t_video=tv, t_audio=ta,
                                           imgvid_cond_timestep=max(tv, 0.999),
                                           audio_ref_cond_timestep=max(ta, 1.))
                context = torch.load(BASELINE_ROOT / 'GPU_corrected_pipeline_context124.pt',
                                     map_location='cpu', weights_only=True)['static_forward_kwargs']
                assert torch.equal(context['update_mask'], bundle['packed']['update_mask'])
                fk.update(context)
                fk['unique_timesteps'] = snapshot['unique_timesteps']
                fk['inverse_indices'] = snapshot['inverse_indices']
                dynamic = {'x', 'audio_x', 'unique_timesteps', 'inverse_indices'}
                if static_device_cache is None:
                    fk = move(fk)
                else:
                    # Invariant arguments are uploaded once; only the per-step
                    # state tensors are transferred. No value is altered.
                    if not static_device_cache:
                        for key in fk:
                            if key not in dynamic:
                                static_device_cache[key] = move(fk[key])
                        report['cached_static_keys'] = sorted(static_device_cache)
                        report['cached_static_bytes'] = int(sum(
                            t.numel() * t.element_size() for t in static_device_cache.values()
                            if isinstance(t, torch.Tensor)))
                    per_step = {key: move(fk[key]) for key in dynamic}
                    fk = {**static_device_cache, **per_step}
                torch.npu.synchronize()
                started = time.perf_counter()
                vv, av = model(**fk)
                torch.npu.synchronize()
                elapsed = time.perf_counter() - started
                row = {'step_index': step, 'sigma_video': sv[0], 'sigma_audio': sa[0],
                       'elapsed_s_warm_after_first_step': elapsed,
                       'video': stats(vv), 'audio': stats(av)}
                if not (row['video']['finite'] and row['audio']['finite']):
                    raise FloatingPointError(f'nonfinite prediction at step {step}')
                if rank == 0:
                    base = torch.load(baseline_dir / f'NPU_prediction_step{step:02d}.pt',
                                      map_location='cpu', weights_only=True)
                    assert base['step_index'] == step
                    row['video_vs_R124_baseline'] = compare(vv.cpu().index_select(0, target),
                                                            base['video_velocity'].index_select(0, target))
                    row['audio_vs_R124_baseline'] = compare(av.cpu(), base['audio_velocity'])
                    row['bit_identical_to_R124_baseline'] = (
                        row['video_vs_R124_baseline']['bit_identical'] and row['audio_vs_R124_baseline']['bit_identical'])
                    path = out / f'candidate_prediction_step{step:02d}.pt'
                    with path.open('xb') as handle:
                        torch.save({'step_index': step, 'video_velocity': vv.cpu(),
                                    'audio_velocity': av.cpu()}, handle)
                    row['candidate_prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    print(json.dumps({'candidate_step': row}, ensure_ascii=False), flush=True)
                if profile:
                    row['region_seconds'] = {k: round(v, 3) for k, v in sorted(region_totals.items(),
                                              key=lambda kv: -kv[1]) if v > 0.001}
                    region_totals.clear()
                report['checks'].append(row)
                publish()
                if args.mode == 'naive':
                    # Saturation is the expected outcome; record it and stop this mode.
                    break
        report['completed'] = True
        if rank == 0:
            report['all_steps_bit_identical_to_R124_baseline'] = all(
                row['bit_identical_to_R124_baseline'] for row in report['checks'])
            warm = [row['elapsed_s_warm_after_first_step'] for row in report['checks'][1:]]
            report['warm_step_median_s'] = sorted(warm)[len(warm) // 2] if warm else None
            report['warm_step_min_s'] = min(warm)
            report['warm_step_max_s'] = max(warm)
        publish()
        if rank == 0:
            print(json.dumps({'candidate_label': args.label, 'mode': args.mode,
                              'all_steps_bit_identical': report.get('all_steps_bit_identical_to_R124_baseline'),
                              'warm_step_median_s': report.get('warm_step_median_s')}, indent=2), flush=True)
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed_env()


if __name__ == '__main__':
    main()