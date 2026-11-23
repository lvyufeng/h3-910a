#!/usr/bin/env python3
"""R126 warm, rank-max candidate harness: replay identical model inputs on original 910A.

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

ROOT = Path('/tmp/h3_ev147')
BASELINE_ROOT = Path('/tmp/h3_align12401')
STEPS = [0, 10, 25, 40, 48]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', required=True,
                        choices=['baseline', 'resident', 'static_cache', 'profile', 'naive', 'half_res', 'fused_mlp'])
    parser.add_argument('--label', required=True)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--usp', type=int, default=4)
    parser.add_argument('--resident', action='store_true')
    parser.add_argument('--audit', choices=['strict','deferred'], default='strict')
    parser.add_argument('--audit-sequence', default='strict,bit_fast,strict')
    parser.add_argument('--offload', choices=['native','legacy'], default='native')
    parser.add_argument('--steps', default='25')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    STEPS = [int(x) for x in args.steps.split(',')]
    assert STEPS and set(STEPS) <= {0,10,25,40,48}
    assert args.warmup >= 1 and args.repeats >= 1
    audit_modes = args.audit_sequence.split(',')
    assert set(audit_modes) <= {'strict','deferred','device','bit_fast'}
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    out = ROOT / args.label
    assert out.parent == ROOT
    import torch
    import torch_npu
    assert out.is_dir(), 'Launcher must create a fresh output directory once before torchrun'
    assert not (out / f'candidate_rank{rank}.json').exists(), 'Refuse to overwrite a prior rank receipt'
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
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'equivalent_safety_source_sha256':hashlib.sha256((ROOT/'equivalent_safety126.py').read_bytes()).hexdigest(),
        'safety_check_count_and_strict_abort_placement_preserved':True,
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

        tp, usp = args.tp, args.usp
        assert tp * usp == 8 and tp >= 1 and usp >= 1, (tp, usp)
        report['parallel_config'] = {'tensor_parallel_size': tp, 'ulysses_degree': usp, 'ring_degree': 1}
        init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank,
                                    distributed_init_method='env://', backend='hccl')
        torch.distributed.barrier()
        assert out.is_dir()
        initialize_model_parallel(tensor_parallel_size=tp, sequence_parallel_size=usp,
                                  ulysses_degree=usp, ring_degree=1, backend='hccl')
        pc = DiffusionParallelConfig(tensor_parallel_size=tp, sequence_parallel_size=usp,
                                     ulysses_degree=usp, ring_degree=1)
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
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tp_group
        report['tp_actual'] = get_tensor_model_parallel_world_size()
        report['tp_group_ranks'] = list(get_tp_group().ranks)
        assert report['tp_actual'] == args.tp
        report['local_heads'] = model.blocks[0].attn.num_heads
        assert report['local_heads'] * args.tp == arch['num_attention_heads']
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

        if args.mode == 'fused_mlp':
            import h3_replay123.transformer as replay_transformer
            from r125_fused_mlp import install_fused_mlp
            report['fused_mlp_variant'] = install_fused_mlp(replay_transformer)
        if args.mode == 'half_res':
            import h3_replay123.transformer as replay_transformer
            from r125_half_residual import install_half_residual
            report['half_residual_variant'] = install_half_residual(replay_transformer)
        if args.mode == 'naive':
            from r125_naive import install_naive
            report['naive_replaced_guards'] = install_naive(model)
            report['candidate_is_reference_only_not_accepted'] = True
        if not args.resident:
            for name, module in model.named_children():
                if name != 'blocks':
                    module.to(device)
            if args.offload == 'native':
                from vllm_omni.diffusion.offloader.layerwise_backend import _install_layerwise_hook_group
                copy_stream = torch.npu.Stream()
                native_hooks = _install_layerwise_hook_group(model.blocks, device, copy_stream, True)
                native_hooks[0].prefetch_layer(non_blocking=False)
                report['offload_kind'] = 'native immutable pinned masters, flattened dtype copies, asynchronous next-block prefetch'
                report['host_weight_bytes'] = sum(t.numel()*t.element_size() for h in native_hooks for t in h.dtype_cpu_flattened_weights.values())
                report['D2H_weight_copies_after_each_layer'] = False
            else:
                for block in model.blocks:
                    block.register_forward_pre_hook(onload)
                    block.register_forward_hook(offload)
                report['offload_kind'] = 'legacy per-block module.to NPU then module.to CPU'
                report['D2H_weight_copies_after_each_layer'] = True
            report['per_block_cpu_offload_used'] = True
        else:
            torch.npu.synchronize()
            started = time.perf_counter()
            model.to(device)
            torch.npu.synchronize()
            report['whole_model_resident_move_s'] = time.perf_counter() - started
            report['per_block_cpu_offload_used'] = False
            report['resident_parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())

        from equivalent_safety126 import EquivalentMatrixSafetyAudit126 as MatrixSafetyAudit126
        audit = MatrixSafetyAudit126(model, mode=args.audit)
        report['safety_mode'] = args.audit
        report['warmup_each_selected_step'] = args.warmup
        report['clean_repeats_each_selected_step'] = args.repeats
        report['numerical_changes_are_not_automatically_accepted'] = True
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
            if usp > 1:
                apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=usp, ring_degree=1),
                                        get_sp_plan_from_model(model))
                get_forward_context().sp_plan_hooks_applied = True
            else:
                report['sequence_parallel_hooks'] = 'skipped because ulysses_degree=1'
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
            warmed_steps = set()
            measured_repeats = {}
            active_case = None
            active_device_state = None
            active_changed = []
            blocked_cases = set()
            from device_range126 import install_device_range, restore_device_range, install_leaf_proof
            for case_index, audit_mode, step in [(i,m,s) for i,m in enumerate(audit_modes) for s in steps_to_run for _ in range(args.repeats)]:
                repeat_key = (case_index, step)
                repeat = measured_repeats.get(repeat_key, 0)
                measured_repeats[repeat_key] = repeat + 1
                if case_index in blocked_cases:
                    continue
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
                if active_case != case_index:
                    if active_changed:
                        restore_device_range(active_changed)
                        active_changed = []
                        active_device_state = None
                    audit.mode = 'strict' if audit_mode in ('strict','bit_fast') else 'deferred'
                    audit.fast = audit_mode=='bit_fast'
                    if audit_mode == 'device':
                        proof_state, proof_changed, proof_rows = install_leaf_proof(model)
                        proof_state.begin();audit.begin()
                        proof_v, proof_a = model(**fk)
                        audit.end();proof_state.end();torch.npu.synchronize()
                        restore_device_range(proof_changed)
                        report['device_scale_leaf_proof'] = proof_rows
                        local_fail = int(any(not r['bit_identical'] or not r['scale_identical'] for r in proof_rows) or len(proof_rows)!=104)
                        flag=torch.tensor([local_fail],dtype=torch.int32,device=device)
                        torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
                        if int(flag.cpu()):
                            blocked_cases.add(case_index)
                            report['device_scale_blocked_for_user_review'] = True
                            publish();continue
                        del proof_v, proof_a
                        active_device_state, active_changed = install_device_range(model)
                        report['device_scale_all104_leaves_bit_identical'] = True
                    active_case = case_index
                if repeat_key not in warmed_steps:
                    for _ in range(args.warmup):
                        audit.begin()
                        if active_device_state is not None:active_device_state.begin()
                        warm_v, warm_a = model(**fk)
                        audit.end()
                        if active_device_state is not None:active_device_state.end()
                        torch.npu.synchronize()
                        assert bool(torch.isfinite(warm_v.cpu()).all() and torch.isfinite(warm_a.cpu()).all())
                        del warm_v, warm_a
                    warmed_steps.add(repeat_key)
                torch.npu.synchronize()
                torch.distributed.barrier()
                audit.begin()
                if active_device_state is not None:active_device_state.begin()
                started = time.perf_counter()
                vv, av = model(**fk)
                safety = audit.end()
                if active_device_state is not None:safety.update(active_device_state.end())
                torch.npu.synchronize()
                elapsed = time.perf_counter() - started
                local_elapsed = elapsed
                elapsed_tensor = torch.tensor([elapsed],dtype=torch.float32,device=device)
                torch.distributed.all_reduce(elapsed_tensor,op=torch.distributed.ReduceOp.MAX)
                rank_max = float(elapsed_tensor.cpu())
                row = {'case_index': case_index, 'audit_mode':audit_mode, 'step_index': step, 'repeat_index': repeat, 'sigma_video': sv[0], 'sigma_audio': sa[0],
                       'local_elapsed_s':local_elapsed, 'rank_max_elapsed_s':rank_max, 'safety':safety,
                       'peak_allocated_bytes':torch.npu.max_memory_allocated(device),
                       'elapsed_s_after_explicit_warmup': elapsed,
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
                    path = out / f'prediction_case{case_index}_step{step:02d}_repeat{repeat}.pt'
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
                if repeat==args.repeats-1:
                    audit.profile=True;audit.begin()
                    profile_v,profile_a=model(**fk)
                    audit.end();torch.npu.synchronize()
                    report['equivalent_safety_profile_'+str(case_index)]=audit.profile_receipt()
                    audit.profile=False
                    del profile_v,profile_a
                    publish()
                if args.mode == 'naive':
                    # Saturation is the expected outcome; record it and stop this mode.
                    break
        report['completed'] = True
        if rank == 0:
            report['all_steps_bit_identical_to_R124_baseline'] = all(
                row['bit_identical_to_R124_baseline'] for row in report['checks'])
            import statistics
            warm = [row['rank_max_elapsed_s'] for row in report['checks']]
            report['warm_step_median_s'] = statistics.median(warm) if warm else None
            report['warm_step_by_sigma'] = {str(s):statistics.median([row['rank_max_elapsed_s'] for row in report['checks'] if row['step_index']==s]) for s in STEPS}
            report['warm_step_by_case'] = {str(i)+':'+m:statistics.median([row['rank_max_elapsed_s'] for row in report['checks'] if row['case_index']==i]) for i,m in enumerate(audit_modes) if any(row['case_index']==i for row in report['checks'])}
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