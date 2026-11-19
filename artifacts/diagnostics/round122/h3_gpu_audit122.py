#!/usr/bin/env python3
"""Runtime dtype, finite-value and full-stage audit for the isolated vLLM-Omni FP16 run."""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path('/home/mseco/h3_gpu12201')


def rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.getenv('RANK', '0'))


def record(event: str, **values) -> None:
    out = Path(os.environ.get('H3_GPU_RESULT_ROOT', str(ROOT / 'results' / 'raw_fp16')))
    out.mkdir(parents=True, exist_ok=True)
    row = {'event': event, 'rank': rank(), 'pid': os.getpid(), 'unix': time.time(), **values}
    with (out / f'worker_audit_rank{rank()}.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def synchronize() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def tensor_info(value: torch.Tensor) -> dict:
    value = value.detach()
    info = {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device),
            'elements': value.numel(), 'bytes': value.numel() * value.element_size()}
    if value.is_floating_point() and value.numel():
        finite = torch.isfinite(value)
        info['finite'] = bool(finite.all())
        if info['finite']:
            info['max_abs'] = float(value.abs().max())
        else:
            info['positive_inf'] = int(torch.isposinf(value).sum())
            info['negative_inf'] = int(torch.isneginf(value).sum())
            info['nan'] = int(torch.isnan(value).sum())
    return info


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_leaves(child)


def check_output(stage: str, value):
    tensors = [tensor_info(x) for x in tensor_leaves(value)]
    record('stage_output', stage=stage, tensors=tensors)
    if any(x.get('finite') is False for x in tensors):
        raise FloatingPointError('Original FP16 trial produced nonfinite values at ' + stage)
    return value


def parameter_dtype_audit(pipeline):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MINIMAX_H3_FP32_PARAM_NAMES
    result = {}
    for label in ['transformer', 'text_encoder', 'video_vae', 'audio_vae']:
        module = getattr(pipeline, label)
        counts = {}
        exceptions = []
        for name, parameter in module.named_parameters():
            kind = str(parameter.dtype)
            counts[kind] = counts.get(kind, 0) + parameter.numel()
            expected = torch.float32 if label in {'video_vae', 'audio_vae'} or (label == 'transformer' and name in MINIMAX_H3_FP32_PARAM_NAMES) else torch.float16
            if parameter.dtype != expected:
                exceptions.append({'name': name, 'actual': kind, 'expected': str(expected)})
        result[label] = {'elements_by_dtype': counts, 'unexpected_dtypes': exceptions}
        if exceptions:
            record('dtype_failure', component=label, mismatches=exceptions[:30])
            raise TypeError('FP16 dtype contract violated for ' + label)
    record('component_weight_dtypes', components=result,
           video_audio_VAE_FP32_intentional=True, BF16_parameters_used=False)
    if rank() == 0:
        # Catch a hidden FP32 -> Half -> FP32 roundtrip during eager VAE
        # construction under the engine's default dtype context. A final
        # float32 dtype alone is not proof that the original weights survived.
        from safetensors import safe_open
        for label, checkpoint in [('video_vae', ROOT / 'checkpoint/FL2VA/video_vae/source/model.safetensors'),
                                  ('audio_vae', ROOT / 'checkpoint/FL2VA/audio_vae/model.safetensors')]:
            comparisons = []
            module = getattr(pipeline, label)
            with safe_open(str(checkpoint), framework='pt', device='cpu') as raw:
                keys = set(raw.keys())
                for name, parameter in module.named_parameters():
                    without_remote = name.removeprefix('remote.')
                    candidates = [without_remote, without_remote.removeprefix('model.'), name]
                    original_name = next((key for key in candidates if key in keys), None)
                    if original_name is None:
                        continue
                    if parameter.numel() > 2_000_000:
                        continue
                    expected = raw.get_tensor(original_name)
                    actual = parameter.detach().cpu()
                    equal = bool(torch.equal(actual, expected.to(actual.dtype)))
                    comparisons.append({'name': original_name, 'elements': parameter.numel(),
                                        'original_dtype': str(expected.dtype), 'loaded_dtype': str(actual.dtype),
                                        'exact_original_values_retained': equal})
                    if not equal:
                        record('VAE_weight_roundtrip_failure', component=label, parameter=original_name)
                        raise ValueError('The original VAE parameter changed during dtype loading: ' + original_name)
                    if len(comparisons) >= 8:
                        break
            assert len(comparisons) >= 4, 'Insufficient original VAE parameter matches for ' + label
            record('original_VAE_weight_value_check', component=label, comparisons=comparisons)


def instrument_pipeline(cls):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if getattr(cls, '_h3_gpu_audited122', False):
        return
    cls._h3_gpu_audited122 = True
    original_load = cls.load_weights

    @functools.wraps(original_load)
    def load(self, weights):
        result = original_load(self, weights)
        parameter_dtype_audit(self)
        self._h3_step_count122 = 0
        seen = {}

        def hook(stage, repeat=False):
            def inner(module, args, value):
                n = seen.get(stage, 0)
                seen[stage] = n + 1
                if repeat or n == 0:
                    check_output(stage, value)
            return inner

        self.transformer.condition_proj.register_forward_hook(hook('condition_proj', repeat=True))
        self.transformer.token_refiner.register_forward_hook(hook('token_refiner', repeat=True))
        for i, block in enumerate(self.transformer.blocks):
            block.register_forward_hook(hook(f'dit_block_{i:02d}_first_step'))
        def pre_step(module, args, kwargs):
            self._h3_step_count122 += 1
            synchronize()
            module._h3_step_started122 = time.perf_counter()
            record('dit_step_start', index=self._h3_step_count122,
                   allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        def post_step(module, args, output):
            synchronize()
            elapsed = time.perf_counter() - module._h3_step_started122
            check_output('dit_prediction', output)
            record('dit_step_end', index=self._h3_step_count122, elapsed_s=elapsed)
        self.transformer.register_forward_pre_hook(pre_step, with_kwargs=True)
        self.transformer.register_forward_hook(post_step)
        return result

    cls.load_weights = load
    for name in ['encode_prompt', '_encode_visual_condition', '_encode_audio_condition', 'diffuse', 'decode']:
        original = getattr(cls, name, None)
        if original is None:
            continue
        def make(fn, stage):
            @functools.wraps(fn)
            def wrapped(self, *args, **kwargs):
                synchronize()
                start = time.perf_counter()
                record('stage_start', stage=stage)
                try:
                    result = fn(self, *args, **kwargs)
                    synchronize()
                    elapsed = time.perf_counter() - start
                    check_output(stage, result)
                    record('stage_end', stage=stage, elapsed_s=elapsed,
                           allocated_bytes=torch.cuda.memory_allocated(), peak_bytes=torch.cuda.max_memory_allocated())
                    if stage == 'encode_prompt' and rank() == 0:
                        hidden = result[0].detach().cpu().contiguous()
                        out = Path(os.environ['H3_GPU_RESULT_ROOT']) / 'vllm_fp16_encoder_hidden122.pt'
                        if not out.exists():
                            with out.open('xb') as f:
                                torch.save({'hidden_states': hidden, 'token_tags': result[1].detach().cpu()}, f)
                            record('encoder_capture_saved', file=str(out),
                                   sha256=hashlib.sha256(out.read_bytes()).hexdigest())
                    return result
                except BaseException as exc:
                    record('stage_failure', stage=stage, elapsed_s=time.perf_counter() - start,
                           error_type=type(exc).__name__, message=str(exc))
                    raise
            return wrapped
        setattr(cls, name, make(original, name))
