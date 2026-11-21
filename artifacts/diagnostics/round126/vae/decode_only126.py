#!/usr/bin/env python3
"""R126 same-final-latent decode-only driver; model execution is explicit.

No Torch import occurs for --plan-only/--validate-only or stdlib tests. Lead alone
runs --execute on its leased host. No DiT, encoder forward, network/weight copy,
old R124 alignment hooks, parameter narrowing, or blanket lossless acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import datetime
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

from contracts126 import (
    AUDIO_LATENT_SHAPE, ContractError, FORBIDDEN_ENV, LATENTS_SHA256,
    MAX_OUTPUT_BYTES, VIDEO_LATENT_SHAPE, WORLD_SIZE, atomic_json,
    check_local_files, function_identity, is_baseline_receipt, sha_file, validate_cli, validate_manifest,
)


class CandidateDelta(RuntimeError):
    pass


class ParameterRewriteRejected(RuntimeError):
    pass


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Original FL2VA root containing video_vae/audio_vae")
    p.add_argument("--latents", required=True, help="Exact accepted-FA final latent bundle")
    p.add_argument("--root", required=True, help="Lead's private output root")
    p.add_argument("--label", required=True, help="New exclusive run name")
    p.add_argument("--runtime-manifest", help="SHA-pinned JSON facts collected by lead")
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--validate-only", action="store_true")
    actions.add_argument("--plan-only", action="store_true")
    p.add_argument("--installer-mode", choices=("preserve-fp32", "observe-native"), default="observe-native")
    p.add_argument("--reference-dir", help="Completed preserve-fp32 run for actual candidate comparison")
    p.add_argument("--warm-repeats", type=int, default=3)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--skip-region-profile", action="store_true")
    p.add_argument("--skip-tile-capture", action="store_true")
    p.add_argument("--no-save-reference", action="store_true")
    p.add_argument("--output-limit-bytes", type=int, default=MAX_OUTPUT_BYTES)
    return p


def planned(args):
    return {"scope": "Decode only; no model execution unless --execute", "execution_requested": bool(args.execute),
            "checkpoint": args.checkpoint, "latents": args.latents, "latents_sha256": LATENTS_SHA256,
            "run": str(Path(args.root) / args.label), "installer_mode": args.installer_mode,
            "parameter_contract": "original FP32 video/audio, Half video autocast, no direct parameter rewrite",
            "cases": ["video_clean_cold_plus_warm", "video_engagement_regions", "video_tile_capture_nonclean",
                      "audio_clean_cold_plus_warm", "audio_engagement_regions"],
            "timing": "clean excludes hooks/counters/copies/hash; instrumented cases reported separately",
            "native_installer": "preserve-fp32 records bypass with native_return=null; observe-native calls actual installer and rejects dtype rewrite before decode",
            "comparison": "full CPU raw outputs; self-repeat != candidate acceptance; native installed candidate requires external preserve-fp32 reference",
            "output_limit_bytes": args.output_limit_bytes, "accepted_lossless": False}


def source_identity(function):
    unbound = inspect.unwrap(getattr(function, "__func__", function))
    result = function_identity(unbound)
    try:
        path = Path(inspect.getfile(unbound)).resolve(strict=True)
        result.update(path=str(path), sha256=sha_file(path))
    except (TypeError, OSError):
        result.update(path=None, sha256=None)
    return result


def tensor_receipt(torch, value):
    cpu = value.detach().cpu().contiguous()
    raw = memoryview(cpu.view(torch.uint8).numpy())
    import hashlib
    flat = cpu.reshape(-1)
    finite = all(bool(torch.isfinite(flat[i:i + 1024 * 1024]).all()) for i in range(0, flat.numel(), 1024 * 1024))
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "stride": list(cpu.stride()),
            "elements": cpu.numel(), "bytes": cpu.numel() * cpu.element_size(), "finite_CPU": finite,
            "sha256": hashlib.sha256(raw).hexdigest()}


def compare_cpu(torch, value, reference):
    if value.device.type != "cpu" or reference.device.type != "cpu":
        raise ContractError("Comparison must be outside the timed graph on CPU")
    if value.shape != reference.shape or value.dtype != reference.dtype:
        raise ContractError("Output shape/dtype differs from the complete reference")
    x, y = value.contiguous().reshape(-1), reference.contiguous().reshape(-1)
    different, different_bytes, squared_error, squared_ref, absolute, maximum = 0, 0, 0.0, 0.0, 0.0, 0.0
    finite = True
    # Bound Float64 scratch; do not create whole-video doubles.
    for start in range(0, x.numel(), 1024 * 1024):
        a, b = x[start:start + 1024 * 1024], y[start:start + 1024 * 1024]
        different += int(a.ne(b).sum())
        different_bytes += int(a.view(torch.uint8).ne(b.view(torch.uint8)).sum())
        chunk_finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        finite = finite and chunk_finite
        if chunk_finite:
            af, bf = a.to(torch.float64), b.to(torch.float64)
            error = af - bf
            absolute += float(error.abs().sum())
            maximum = max(maximum, float(error.abs().max()))
            squared_error += float(error.square().sum())
            squared_ref += float(bf.square().sum())
    return {"shape": list(value.shape), "dtype": str(value.dtype), "elements": x.numel(),
            "raw_bits_equal": different_bytes == 0, "different_values": different, "different_bytes": different_bytes,
            "finite_both": finite, "max_abs_error": maximum if finite else None,
            "mean_abs_error": absolute / x.numel() if finite else None,
            "relative_rms": math.sqrt(squared_error / max(squared_ref, 1e-60)) if finite else None,
            "accepted_lossless": False}


def parameters(module):
    rows = list(module.parameters())
    return {"parameter_tensors": len(rows), "dtypes": dict(Counter(str(x.dtype) for x in rows)),
            "elements_by_dtype": {str(dtype): sum(x.numel() for x in rows if x.dtype == dtype) for dtype in {x.dtype for x in rows}},
            "bytes": sum(x.numel() * x.element_size() for x in rows),
            "devices": dict(Counter(str(x.device) for x in rows)),
            "inference_tensors": dict(Counter(str(x.is_inference()) for x in rows))}


def all_float_fp32(torch, module):
    return all(not x.is_floating_point() or x.dtype == torch.float32 for x in module.parameters())


class Journal:
    """Instance-only modifications, restored even after capture/profiling failure."""
    def __init__(self):
        self.entries = []

    def bind(self, owner, name, replacement):
        prior = owner.__dict__.get(name, _MISSING)
        setattr(owner, name, replacement)
        self.entries.append((owner, name, prior, replacement))

    def close(self):
        errors = []
        for owner, name, prior, installed in reversed(self.entries):
            if owner.__dict__.get(name, _MISSING) is not installed:
                errors.append("Unknown replacement after install: " + name)
                continue
            if prior is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, prior)
        self.entries.clear()
        if errors:
            raise ContractError("; ".join(errors))


_MISSING = object()


class EventRegions:
    """Finite preallocated event pool; no sync or CPU transfer inside callbacks."""
    NAMES = ("decoder_tiles", "post_quant_conv", "tile_gather_helper", "blend", "denormalize", "revert", "temporal_streaming")

    def __init__(self, torch, slots, names=None):
        self.torch, self.slots = torch, slots
        self.names = tuple(names) if names is not None else self.NAMES
        self.events = {name: [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(slots)] for name in self.names}
        self.counts = Counter()
        self.overflows = Counter()
        self.host = {name: [] for name in self.names}
        self.journal = Journal()
        # Event lazy allocation also occurs outside any measured call.
        for pairs in self.events.values():
            for start, end in pairs:
                start.record(); end.record()
        torch.npu.synchronize()

    def bind(self, owner, method, region):
        original = getattr(owner, method)
        def wrapped(*args, **kwargs):
            index = self.counts[region]
            self.counts[region] += 1
            if index >= self.slots:
                self.overflows[region] += 1
                # Do not strand other ranks midway through a native collective.
                return original(*args, **kwargs)
            start, end = self.events[region][index]
            begin = time.perf_counter()
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                self.host[region].append(time.perf_counter() - begin)
        self.journal.bind(owner, method, wrapped)

    def install_video(self, vae):
        for owner, method, region in ((vae.model.decoder, "forward", "decoder_tiles"),
                                      (vae.model.post_quant_conv, "forward", "post_quant_conv"),
                                      (vae.model, "_all_gather_tiled_results", "tile_gather_helper"),
                                      (vae.model, "blend", "blend"),
                                      (vae, "_denormalize_latent", "denormalize"),
                                      (vae, "_revert_decoded_inplace", "revert"),
                                      (vae.model, "_decode_temporal_streaming", "temporal_streaming")):
            if callable(getattr(owner, method, None)):
                self.bind(owner, method, region)

    def receipt(self):
        result = {"inclusive_regions_not_additive": True,
                  "tile_gather_helper_excludes_tiled_decode_barrier_outside_helper": True,
                  "no_module_synchronize_or_CPU_copies": True, "preallocated_slots_per_region": self.slots, "regions": {}}
        for name in self.names:
            count = min(self.counts[name], self.slots)
            streams = [start.elapsed_time(end) / 1000 for start, end in self.events[name][:count]]
            result["regions"][name] = {"calls": self.counts[name], "overflow": self.overflows[name],
                                      "current_stream_event_s": streams, "inclusive_current_stream_sum_s": sum(streams),
                                      "host_inclusive_wall_s": self.host[name], "host_inclusive_sum_s": sum(self.host[name])}
        if self.overflows:
            result["incomplete_profile"] = True
        return result

    def close(self):
        self.journal.close()


class Engagement:
    """Observe actual fast method/operator entries, separately from clean timing."""
    METHODS = (("attn", "_optimized_attention"), ("ff", "_optimized_feed_forward"), (None, "_optimized_transformer_block"))

    def __init__(self, decoder, optimizations):
        self.decoder, self.optimizations = decoder, optimizations
        self.journal, self.counts, self.successes, self.inputs = Journal(), Counter(), Counter(), Counter()
        self.bindings = []

    def snapshot(self):
        rows = []
        for index, block in enumerate(self.decoder.transformer_blocks):
            row = {"block": index, "methods": {}, "operators": {}}
            for child, function in self.METHODS:
                owner = getattr(block, child) if child else block
                actual = inspect.unwrap(getattr(owner.forward, "__func__", owner.forward))
                expected = getattr(self.optimizations, function, None)
                row["methods"][function] = {"actual": source_identity(actual), "matches_native_fast_function": expected is not None and actual is inspect.unwrap(expected)}
            for owner, name in ((block.attn, "_omni_qk_norm_rope"), (block, "_omni_scaled_residual")):
                value = getattr(owner, name, None)
                row["operators"][name] = source_identity(value) if callable(value) else None
            fused = getattr(block.ff, "_omni_silu_and_mul", None)
            row["operators"]["_omni_silu_and_mul"] = {"type": type(fused).__name__,
                "forward": source_identity(fused.forward) if fused is not None else None,
                "cached_dispatch": source_identity(fused._forward_method) if callable(getattr(fused, "_forward_method", None)) else None}
            rows.append(row)
        self.bindings = rows
        return {"installed_marker": bool(getattr(self.decoder, "_omni_h3_vae_optimizations_installed", False)), "blocks": rows}

    def count_method(self, owner, name, label, success=False):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            self.counts[label] += 1
            if args and hasattr(args[0], "dtype"):
                self.inputs[label + ":" + str(args[0].dtype)] += 1
                self.inputs[label + ":is_cuda=" + str(bool(args[0].is_cuda))] += 1
            value = original(*args, **kwargs)
            if success and value is not None:
                self.successes[label] += 1
            return value
        self.journal.bind(owner, name, wrapped)

    def install(self):
        if not self.bindings:
            self.snapshot()
        for index, block in enumerate(self.decoder.transformer_blocks):
            for child, function in self.METHODS:
                owner = getattr(block, child) if child else block
                if self.bindings[index]["methods"][function]["matches_native_fast_function"]:
                    self.count_method(owner, "forward", function)
            for owner, attr in ((block.attn, "_omni_qk_norm_rope"), (block, "_omni_scaled_residual")):
                if callable(getattr(owner, attr, None)):
                    self.count_method(owner, attr, attr, success=True)
            fused = getattr(block.ff, "_omni_silu_and_mul", None)
            if fused is not None and callable(getattr(fused, "forward", None)):
                self.count_method(fused, "forward", "_omni_silu_and_mul.forward", success=True)
                if callable(getattr(fused, "_forward_method", None)):
                    self.count_method(fused, "_forward_method", "_omni_silu_and_mul.cached_dispatch", success=True)

    def receipt(self):
        labels = [name for _, name in self.METHODS] + ["_omni_qk_norm_rope", "_omni_scaled_residual", "_omni_silu_and_mul.forward", "_omni_silu_and_mul.cached_dispatch"]
        return {"fast_function_entry_counts": {name: self.counts[name] for name in labels},
                "non_None_operator_return_counts": {name: self.successes[name] for name in labels[3:]},
                "input_metadata_counts": dict(self.inputs), "clean_timer_contains_these_counters": False,
                "entry_is_not_proof_of_CUDA_only_guard_success": True,
                "native_NPU_fusion_proven": False,
                "note": "Read actual binding identity, operator non-None returns and pinned source guards; do not infer kernel launch from an installed marker or native-return=True."}

    def close(self):
        self.journal.close()


class TileCapture:
    """A separate, explicitly nonclean pre-quantization full-tile capture call."""
    def __init__(self, decoder, maximum_tiles, maximum_bytes):
        self.decoder, self.maximum_tiles, self.maximum_bytes = decoder, maximum_tiles, maximum_bytes
        self.values, self.bytes, self.overflow = [], 0, False
        self.journal = Journal()

    def install(self):
        original = self.decoder.forward
        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            size = output.numel() * output.element_size()
            if len(self.values) >= self.maximum_tiles or self.bytes + size > self.maximum_bytes:
                self.overflow = True
            else:
                self.values.append(output.detach().clone())
                self.bytes += size
            return output
        self.journal.bind(self.decoder, "forward", wrapped)

    def receipt(self, torch, external):
        rows = []
        for index, value in enumerate(self.values):
            row = {"tile_index": index, **tensor_receipt(torch, value)}
            if external is not None:
                if index >= len(external):
                    row["reference_match"] = False
                else:
                    ref = external[index]
                    row["reference_match"] = all(row[key] == ref[key] for key in ("shape", "dtype", "elements", "sha256"))
            rows.append(row)
        return {"scope": "Entire 36-layer decoder output for each locally owned tile before gather/blend/quantization; not full prequantized global video",
                "extra_GPU_clone_bytes": self.bytes, "tiles": rows, "complete_capture": not self.overflow,
                "expected_reference_count_match": external is None or len(external) == len(rows),
                "all_reference_tiles_raw_equal": None if external is None else not self.overflow and len(external) == len(rows) and all(row["reference_match"] for row in rows),
                "timing_is_nonclean": True, "no_callback_CPU_copy_or_synchronize": True}

    def close(self):
        self.journal.close()


@contextmanager
def observed_installer(torch, adapter, mode, record):
    native = adapter.install_h3_vae_optimizations
    def call(decoder, *, device):
        record["calls"] += 1
        record["before"] = parameters(decoder)
        record["native_function"] = source_identity(native)
        if not all_float_fp32(torch, decoder):
            raise ParameterRewriteRejected("Loader did not construct original FP32 decoder parameters")
        if mode == "preserve-fp32":
            result = False
            record.update(native_called=False, native_return=None, returned_to_adapter=False,
                          meaning="Explicit R124 parameter-preserving bypass; NOT an observed native False")
        else:
            record["native_called"] = True
            try:
                result = native(decoder, device=device)
                if type(result) is not bool:
                    raise ContractError("Unexpected native installer return schema")
                record.update(native_return=result, returned_to_adapter=result)
            finally:
                record["after"] = parameters(decoder)
                record["all_float_parameters_fp32_after"] = all_float_fp32(torch, decoder)
        record["after"] = parameters(decoder)
        record["all_float_parameters_fp32_after"] = all_float_fp32(torch, decoder)
        if not record["all_float_parameters_fp32_after"]:
            record["forward_executed_after_parameter_rewrite"] = False
            raise ParameterRewriteRejected("Native installer changed original FP32 parameters; stop before decode, never widen to pretend restoration")
        return result
    adapter.install_h3_vae_optimizations = call
    try:
        yield
    finally:
        adapter.install_h3_vae_optimizations = native


def verify_native_sources(manifest, video, audio):
    attention = video.model.decoder.transformer_blocks[0].attn
    blocks = video.model.decoder.transformer_blocks[0]
    attention_forward = inspect.unwrap(type(attention).forward)
    flash = inspect.unwrap(attention_forward.__globals__["flash_attn"])
    rotary = inspect.unwrap(attention_forward.__globals__["apply_rotary_pos_emb"])
    native_model_module = importlib.import_module(type(video.model).__module__)
    parallel = importlib.import_module(native_model_module.get_parallel_state.__module__)
    functions = {"native_video": type(video.model), "native_decoder": type(video.model.decoder),
                 "native_attention": type(attention), "native_blocks": type(blocks), "native_func": rotary,
                 "native_flash": flash, "native_parallel": parallel, "native_audio": type(audio.remote)}
    result = {}
    for role, value in functions.items():
        path = Path(inspect.getfile(value)).resolve(strict=True)
        digest = sha_file(path)
        if digest != manifest["native_source_sha256"][role]:
            raise ContractError("Dynamic/native source changed: " + role)
        result[role] = {"path": str(path), "sha256": digest}
    return result


def run(args, manifest):
    # All file/config/source validation precedes any Torch import.
    check_local_files(manifest, args.checkpoint, args.latents)
    for key in FORBIDDEN_ENV:
        os.environ.pop(key, None)
    os.environ.update(manifest["environment"])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", H3_NPU_ALIGNMENT124_BOOTSTRAP="0",
                      H3_NPU_SAVE_FINAL_LATENTS124="0", H3_NPU_CAPTURE_NATIVE_MEDIA124="0",
                      VLLM_OMNI_910A_NO_FUSED_ACT="1")
    compat_dir = Path(manifest["nan_compat_dir"]).resolve(strict=True)
    sys.path.insert(0, str(compat_dir))
    import torch
    import torch_npu
    from vllm_omni.platforms import current_omni_platform
    if torch.__version__ != manifest["torch_version"] or torch_npu.__version__ != manifest["torch_npu_version"]:
        raise ContractError("Torch/torch_npu version differs from the pinned runtime")
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
    if world != WORLD_SIZE or rank != local or not 0 <= rank < WORLD_SIZE:
        raise ContractError("Only lead's single-node eight-rank LOCAL_RANK=device mapping is supported")
    current_omni_platform.set_device(torch.device("npu", local))
    torch.set_num_threads(args.cpu_threads)
    torch.npu.matmul.allow_hf32 = False; torch.npu.conv.allow_hf32 = False
    adapter = importlib.import_module(manifest["modules"]["adapter"]["import"])
    ops = importlib.import_module(manifest["modules"]["optimizations"]["import"])
    compat = importlib.import_module(manifest["modules"]["nan_compat"]["import"])
    for role in ("adapter", "optimizations", "dispatch", "sdpa_shim", "nan_compat", "activation"):
        module = importlib.import_module(manifest["modules"][role]["import"])
        if sha_file(Path(inspect.getfile(module)).resolve(strict=True)) != manifest["modules"][role]["sha256"]:
            raise ContractError("Actual imported source drift: " + role)
    root = Path(args.root).resolve(strict=True)
    if root != Path(manifest["approved_root"]).resolve(strict=True):
        raise ContractError("Output root differs from lead's approved private root")
    destination = root / args.label
    if destination.parent != root or destination.is_symlink():
        raise ContractError("Unsafe output label path")
    receipt = {"schema_version": 1, "rank": rank, "world_size": world, "local_rank": local,
               "driver_sha256": sha_file(__file__), "contracts_sha256": sha_file(Path(__file__).with_name("contracts126.py")),
               "manifest_sha256": sha_file(args.runtime_manifest), "latents_sha256": LATENTS_SHA256,
               "installer_mode": args.installer_mode, "cases": {}, "residency": {}, "installer": {"calls": 0, "native_return": None},
               "parameter_contract": {"all_float_parameters_fp32": False}, "completed": False,
               "scope": "Same-final-latent video/audio VAE only. No DiT/encoder inference, no E2E claim", "accepted_lossless": False}
    output_path = None
    video = audio = None
    try:
        torch.distributed.init_process_group("hccl", timeout=datetime.timedelta(seconds=args.timeout_seconds))
        creation_error = None
        if rank == 0:
            try:
                destination.mkdir(mode=0o700, exist_ok=False)
            except OSError as exc:
                creation_error = str(exc)
        status = torch.tensor([int(creation_error is not None)], dtype=torch.int32, device="npu")
        torch.distributed.broadcast(status, src=0)
        if int(status.cpu()[0]):
            raise ContractError("Refuse reused/uncreatable run directory: " + (creation_error or "rank0 failed"))
        torch.distributed.barrier()
        output_path = destination / ("rank%d.json" % rank)
        if rank == 0:
            atomic_json(destination / "runtime_manifest.json", manifest)
        def publish():
            atomic_json(output_path, receipt)
        publish()
        dispatch_function = ops.resolve_h3_vae_operators
        operator_set = dispatch_function(torch.device("npu", local))
        receipt["dispatch_probe"] = {"function": source_identity(dispatch_function),
            "device": "npu:%d" % local, "returns_operator_set": operator_set is not None,
            "operator_identities": {name: source_identity(getattr(operator_set, name)) for name in ("qk_norm_rope", "scaled_residual")} if operator_set is not None else None,
            "meaning": "Observed actual ops.vae.dispatch capability lookup, not a wrongly named module import"}
        receipt["activation_contract"] = {"source_sha256": manifest["modules"]["activation"]["sha256"],
            "VLLM_OMNI_910A_NO_FUSED_ACT": os.environ["VLLM_OMNI_910A_NO_FUSED_ACT"],
            "forward_npu_default_is_eager": True, "native_fast_FF_silu_guard_is_cuda_only": True}
        previous_dtype = torch.get_default_dtype()
        begun = time.perf_counter()
        torch.set_default_dtype(torch.float32)
        try:
            with observed_installer(torch, adapter, args.installer_mode, receipt["installer"]):
                video = adapter.MiniMaxH3VideoVAE(str(Path(args.checkpoint) / "video_vae"), device=torch.device("npu", local), load_device=torch.device("cpu"), decode_only=True, trust_remote_code=True)
            audio = adapter.MiniMaxH3AudioVAE(str(Path(args.checkpoint) / "audio_vae"), device=torch.device("npu", local), load_device=torch.device("cpu"), decode_only=True, trust_remote_code=True)
        finally:
            torch.set_default_dtype(previous_dtype)
        receipt["cpu_construct_video_audio_s"] = time.perf_counter() - begun
        receipt["native_sources"] = verify_native_sources(manifest, video, audio)
        receipt["parameter_contract"] = {"all_float_parameters_fp32": all_float_fp32(torch, video) and all_float_fp32(torch, audio),
                                          "video": parameters(video), "audio": parameters(audio),
                                          "weights_saved_or_transferred_as_artifacts": False}
        if not receipt["parameter_contract"]["all_float_parameters_fp32"]:
            raise ParameterRewriteRejected("Original FP32 video/audio parameters required")
        video.set_parallel_size(world, mode="tile", process_group=torch.distributed.group.WORLD)
        if not video.model.parallel_tiling or video.model.decoder.spatial_parallel:
            raise ContractError("Expected native eight-rank tile parallel, not decoder spatial parallel")
        receipt["native_installer_false_fused_gain_not_measured"] = receipt["installer"]["native_return"] is False
        compat.install_half_nan_compat()
        if len(video.model.decoder.transformer_blocks) != 36:
            raise ContractError("Expected complete 36-layer native decoder")
        tracker = Engagement(video.model.decoder, ops)
        receipt["binding_snapshot"] = tracker.snapshot()
        fast_active = any(v["matches_native_fast_function"] for row in receipt["binding_snapshot"]["blocks"] for v in row["methods"].values())
        if args.installer_mode == "preserve-fp32" and fast_active:
            raise ContractError("Unexpected pre-installed fast forward in the R124 preserve-fp32 baseline")
        if args.installer_mode == "observe-native" and (fast_active or receipt["installer"]["native_return"] is True) and args.reference_dir is None:
            raise ContractError("An installed native candidate requires a completed external preserve-fp32 reference before forward")
        bundle = torch.load(args.latents, map_location="cpu", weights_only=True)
        if not isinstance(bundle, dict) or not {"video_latent", "audio_latent"}.issubset(bundle):
            raise ContractError("Unexpected final-latent bundle keys")
        if bundle.get("height") != 768 or bundle.get("width") != 1344 or bundle.get("actual_DiT_evaluations") != 49 or bundle.get("contains_model_weights") is not False:
            raise ContractError("Final-latent provenance/geometry metadata changed")
        vc, ac = bundle["video_latent"], bundle["audio_latent"]
        if list(vc.shape) != VIDEO_LATENT_SHAPE or list(ac.shape) != AUDIO_LATENT_SHAPE or vc.dtype != torch.float32 or ac.dtype != torch.float32:
            raise ContractError("Final latent shape/dtype drift")
        if not bool(torch.isfinite(vc).all() and torch.isfinite(ac).all()):
            raise ContractError("Nonfinite source latent")
        receipt["inputs"] = {"video": tensor_receipt(torch, vc), "audio": tensor_receipt(torch, ac)}
        video_latent, audio_latent = vc.to(torch.device("npu", local)).contiguous(), ac.to(torch.device("npu", local)).contiguous()
        torch.npu.synchronize()
        references = {}
        external_tiles = None
        if args.reference_dir is not None:
            reference_dir = Path(args.reference_dir).resolve(strict=True)
            rank_receipt = json.loads((reference_dir / ("rank%d.json" % rank)).read_text())
            if not rank_receipt.get("completed") or rank_receipt.get("failure") or not is_baseline_receipt(rank_receipt) or rank_receipt["latents_sha256"] != LATENTS_SHA256:
                raise ContractError("External baseline is incomplete, a candidate, or a different input")
            if rank_receipt.get("driver_sha256") != receipt["driver_sha256"] or rank_receipt.get("contracts_sha256") != receipt["contracts_sha256"] or rank_receipt.get("manifest_sha256") != receipt["manifest_sha256"]:
                raise ContractError("External baseline sources/config differ from this pinned comparison")
            if not rank_receipt.get("full_reference_outputs_saved"):
                raise ContractError("External baseline did not persist the full raw outputs")
            references["video"] = torch.load(reference_dir / "reference_video_rank0.pt", map_location="cpu", weights_only=True)
            references["audio"] = torch.load(reference_dir / ("reference_audio_rank%d.pt" % rank), map_location="cpu", weights_only=True)
            for name, value in references.items():
                expected = rank_receipt["reference_outputs"][name]
                if tensor_receipt(torch, value)["sha256"] != expected["sha256"]:
                    raise ContractError("External full output bytes differ from the baseline receipt: " + name)
            if not args.skip_tile_capture:
                capture = rank_receipt.get("tile_capture", {})
                if not capture.get("complete_capture"):
                    raise ContractError("External baseline needs complete prequantization tile fingerprints")
                external_tiles = capture["tiles"]
        # One reused root pair, allocated and warmed before any timed decode.
        start_event, end_event = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start_event.record(); end_event.record(); torch.npu.synchronize()
        def candidate_gate(equal, reason):
            if args.reference_dir is None:
                return
            flag = torch.tensor([int(not equal)], dtype=torch.int32, device=torch.device("npu", local))
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            if int(flag.cpu()[0]):
                receipt["requires_user_numerical_decision"] = True
                publish()
                raise CandidateDelta(reason + "; stop on all ranks and ask user")
        def residency(name, function):
            torch.distributed.barrier(); torch.npu.synchronize()
            begin = time.perf_counter(); start_event.record()
            function(); end_event.record(); torch.npu.synchronize()
            result = {"wall_s": time.perf_counter() - begin, "current_stream_s": start_event.elapsed_time(end_event) / 1000,
                      "scope": "staging/offload only, outside resident decode"}
            receipt["residency"][name] = result
            publish()
        def decode(name, index, instrumented=False):
            torch.distributed.barrier(); torch.npu.synchronize()
            latent = video_latent if name == "video" else audio_latent
            model = video if name == "video" else audio
            begin = time.perf_counter(); start_event.record()
            with torch.inference_mode(), compat.half_nan_scope(enabled=True) as nan_counter:
                with current_omni_platform.create_autocast_context(device_type="npu", dtype=torch.float16, enabled=name == "video"):
                    result = model.decode_latent(latent)
            end_event.record(); torch.npu.synchronize()
            elapsed = time.perf_counter() - begin
            stream_s = start_event.elapsed_time(end_event) / 1000
            # Copies, hashing, metrics and serialization are always after timer end.
            copy_begin = time.perf_counter(); cpu = result.detach().cpu().contiguous()
            torch.npu.synchronize()
            d2h = time.perf_counter() - copy_begin
            del result
            proof = tensor_receipt(torch, cpu)
            if proof["shape"] != manifest[name + "_output_shape"] or proof["dtype"] != manifest[name + "_output_dtype"]:
                raise ContractError("Actual complete output contract changed: " + name)
            if not proof["finite_CPU"]:
                raise ContractError("Nonfinite complete output observed on CPU: " + name)
            if name not in references:
                references[name] = cpu
                comparison, role = None, "first_local_self_reference"
            else:
                comparison = compare_cpu(torch, cpu, references[name])
                role = "against_external_baseline" if args.reference_dir else "same_graph_self_repeat"
            row = {"call_index": index, "wall_s": elapsed, "stream_s": stream_s, "D2H_and_CPU_contiguous_s": d2h,
                   "output": proof, "comparison": comparison, "comparison_role": role,
                   "instrumented_not_clean": instrumented, "NanToNum_compatibility_calls": nan_counter["calls"],
                   "root_events_current_stream_only_wall_drains_all_NPU_streams": True}
            return cpu, row
        receipt["reference_outputs"] = {}
        residency("video_load", video.decoder_component.load_to_device)
        if not all_float_fp32(torch, video):
            raise ParameterRewriteRejected("Video staging changed FP32 parameter storage")
        receipt["parameter_contract"]["video_resident"] = parameters(video)
        for name in ("video", "audio"):
            if name == "audio":
                residency("video_offload", video.decoder_component.offload_to_cpu)
                residency("audio_load", audio.load_to_device)
                if not all_float_fp32(torch, audio):
                    raise ParameterRewriteRejected("Audio staging changed FP32 parameter storage")
                receipt["parameter_contract"]["audio_resident"] = parameters(audio)
            receipt["cases"][name + "_clean"] = []
            for index in range(args.warm_repeats + 1):
                cpu, row = decode(name, index)
                receipt["cases"][name + "_clean"].append(row)
                receipt["reference_outputs"][name] = tensor_receipt(torch, references[name])
                publish()
                candidate_gate(row["comparison"] is None or row["comparison"]["raw_bits_equal"], "Complete " + name + " output differs from the external baseline")
                del cpu
            if name == "video":
                # Actual hit counts and event attribution are in a separate call.
                regions = None
                try:
                    tracker.install()
                    if not args.skip_region_profile:
                        regions = EventRegions(torch, manifest.get("profile_slots_per_region", 512))
                        regions.install_video(video)
                    cpu, row = decode("video", 0, instrumented=True)
                    row["engagement"] = tracker.receipt()
                    if regions is not None:
                        row["region_profile"] = regions.receipt()
                    receipt["cases"]["video_engagement_profile"] = [row]
                    del cpu
                    publish()
                finally:
                    try:
                        if regions is not None:
                            regions.close()
                    finally:
                        tracker.close()
                candidate_gate(row["comparison"] is None or row["comparison"]["raw_bits_equal"], "Instrumented complete video differs from external baseline")
                if regions is not None and row["region_profile"].get("incomplete_profile"):
                    raise ContractError("Region event pool overflow; attribution is incomplete")
                if not args.skip_tile_capture:
                    capture = TileCapture(video.model.decoder, manifest.get("max_decoder_tiles_per_rank", 128), manifest.get("tile_capture_device_limit_bytes", 1024 ** 3))
                    try:
                        capture.install()
                        cpu, row = decode("video", 0, instrumented=True)
                        receipt["cases"]["video_tile_capture"] = [row]
                        receipt["tile_capture"] = capture.receipt(torch, external_tiles)
                        del cpu
                        publish()
                    finally:
                        capture.close()
                        capture.values.clear()
                    if not receipt["tile_capture"]["complete_capture"]:
                        raise ContractError("Prequantization tile capture exceeded its explicit bounds")
                    candidate_gate(external_tiles is None or bool(receipt["tile_capture"]["all_reference_tiles_raw_equal"]), "Prequantization tile words differ even if uint8 output hides it")
            elif not args.skip_region_profile:
                audio_regions = EventRegions(torch, manifest.get("profile_slots_per_region", 512), names=("audio_remote_decode",))
                try:
                    audio_regions.bind(audio.remote, "decode", "audio_remote_decode")
                    cpu, row = decode("audio", 0, instrumented=True)
                    row["region_profile"] = audio_regions.receipt()
                    receipt["cases"]["audio_engagement_profile"] = [row]
                    del cpu
                    publish()
                finally:
                    audio_regions.close()
                candidate_gate(row["comparison"] is None or row["comparison"]["raw_bits_equal"], "Instrumented complete audio differs from external baseline")
                if row["region_profile"].get("incomplete_profile"):
                    raise ContractError("Audio region profile exceeded its event pool")
        residency("audio_offload", audio.offload_to_cpu)
        if not args.no_save_reference:
            total_output = references["video"].numel() * references["video"].element_size() + WORLD_SIZE * references["audio"].numel() * references["audio"].element_size()
            # Include bounded serializer overhead. This is a run-wide cap, stricter than per-case.
            if total_output + 1024 * 1024 > min(args.output_limit_bytes, manifest.get("output_limit_bytes", MAX_OUTPUT_BYTES)):
                raise ContractError("Full raw reference output exceeds disk cap; use explicit 2GiB approved manifest or --no-save-reference")
            if rank == 0:
                with (destination / "reference_video_rank0.pt").open("xb") as target:
                    torch.save(references["video"], target)
            with (destination / ("reference_audio_rank%d.pt" % rank)).open("xb") as target:
                torch.save(references["audio"], target)
            receipt["full_reference_outputs_saved"] = True
            receipt["saved_reference_raw_bytes_run_wide"] = total_output
        receipt["inputs_after"] = {"video": tensor_receipt(torch, video_latent), "audio": tensor_receipt(torch, audio_latent)}
        if receipt["inputs_after"] != receipt["inputs"]:
            raise ContractError("Decode modified the source latent")
        if not all_float_fp32(torch, video) or not all_float_fp32(torch, audio):
            raise ParameterRewriteRejected("Offload cycle failed to preserve original FP32 parameter dtypes")
        receipt["parameter_contract"]["after_offload"] = {"video": parameters(video), "audio": parameters(audio)}
        receipt["completed"] = True
        publish()
        print(json.dumps({"rank": rank, "completed": True, "native_installer_return": receipt["installer"]["native_return"],
                          "native_installer_called": receipt["installer"].get("native_called"), "receipt": str(output_path), "accepted_lossless": False}), flush=True)
    except BaseException as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Failures must still attempt instance offload and group cleanup; preserve
        # the original error if cleanup or atomic receipt publication also fails.
        cleanup_errors = []
        for name, function in (("video_offload", video.decoder_component.offload_to_cpu if video is not None else None),
                               ("audio_offload", audio.offload_to_cpu if audio is not None else None)):
            if function is not None:
                try:
                    function()
                except BaseException as exc:
                    cleanup_errors.append({"operation": name, "type": type(exc).__name__, "message": str(exc)})
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except BaseException as exc:
                cleanup_errors.append({"operation": "destroy_process_group", "type": type(exc).__name__, "message": str(exc)})
        receipt["cleanup"] = {"completed": not cleanup_errors, "errors": cleanup_errors}
        if cleanup_errors:
            receipt["cleanup_failure"] = cleanup_errors
        if output_path is not None:
            try:
                atomic_json(output_path, receipt)
            except BaseException as exc:
                print(json.dumps({"rank": rank, "receipt_save_failure": str(exc)}), file=sys.stderr, flush=True)
        if cleanup_errors and "failure" not in receipt:
            raise RuntimeError("Decode completed but resource cleanup failed: " + str(cleanup_errors))


def main(argv=None):
    args = validate_cli(parser().parse_args(argv))
    if args.output_limit_bytes < 1 or args.output_limit_bytes > 2 * MAX_OUTPUT_BYTES:
        raise ContractError("Output cap must be explicitly between 1 byte and 2GiB")
    if not args.execute and not args.validate_only:
        print(json.dumps(planned(args), indent=2)); return 0
    if not args.runtime_manifest:
        raise ContractError("--runtime-manifest is required before execution or validation")
    manifest = validate_manifest(json.loads(Path(args.runtime_manifest).read_text()))
    if args.validate_only:
        check_local_files(manifest, args.checkpoint, args.latents)
        print(json.dumps({"validated": True, "Torch_imported": False, "model_run": False}, indent=2)); return 0
    run(args, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
