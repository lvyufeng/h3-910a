"""Stdlib-only contracts for R126 same-final-latent VAE replay.

Importing this module never imports Torch, remote code or a model. Execution
requires an explicit flag plus a SHA-pinned runtime manifest supplied by lead.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import uuid

SCHEMA_VERSION = 1
VIDEO_LATENT_SHAPE = [1, 24, 37, 48, 84]
AUDIO_LATENT_SHAPE = [2, 32, 207]
VIDEO_OUTPUT_SHAPE = [1, 3, 124, 768, 1344]
LATENTS_SHA256 = "5ccba4c9293651bcc57756a0fb2acb39f4e5bd04439c5df6470901bc4c6759b3"
LATENTS_BYTES = 14376549
WORLD_SIZE = 8
MAX_OUTPUT_BYTES = 1024 ** 3
ALLOWED_MODULE_ROLES = {
    "adapter", "optimizations", "dispatch", "sdpa_shim", "nan_compat",
    "fa_backend", "native_video", "native_decoder", "native_attention",
    "native_blocks", "native_func", "native_flash", "native_parallel",
    "native_audio", "native_temporal", "module_residency", "activation",
}
REQUIRED_MODULE_ROLES = {"adapter", "optimizations", "dispatch", "sdpa_shim", "nan_compat", "activation"}
REQUIRED_NATIVE_ROLES = {
    "native_video", "native_decoder", "native_attention", "native_blocks",
    "native_func", "native_flash", "native_parallel", "native_audio",
}
FORBIDDEN_ENV = {
    "VLLM_OMNI_910A_VAE_AUDIT_DIR", "VLLM_OMNI_910A_TEXT_AUDIT_DIR",
    "VLLM_OMNI_910A_AUDIT_DIR", "VLLM_OMNI_910A_ADALN_AUDIT_DIR",
    "VLLM_OMNI_910A_BLOCK_AUDIT_DIR",
}


class ContractError(ValueError):
    pass


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(label + " must be a lowercase SHA-256")
    return value


def absolute_path(value, label):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ContractError(label + " must be an absolute path")
    return Path(value)


def positive_int(value, label, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ContractError(label + " has an invalid integer value")
    return value


def validate_manifest(data):
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("Unknown runtime manifest schema")
    if data.get("approved_executor") != "lead" or data.get("world_size") != WORLD_SIZE:
        raise ContractError("Execution is owned by lead on eight ranks")
    absolute_path(data.get("checkpoint"), "checkpoint")
    absolute_path(data.get("latents"), "latents")
    absolute_path(data.get("nan_compat_dir"), "nan_compat_dir")
    absolute_path(data.get("approved_root"), "approved_root")
    if data.get("latents_sha256") != LATENTS_SHA256 or data.get("latents_bytes") != LATENTS_BYTES:
        raise ContractError("Only the exact accepted-FA final-latent bundle is supported")
    if data.get("video_latent_shape") != VIDEO_LATENT_SHAPE or data.get("audio_latent_shape") != AUDIO_LATENT_SHAPE:
        raise ContractError("This replay is scoped to the actual R124+FA latent geometry")
    if data.get("latent_dtype") != "torch.float32" or data.get("parameter_dtype") != "torch.float32":
        raise ContractError("Original FP32 latent and parameter storage is required")
    if data.get("video_autocast_dtype") != "torch.float16" or data.get("hf32") is not False:
        raise ContractError("Keep the standard Half video autocast and disabled HF32")
    for key in ("torch_version", "torch_npu_version"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ContractError("Missing " + key)
    for key in ("video_config_sha256", "audio_config_sha256", "video_source_config_sha256"):
        validate_sha(data.get(key), key)
    modules = data.get("modules")
    if not isinstance(modules, dict) or not REQUIRED_MODULE_ROLES.issubset(modules):
        raise ContractError("Missing SHA-pinned imported runtime module roles")
    if not set(modules).issubset(ALLOWED_MODULE_ROLES):
        raise ContractError("Unknown runtime source role")
    for role, record in modules.items():
        if not isinstance(record, dict):
            raise ContractError("Bad source record: " + role)
        absolute_path(record.get("path"), role + ".path")
        validate_sha(record.get("sha256"), role + ".sha256")
        module = record.get("import")
        if role in REQUIRED_MODULE_ROLES and (not isinstance(module, str) or not module):
            raise ContractError("Missing import name: " + role)
    sources = data.get("component_source_sha256")
    if not isinstance(sources, dict) or set(sources) != {"video_vae", "audio_vae"}:
        raise ContractError("Missing complete native component .py source pins")
    for component, files in sources.items():
        if not isinstance(files, dict) or not files:
            raise ContractError("Empty component source map")
        for filename, digest in files.items():
            if Path(filename).name != filename or not filename.endswith(".py"):
                raise ContractError("Unsafe native source filename")
            validate_sha(digest, component + "/" + filename)
    native = data.get("native_source_sha256")
    if not isinstance(native, dict) or not REQUIRED_NATIVE_ROLES.issubset(native):
        raise ContractError("Missing dynamic/native source fingerprints")
    for role, digest in native.items():
        if role not in ALLOWED_MODULE_ROLES:
            raise ContractError("Unknown native source role")
        validate_sha(digest, role)
    env = data.get("environment")
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ContractError("environment must map strings to strings")
    if env.get("VLLM_OMNI_910A_FLASH_ATTN") != "1":
        raise ContractError("The user-approved FA baseline environment must be retained")
    if any(env.get(key) for key in FORBIDDEN_ENV):
        raise ContractError("Legacy capture hooks would contaminate clean timing")
    if env.get("H3_NPU_ALIGNMENT124_BOOTSTRAP", "0") != "0":
        raise ContractError("Do not install DiT-dependent R124 alignment hooks")
    if env.get("H3_NPU_SAVE_FINAL_LATENTS124", "0") != "0" or env.get("H3_NPU_CAPTURE_NATIVE_MEDIA124", "0") != "0":
        raise ContractError("Use this decoder's own guards/output path, not DiT-dependent R124 capture")
    if env.get("VLLM_OMNI_910A_NO_FUSED_ACT", "1") != "1":
        raise ContractError("Do not default to unsupported aclnn SwiGLU; preserve eager activation")
    if modules["dispatch"]["import"] != "vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch":
        raise ContractError("The actual VAE dispatch is inside the ops.vae package")
    if env.get("VLLM_OMNI_910A_VAE_FINITE_SELECT", "0") != "0":
        raise ContractError("Do not silently introduce the old finite-select candidate")
    if data.get("video_output_shape") != VIDEO_OUTPUT_SHAPE:
        raise ContractError("Unexpected full video contract")
    if data.get("video_output_dtype") not in ("torch.uint8", "torch.float32"):
        raise ContractError("Unsupported video output dtype")
    audio_shape = data.get("audio_output_shape")
    if not isinstance(audio_shape, list) or math.prod(audio_shape) != 2 * 165600:
        raise ContractError("Expected the entire native stereo waveform")
    if data.get("audio_output_dtype") != "torch.float32":
        raise ContractError("Audio must remain FP32")
    positive_int(data.get("output_limit_bytes", MAX_OUTPUT_BYTES), "output_limit_bytes", maximum=2 * MAX_OUTPUT_BYTES)
    positive_int(data.get("profile_slots_per_region", 512), "profile_slots_per_region", maximum=4096)
    if data.get("initialization_model_scope") != "video_audio_vae_only_no_DiT_no_encoder_forward":
        raise ContractError("This is a decode-only harness, not a pipeline generation")
    return data


def check_local_files(manifest, checkpoint, latents):
    """Source/config/bundle checks only; never inspect or copy weight files."""
    if Path(checkpoint).resolve(strict=True) != Path(manifest["checkpoint"]).resolve(strict=True):
        raise ContractError("Checkpoint argument differs from lead's pinned root")
    if Path(latents).resolve(strict=True) != Path(manifest["latents"]).resolve(strict=True):
        raise ContractError("Latent argument differs from the pinned source")
    if Path(latents).stat().st_size != LATENTS_BYTES or sha_file(latents) != LATENTS_SHA256:
        raise ContractError("Final-latent bytes do not match the accepted-FA input")
    for name in ("video", "audio"):
        config = Path(checkpoint) / (name + "_vae") / "config.json"
        if sha_file(config) != manifest[name + "_config_sha256"]:
            raise ContractError("Wrong native component config: " + name)
    if sha_file(Path(checkpoint) / "video_vae" / "source" / "config.json") != manifest["video_source_config_sha256"]:
        raise ContractError("Native video architecture source/config.json drift")
    for component, sources in manifest["component_source_sha256"].items():
        for filename, digest in sources.items():
            if sha_file(Path(checkpoint) / component / filename) != digest:
                raise ContractError("Native component source drift: " + component + "/" + filename)
    for role, record in manifest["modules"].items():
        if sha_file(record["path"]) != record["sha256"]:
            raise ContractError("Source drift before execution: " + role)


def validate_cli(args):
    for name in ("checkpoint", "latents", "root"):
        absolute_path(getattr(args, name), name)
    if not isinstance(args.label, str) or re.fullmatch(r"[a-z][a-z0-9_-]{2,79}", args.label) is None:
        raise ContractError("label must be a new lowercase run name, 3-80 characters")
    if args.installer_mode not in ("preserve-fp32", "observe-native"):
        raise ContractError("Unknown installer mode")
    positive_int(args.warm_repeats, "warm_repeats", minimum=2, maximum=10)
    positive_int(args.cpu_threads, "cpu_threads", maximum=16)
    positive_int(args.timeout_seconds, "timeout_seconds", minimum=60, maximum=7200)
    if args.reference_dir is not None:
        absolute_path(args.reference_dir, "reference_dir")
    return args


def atomic_json(path, data):
    """Replace only our own receipt inside an already exclusively-created run."""
    path = Path(path)
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    if path.is_symlink() or target != path.absolute():
        raise ContractError("Unexpected receipt publication target")
    temporary = parent / (path.name + ".part-" + uuid.uuid4().hex)
    if temporary.is_symlink() or temporary.parent != parent:
        raise ContractError("Unexpected temporary publication path")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    # Verify both resolved absolute targets before this move/publication.
    if temporary.resolve(strict=True).parent != parent or target.parent != parent:
        raise ContractError("Receipt paths changed during publication")
    os.replace(str(temporary), str(target))


def function_identity(function):
    """Names only, usable without importing Torch or calling a function."""
    import inspect
    function = getattr(function, "__func__", function)
    function = inspect.unwrap(function)
    return {"module": getattr(function, "__module__", None),
            "qualname": getattr(function, "__qualname__", type(function).__name__)}


def is_baseline_receipt(row):
    installer = row.get("installer", {})
    bypass = row.get("installer_mode") == "preserve-fp32" and installer.get("native_called") is False
    native_noop = row.get("installer_mode") == "observe-native" and installer.get("native_called") is True and installer.get("native_return") is False
    binding = row.get("binding_snapshot", {})
    fast = any(value.get("matches_native_fast_function") for block in binding.get("blocks", []) for value in block.get("methods", {}).values())
    return (bypass or native_noop) and not fast and not binding.get("installed_marker", False) and row.get("parameter_contract", {}).get("all_float_parameters_fp32") is True


def aggregate_rank_receipts(rows):
    """Timing only: median of per-call rank maxima, not max of local medians."""
    if len(rows) != WORLD_SIZE or {row.get("rank") for row in rows} != set(range(WORLD_SIZE)):
        raise ContractError("Need eight unique rank receipts")
    if any(not row.get("completed") or row.get("failure") or row.get("cleanup_failure") for row in rows):
        raise ContractError("Cannot aggregate an incomplete/failed execution")
    identities = {(row["latents_sha256"], row["driver_sha256"], row["manifest_sha256"], row["installer_mode"]) for row in rows}
    if len(identities) != 1:
        raise ContractError("Mixed source/latent/installer modes")
    result = {"scope": "Same-final-latent decode only; timing does not imply numerical acceptance", "cases": {}}
    keys = set(rows[0]["cases"])
    if any(set(row["cases"]) != keys for row in rows):
        raise ContractError("Ranks executed different cases")
    for key in sorted(keys):
        cases = [row["cases"][key] for row in rows]
        sizes = {len(case) for case in cases}
        if len(sizes) != 1 or not sizes or sizes == {0}:
            raise ContractError("Unaligned case indices")
        count = sizes.pop()
        if any(case[i]["call_index"] != i for case in cases for i in range(count)):
            raise ContractError("Missing or reordered call indices")
        wall = [max(case[i]["wall_s"] for case in cases) for i in range(count)]
        events = [max(case[i]["stream_s"] for case in cases) for i in range(count)]
        warm = wall[1:] if key.endswith("_clean") else []
        d2h = [max(case[i]["D2H_and_CPU_contiguous_s"] for case in cases) for i in range(count)] if all("D2H_and_CPU_contiguous_s" in call for case in cases for call in case) else None
        result["cases"][key] = {"same_call_rank_max_wall_s": wall, "same_call_rank_max_stream_s": events,
                                 "same_call_rank_max_D2H_and_CPU_contiguous_s": d2h,
                                 "cold_rank_max_wall_s": wall[0], "median_warm_rank_max_wall_s": statistics.median(warm) if warm else None,
                                 "profile_instrumented_not_clean": not key.endswith("_clean"),
                                 "raw_output_shas": [[case[i]["output"]["sha256"] for case in cases] for i in range(count)],
                                 "any_comparison_delta": any(case[i]["comparison"] and not case[i]["comparison"]["raw_bits_equal"] for case in cases for i in range(count))}
    residency_keys = set(rows[0].get("residency", {}))
    if any(set(row.get("residency", {})) != residency_keys for row in rows):
        raise ContractError("Ranks have incomplete staging/offload boundaries")
    result["residency"] = {key: {"rank_max_wall_s": max(row["residency"][key]["wall_s"] for row in rows),
                                  "rank_max_current_stream_s": max(row["residency"][key]["current_stream_s"] for row in rows)} for key in sorted(residency_keys)}
    result["accepted_lossless"] = False
    result["native_installer_returns"] = [row["installer"]["native_return"] for row in rows]
    result["original_FP32_parameters_preserved"] = all(row["parameter_contract"]["all_float_parameters_fp32"] for row in rows)
    return result
