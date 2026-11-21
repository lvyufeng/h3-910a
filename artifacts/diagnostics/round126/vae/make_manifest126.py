#!/usr/bin/env python3
"""Build the VAE runtime manifest from lead's source-only JSON preflights.

No Torch imports, remote calls or weight reads. Missing pins fail closed. Output
is printed unless --output explicitly names a new file; no old source is edited.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from contracts126 import (AUDIO_LATENT_SHAPE, LATENTS_BYTES, LATENTS_SHA256,
                          MAX_OUTPUT_BYTES, VIDEO_LATENT_SHAPE, VIDEO_OUTPUT_SHAPE,
                          ContractError, validate_manifest)

IMPORTS = {
    "adapter": "vllm_omni.diffusion.models.minimax_h3.vae",
    "optimizations": "vllm_omni.diffusion.models.minimax_h3.ops.vae",
    "dispatch": "vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch",
    "sdpa_shim": "vllm_omni.h3_910a_sdpa_shim",
    "nan_compat": "npu_half_nan_compat124",
    "activation": "vllm_omni.diffusion.layers.activation",
    "module_residency": "vllm_omni.diffusion.offloader.module_residency",
    "native_temporal": "vllm_omni.diffusion.models.minimax_h3.vae_temporal",
    "fa_backend": "vllm_omni.diffusion.attention.backends.fa910a",
}
NATIVE_FILES = {"native_video": "klvae.py", "native_decoder": "vae_vit.py", "native_attention": "attention.py",
                "native_blocks": "base_module.py", "native_func": "func.py", "native_flash": "flash.py", "native_parallel": "parallel.py"}


def build(facts):
    modules = {}
    components = {}
    versions = {}
    config_files = {}
    for fact in facts:
        if not fact.get("completed") or fact.get("weights_loaded", False):
            raise ContractError("Need completed source-only preflight facts")
        versions.update({key: fact[key] for key in ("torch_version", "torch_npu_version") if key in fact})
        for record in list(fact.get("modules", {}).values()) + list(fact.get("source_files", {}).values()):
            name = record.get("import_name", record.get("import"))
            if name and "sha256" in record and "path" in record:
                modules[name] = {"import": name, "path": record["path"], "sha256": record["sha256"]}
        candidates = fact.get("sdpa_shim_candidates", [])
        if candidates:
            if len(candidates) != 1:
                raise ContractError("Ambiguous SDPA shim source candidates")
            record = candidates[0]
            modules[IMPORTS["sdpa_shim"]] = {"import": IMPORTS["sdpa_shim"], "path": record["path"], "sha256": record["sha256"]}
        if "nan_compat_path" in fact:
            modules[IMPORTS["nan_compat"]] = {"import": IMPORTS["nan_compat"], "path": fact["nan_compat_path"], "sha256": fact["nan_compat_sha256"]}
        components.update(fact.get("component_config", {}))
        config_files.update(fact.get("config_files", {}))
    missing = set(IMPORTS.values()) - set(modules)
    if missing:
        raise ContractError("Missing live module pins: " + ", ".join(sorted(missing)))
    if not {"video_vae", "audio_vae"}.issubset(components):
        raise ContractError("Missing native component source/config facts")
    source_config = "/home/mseco/models/MiniMax-H3/FL2VA/video_vae/source/config.json"
    if source_config not in config_files:
        raise ContractError("Missing native video source/config.json pin")
    native = {role: components["video_vae"]["source_files"][filename] for role, filename in NATIVE_FILES.items()}
    native["native_audio"] = components["audio_vae"]["source_files"]["minimax_h3_audio_vae.py"]
    data = {"schema_version": 1, "approved_executor": "lead", "approved_root": "/tmp/h3_perf12601", "world_size": 8,
            "checkpoint": "/home/mseco/models/MiniMax-H3/FL2VA",
            "latents": "/tmp/h3_align12401/final_NPU_latents_pid2331981124.pt",
            "latents_sha256": LATENTS_SHA256, "latents_bytes": LATENTS_BYTES,
            "video_latent_shape": VIDEO_LATENT_SHAPE, "audio_latent_shape": AUDIO_LATENT_SHAPE,
            "latent_dtype": "torch.float32", "parameter_dtype": "torch.float32", "video_autocast_dtype": "torch.float16", "hf32": False,
            "nan_compat_dir": "/tmp/h3_align12401", **versions,
            "video_config_sha256": components["video_vae"]["sha256"], "audio_config_sha256": components["audio_vae"]["sha256"],
            "video_source_config_sha256": config_files[source_config]["sha256"],
            "modules": {role: modules[name] for role, name in IMPORTS.items()}, "native_source_sha256": native,
            "component_source_sha256": {name: components[name]["source_files"] for name in ("video_vae", "audio_vae")},
            "environment": {"VLLM_OMNI_910A_FLASH_ATTN": "1", "VLLM_OMNI_910A_NO_FUSED_ACT": "1",
                            "VLLM_OMNI_910A_NO_FUSED_NORM": "1", "H3_NPU_ALIGNMENT124_BOOTSTRAP": "0",
                            "H3_NPU_SAVE_FINAL_LATENTS124": "0", "H3_NPU_CAPTURE_NATIVE_MEDIA124": "0",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            "preserve_inherited_CANN_PYTHONPATH": True, "IPC_key_environment_inherited_not_read_or_printed": "H3_R126_IPC_KEY_FILE",
            "video_output_shape": VIDEO_OUTPUT_SHAPE, "video_output_dtype": "torch.uint8",
            "video_output_schema_is_expected_fail_closed_not_previously_executed126": True,
            "audio_output_shape": [1, 2, 165600], "audio_output_dtype": "torch.float32", "output_limit_bytes": MAX_OUTPUT_BYTES,
            "profile_slots_per_region": 512, "max_decoder_tiles_per_rank": 128, "tile_capture_device_limit_bytes": 1024 ** 3,
            "initialization_model_scope": "video_audio_vae_only_no_DiT_no_encoder_forward"}
    return validate_manifest(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", action="append", required=True, help="Repeat for runtime/extensions/compat preflight JSON")
    parser.add_argument("--output", help="New JSON path; omitted means stdout only")
    args = parser.parse_args(argv)
    data = build([json.loads(Path(path).read_text()) for path in args.facts])
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as output:
            output.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
