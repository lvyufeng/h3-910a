# ts-217 H3 model manifest

This manifest records every H3 model file currently present under
`/home/mseco/h3-910a/models`. The production path is the five-file set: DiT,
text encoder, video VAE, audio VAE and
`minimax_h3_turbo_v4_step600_ema.safetensors`. The JSON file is consumed by
`scripts/check_ts217_env.sh` through `H3_MODEL_MANIFEST`.

The sixth file is the older
`minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` LoRA. Keep it
for rollback or comparison, but do not use it in the production workflow.

| Destination | Bytes | SHA256 |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 | `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| `loras/minimax_h3_turbo_v4_step600_ema.safetensors` | 779,849,816 | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 1,956,192,992 | `c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc` |

The last row is the legacy, non-production LoRA.

The checked-in runtime template is `config/extra_model_paths.yaml`, but that
file does not yet exist on `ts-217`. The current processes explicitly load
`run/h3-a/extra_model_paths.yaml` and `run/h3-b/extra_model_paths.yaml`; both
point at the shared model tree and match the root
`/home/mseco/h3-910a/extra_model_paths.yaml`. Writable state remains under
separate `run/h3-a` and `run/h3-b` directories.
