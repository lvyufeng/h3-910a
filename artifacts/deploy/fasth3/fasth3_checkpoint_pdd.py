# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampling contract for FastVideo FastH3 full checkpoints.

Two releases share this module:

* the V2 8-step DMD ladder (``dmd_denoising_steps``, video shift 10, T2AV);
* the PDD-8 dense export (``fasth3-inference-contract-v1`` with
  ``pdd_steps``/``pdd_step_indices``, ref2va conditioning, video/audio shifts
  12/3), whose widened final projection is sampled by the PDD runtime in
  :mod:`.pdd`.

Both are full checkpoints: the released weights already contain the whole
student, including its learned compression gates, so ``FastH3WeightFusion``
(LoRA-style deltas) never applies to them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .fasth3 import FASTH3_BASE_MODEL, _resolve_dit_attention_backend

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

FASTH3_V2_MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
FASTH3_V2_BASE_SCHEDULE = DMD2SigmaSchedule.from_positions((0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0))

# PDD dense export contract (FastVideo/FastH3-Omni-FP8).
FASTH3_PDD_SCHEMA = "fasth3-inference-contract-v1"
FASTH3_PDD_MODEL_TYPE = "ref2va"
FASTH3_PDD_CONDITIONING = "fixed_ordered_references_target_only_flow"
# Runtime node sigmas of the PDD-8 plan (8 steps over the [0.999, 0] fine grid
# with the video shift); the pipeline's default step count is 8 either way and
# the PDD path replaces the schedule with the plan's own node sigmas.
FASTH3_PDD_BASE_SCHEDULE = (
    1.0,
    0.9872470588235294,
    0.972,
    0.9514285714285714,
    0.9221538461538461,
    0.8771707317073171,
    0.7992,
    0.6309469509269506,
    0.0,
)


@dataclass(frozen=True)
class FastH3CheckpointSpec:
    """The released checkpoint's own schedule and trained attention policy.

    This is independent of ``FastH3WeightFusion``: the released weights already
    contain the entire student, including its learned compression gates.
    """

    vsa_sparsity: float = 0.8
    vsa_tile_size: int = 64
    # PDD dense export only; zero selects the legacy V2 behaviour.
    pdd_steps: int = 0
    pdd_step_indices: tuple[int, ...] = ()
    pdd_video_shift: float = 12.0
    pdd_audio_shift: float = 3.0
    pdd_block_sigmas: tuple[float, ...] = field(default_factory=tuple)

    @property
    def is_pdd(self) -> bool:
        return self.pdd_steps > 0

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> FastH3CheckpointSpec:
        """Validate the release's own fastvideo_inference.json."""
        if metadata.get("pdd_steps") is not None or metadata.get("schema") == FASTH3_PDD_SCHEMA:
            return cls._from_pdd_metadata(metadata)
        required = {
            "model_id": FASTH3_V2_MODEL_ID,
            "schema_version": "fasth3-inference-contract-v1",
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "transformer_forwards": 8,
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
            "guidance_scale": 1.0,
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "vsa_sparsity": 0.8,
            "vsa_tile_size": 64,
            "task": "t2av",
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"unsupported FastH3 contract: {key}={metadata.get(key)!r}, expected {expected!r}")
        return cls()

    @classmethod
    def _from_pdd_metadata(cls, metadata: Mapping[str, object]) -> FastH3CheckpointSpec:
        """Validate the PDD-8 dense export contract."""
        schema = metadata.get("schema") or metadata.get("schema_version")
        if schema != FASTH3_PDD_SCHEMA:
            raise ValueError(f"unsupported FastH3 PDD contract schema: {schema!r}")
        required = {
            "pdd_steps": 32,
            "num_inference_steps": 8,
            "transformer_forwards": 8,
            "grid_max_t": 0.999,
            "video_scheduler_shift": 12.0,
            "audio_scheduler_shift": 3.0,
            "guidance_scale": 1.0,
            "model_type": FASTH3_PDD_MODEL_TYPE,
            "conditioning": FASTH3_PDD_CONDITIONING,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"unsupported FastH3 PDD contract: {key}={metadata.get(key)!r}, expected {expected!r}")
        indices = metadata.get("pdd_step_indices")
        expected_indices = list(range(0, 33, 4))
        if list(indices or []) != expected_indices:
            raise ValueError(
                f"FastH3 PDD requires pdd_step_indices={expected_indices}, got {indices!r}"
            )
        return cls(
            vsa_sparsity=float(metadata.get("vsa_sparsity", 0.9)),
            vsa_tile_size=int(metadata.get("vsa_tile_size", 128)),
            pdd_steps=32,
            pdd_step_indices=tuple(expected_indices),
            pdd_video_shift=12.0,
            pdd_audio_shift=3.0,
            pdd_block_sigmas=FASTH3_PDD_BASE_SCHEDULE,
        )

    def release_metadata(self) -> dict[str, Any]:
        """Express the sampling policy in the existing H3 pipeline schema."""
        if self.is_pdd:
            return {
                "partition": "ref2va",
                "tasks": ["ref2va"],
                "sigma_shift_scales": {"video": self.pdd_video_shift, "audio": self.pdd_audio_shift},
                "base_schedule": list(self.pdd_block_sigmas),
            }
        return {
            "partition": "fl2va",
            "tasks": ["t2va"],
            "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
            "base_schedule": list(FASTH3_V2_BASE_SCHEDULE.base_schedule),
        }

    def resolve_native_vaes(self, model_root: Path) -> Path:
        """Reuse the frozen base VAEs with Omni's native tiled/parallel runtime.

        PDD releases mirror the frozen base VAEs next to the student, so the
        local root is used directly when present; the V2 release instead pins
        the base revision and only fetches the two VAE components.
        """
        if self.is_pdd:
            if (model_root / "video_vae").is_dir() and (model_root / "audio_vae").is_dir():
                return model_root
            raise ValueError(
                f"FastH3 PDD model root {model_root} must provide video_vae/ and audio_vae/ "
                "(frozen MiniMax-H3 base VAEs)"
            )
        provenance = json.loads((model_root / "provenance.json").read_text(encoding="utf-8"))
        base = provenance.get("base_model", "")
        prefix = f"hf://{FASTH3_BASE_MODEL}@"
        if not isinstance(base, str) or not base.startswith(prefix) or not base[len(prefix) :]:
            raise ValueError("FastH3 V2 provenance must pin a MiniMaxAI/MiniMax-H3 base revision")
        return (
            Path(
                download_weights_from_hf_specific(
                    model_name_or_path=FASTH3_BASE_MODEL,
                    cache_dir=None,
                    allow_patterns=["FL2VA/video_vae/**", "FL2VA/audio_vae/**"],
                    revision=base[len(prefix) :],
                    require_all=True,
                )
            )
            / "FL2VA"
        )

    def check_serving_contract(self, *, partition: str, od_config: OmniDiffusionConfig) -> None:
        if self.is_pdd:
            if partition != "ref2va":
                raise ValueError("FastH3 PDD requires --task-type ref2va (ordered-reference conditioning)")
            if getattr(od_config, "lora_path", None):
                raise ValueError("FastH3 PDD is a full checkpoint; additional LoRA adapters are unsupported")
            return
        if partition != "fl2va":
            raise ValueError("FastH3 V2 requires --task-type fl2va (T2VA requests only)")
        if getattr(od_config, "lora_path", None):
            raise ValueError("FastH3 V2 is a full checkpoint; additional LoRA adapters are unsupported")
        if _resolve_dit_attention_backend(od_config) != "FASTVIDEO_VSA":
            raise ValueError("FastH3 V2 requires --diffusion-attention-backend FASTVIDEO_VSA")
        attention_config = getattr(od_config, "diffusion_attention_config", None)
        per_role = getattr(attention_config, "per_role", None) or {}
        spec = per_role.get("self") or getattr(attention_config, "default", None)
        if getattr(spec, "fastvideo_vsa_topk", None) is not None:
            raise ValueError("FastH3 V2 pins VSA sparsity=0.8; remove the fixed fastvideo_vsa_topk override")
        parallel = getattr(od_config, "parallel_config", None)
        if any(int(getattr(parallel, key, 1) or 1) != 1 for key in ("ring_degree", "allgather_degree")):
            raise ValueError("FastH3 V2 supports local attention or pure Ulysses sequence parallelism")

    def check_request(self, sampling: OmniDiffusionSamplingParams, *, step_execution: bool = False) -> None:
        steps_expected = self.num_inference_steps
        if step_execution:
            if sampling.num_inference_steps != steps_expected:
                raise OmniClientError(
                    f"FastH3 requires num_inference_steps={steps_expected} for step execution"
                )
            if sampling.timesteps is not None or sampling.sigmas is not None:
                raise OmniClientError("FastH3 does not support custom timesteps or sigmas")
        if sampling.lora_request is not None:
            raise OmniClientError("FastH3 does not support per-request LoRA adapters")
        steps = sampling.num_inference_steps
        if steps is not None and steps != steps_expected:
            raise OmniClientError(
                f"FastH3 requires num_inference_steps={steps_expected}, got {steps}"
            )
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", self.video_shift), ("audio_flow_shift", self.audio_shift)):
            try:
                value = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"FastH3 requires {key}={expected:g}") from exc
            if not math.isclose(value, expected):
                raise OmniClientError(f"FastH3 requires {key}={expected:g}, got {value:g}")
        if sampling.guidance_scale is not None and sampling.guidance_scale != 1.0:
            raise OmniClientError("FastH3 requires guidance_scale=1")

    @property
    def num_inference_steps(self) -> int:
        return 8

    @property
    def video_shift(self) -> float:
        return self.pdd_video_shift if self.is_pdd else 10.0

    @property
    def audio_shift(self) -> float:
        return self.pdd_audio_shift if self.is_pdd else 3.0


__all__ = [
    "FASTH3_PDD_SCHEMA",
    "FASTH3_V2_BASE_SCHEDULE",
    "FASTH3_V2_MODEL_ID",
    "FastH3CheckpointSpec",
]
