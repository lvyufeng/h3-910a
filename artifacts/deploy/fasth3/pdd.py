# SPDX-License-Identifier: Apache-2.0
"""Parallel Decoding Distillation (PDD) sampling contract for FastH3.

FastH3 is the PDD-8 distillation of MiniMax-H3: the final projection of the
transformer is widened to ``pdd_steps`` heads (one prediction per interval of a
fixed fine time grid), and the runtime sampler only evaluates the transformer
``num_steps`` times, fusing the heads of one contiguous fine-grid block inside
the final projection.

Conventions (mirrors the released FastVideo implementation):

* The fine grid lives on the *base* (unshifted) clock
  ``u_0 = max_t > ... > u_N = 0`` with ``N = pdd_steps``.
* Each modality reaches its own noise level through the rational shift
  ``f_s(u) = s * u * M / (u * (s - 1) + M)``, ``M = max_t = 0.999``.
  MiniMax-H3 keeps its native shifts (video 12, audio 3).
* Head ``j`` is frozen on interval ``j`` and predicts the native rectified-flow
  direction ``q_j = eps - x0 = dx/dsigma``.  Advancing the state over a block is
  ``x += sum_j (sigma(u_{j+1}) - sigma(u_j)) * q_j``, which the fused forward
  computes as ``x += block_total * (normalized weighted mean of the heads)``.
* The model's time input is ``1 - sigma``; it predicts ``v = eps - x0`` (the
  same convention as the non-distilled H3 denoise loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

PDD_GRID_MAX_T = 0.999
# MiniMax-H3 keeps its native rectified-flow shifts for both modalities.
PDD_VIDEO_SHIFT = 12.0
PDD_AUDIO_SHIFT = 3.0
PDD_PRIMARY_MODALITY = "video"


def _validate_shift(shift: float) -> float:
    if not isinstance(shift, (int, float)) or isinstance(shift, bool) or shift <= 0:
        raise ValueError(f"shift must be a positive number, got {shift!r}")
    return float(shift)


def shifted_noise_amount(
    base: torch.Tensor, shift: float, *, max_t: float = PDD_GRID_MAX_T
) -> torch.Tensor:
    """``f_s(u) = s * u * M / (u * (s - 1) + M)`` on ``[0, M]``."""
    shift = _validate_shift(shift)
    if not 0.0 < max_t <= 1.0:
        raise ValueError(f"max_t must satisfy 0 < max_t <= 1, got {max_t}")
    if shift == 1.0:
        return base
    return shift * base * max_t / (base * (shift - 1.0) + max_t)


def shifted_noise_delta(
    base_start: torch.Tensor,
    base_end: torch.Tensor,
    shift: float,
    *,
    max_t: float = PDD_GRID_MAX_T,
) -> torch.Tensor:
    """``f_s(end) - f_s(start)`` without subtractive cancellation."""
    shift = _validate_shift(shift)
    if shift == 1.0:
        return base_end - base_start
    denominator_start = max_t + (shift - 1.0) * base_start
    denominator_end = max_t + (shift - 1.0) * base_end
    return shift * max_t**2 * (base_end - base_start) / (denominator_start * denominator_end)


def unshifted_noise_amount(
    sigma: torch.Tensor, shift: float, *, max_t: float = PDD_GRID_MAX_T
) -> torch.Tensor:
    """Base time whose shifted noise amount is *sigma*: ``f_(1/s)(sigma)``."""
    return shifted_noise_amount(sigma, 1.0 / _validate_shift(shift), max_t=max_t)


def pdd_fine_grid(
    pdd_steps: int,
    *,
    max_t: float = PDD_GRID_MAX_T,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Descending base-clock grid ``u_0 = max_t > ... > u_N = 0`` in float64."""
    if isinstance(pdd_steps, bool) or not isinstance(pdd_steps, int) or pdd_steps < 2:
        raise ValueError(f"pdd_steps must be an int >= 2, got {pdd_steps!r}")
    grid = torch.linspace(max_t, 0.0, pdd_steps + 1, dtype=torch.float64, device=device)
    return grid.clamp(max=max_t)


def pdd_student_step_indices(
    source_steps: int, target_steps: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Fine-grid node indices of the balanced ``target_steps`` partition.

    ``arange(n + 1) * N // n`` matches FastGen's
    ``BaseNoiseSchedule._student_step_indices``; for the released FastH3
    (``N = 32``, ``n = 8``) this is exactly ``[0, 4, 8, ..., 32]``.
    """
    if source_steps < 1 or target_steps < 1:
        raise ValueError("source_steps and target_steps must be positive")
    if target_steps > source_steps:
        raise ValueError(f"cannot select {target_steps} steps from a {source_steps}-step grid")
    return torch.arange(target_steps + 1, dtype=torch.long, device=device) * source_steps // target_steps


@dataclass(frozen=True)
class PDDModalitySchedule:
    """One modality's noise level as a function of the base clock."""

    shift: float
    max_t: float = PDD_GRID_MAX_T

    def __post_init__(self) -> None:
        object.__setattr__(self, "shift", _validate_shift(self.shift))
        if not 0.0 < self.max_t <= 1.0:
            raise ValueError(f"max_t must satisfy 0 < max_t <= 1, got {self.max_t}")

    def sigma(self, base: torch.Tensor) -> torch.Tensor:
        return shifted_noise_amount(base, self.shift, max_t=self.max_t)

    def sigma_delta(self, base_start: torch.Tensor, base_end: torch.Tensor) -> torch.Tensor:
        return shifted_noise_delta(base_start, base_end, self.shift, max_t=self.max_t)

    def unwarp(self, sigma: torch.Tensor) -> torch.Tensor:
        return unshifted_noise_amount(sigma, self.shift, max_t=self.max_t)

    def integration_weights(self, fine_grid: torch.Tensor) -> torch.Tensor:
        return self.sigma_delta(fine_grid[:-1], fine_grid[1:])


@dataclass(frozen=True)
class PDDSamplingPlan:
    """One runtime partition of the fine grid plus every modality's coefficients."""

    fine_grid: torch.Tensor
    indices: torch.Tensor
    node_sigmas: Mapping[str, torch.Tensor]
    integration_weights: Mapping[str, torch.Tensor]

    @property
    def pdd_steps(self) -> int:
        return int(self.fine_grid.numel()) - 1

    @property
    def num_steps(self) -> int:
        return int(self.indices.numel()) - 1

    def block(self, step: int) -> tuple[int, int]:
        if not 0 <= step < self.num_steps:
            raise ValueError(f"step must be in [0, {self.num_steps - 1}], got {step}")
        return int(self.indices[step]), int(self.indices[step + 1])

    def block_total(self, modality: str, step: int) -> float:
        start, end = self.block(step)
        return float(self.integration_weights[modality][start:end].sum())

    def normalized_block_weights(self, modality: str, step: int, *, dtype: torch.dtype) -> torch.Tensor:
        start, end = self.block(step)
        weights = self.integration_weights[modality][start:end]
        total = weights.sum()
        if not bool(torch.isfinite(total)) or float(total) == 0.0:
            raise ValueError(f"PDD {modality} block {step} has a non-finite or zero total weight")
        return (weights / total).to(dtype=dtype)


def build_pdd_sampling_plan(
    *,
    pdd_steps: int,
    num_steps: int,
    schedules: Mapping[str, PDDModalitySchedule],
    device: torch.device | str | None = None,
) -> PDDSamplingPlan:
    """Build the runtime plan for a ``pdd_steps``-head checkpoint run in ``num_steps``."""
    if not schedules:
        raise ValueError("at least one modality schedule is required")
    max_ts = {schedule.max_t for schedule in schedules.values()}
    if len(max_ts) != 1:
        raise ValueError(f"every PDD modality schedule must share max_t, got {sorted(max_ts)}")
    fine_grid = pdd_fine_grid(pdd_steps, max_t=next(iter(max_ts)), device=device)
    indices = pdd_student_step_indices(pdd_steps, num_steps, device=fine_grid.device)
    integration_weights = {}
    node_sigmas = {}
    for name, schedule in schedules.items():
        weights = schedule.integration_weights(fine_grid)
        if not bool(torch.all(torch.isfinite(weights))) or not bool(torch.all(weights != 0)):
            raise ValueError(f"PDD {name} integration weights must be finite and non-zero")
        integration_weights[name] = weights
        node_sigmas[name] = schedule.sigma(fine_grid[indices])
    return PDDSamplingPlan(
        fine_grid=fine_grid,
        indices=indices,
        node_sigmas=node_sigmas,
        integration_weights=integration_weights,
    )


def fuse_head_rows(
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_rows: int,
    step: int,
    plan: PDDSamplingPlan,
    modality: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fuse one PDD block of a widened ``[num_heads * head_rows, ...]`` weight.

    Returns the effective ``[head_rows, ...]`` weight whose output equals the
    normalized weighted mean of the block's head outputs:
    ``(sum_j w_j W_j) @ x`` with ``w`` normalized inside the block.
    """
    flat = weight.reshape(num_heads * head_rows, *weight.shape[1:])
    start, end = plan.block(step)
    weights = plan.normalized_block_weights(modality, step, dtype=dtype or weight.dtype)
    weight_device = weight.device
    heads = flat.to(dtype=weights.dtype).unflatten(0, (num_heads, head_rows))[start:end]
    weights = weights.to(device=weight_device)
    fused = torch.einsum("n,n...->...", weights, heads)
    return fused.to(dtype=weight.dtype if dtype is None else dtype)


def fused_block_heads(
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_rows: int,
    plan: PDDSamplingPlan,
    modality: str,
) -> torch.Tensor:
    """All fused heads of a widened projection: ``[num_steps * head_rows, ...]``."""
    steps = [
        fuse_head_rows(
            weight,
            num_heads=num_heads,
            head_rows=head_rows,
            step=step,
            plan=plan,
            modality=modality,
        )
        for step in range(plan.num_steps)
    ]
    return torch.cat(steps, dim=0)


__all__ = [
    "PDD_AUDIO_SHIFT",
    "PDD_GRID_MAX_T",
    "PDD_PRIMARY_MODALITY",
    "PDD_VIDEO_SHIFT",
    "PDDModalitySchedule",
    "PDDSamplingPlan",
    "build_pdd_sampling_plan",
    "fuse_head_rows",
    "fused_block_heads",
    "pdd_fine_grid",
    "pdd_student_step_indices",
    "shifted_noise_amount",
    "shifted_noise_delta",
    "unshifted_noise_amount",
]
