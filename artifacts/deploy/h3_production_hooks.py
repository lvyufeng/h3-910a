"""Production hot path for the H3 service.

The R126 alignment hooks (:mod:`integration_hooks126`) are written for a one-shot
acceptance run: every DiT step copies both predictions to the CPU for statistics,
writes audit events, and runs an exhaustive Half-range count on each of the 155
matrix modules. Those are *observations* -- they never mutate a tensor -- but they
cost roughly 2 s per DiT evaluation (155 device synchronisations per step plus CPU
readbacks). That is the gap between the isolated benchmark (15.55 s/eval) and the
service (17.6 s/eval).

This module keeps every hook that changes numerics -- FA patch, FP32 projection
and time guards, exact initial noise, FP32 VAE parameters, NaN compatibility
scope, VAE operand cache -- and replaces only the observation-only part:

* the DiT step performs no ``stats()`` CPU readback and no ``event()`` write;
* the 155 per-module exhaustive counts are removed in favour of a single
  end-to-end finiteness / saturation gate on the returned predictions.

``H3_SERVICE_STRICT_HALF_AUDIT=1`` keeps the original acceptance audit untouched
(parity mode); ``H3_SERVICE_PROD_HOTPATH=0`` disables this module entirely.
"""
from __future__ import annotations

import os

_APPLIED: list[str] = []
_HALF_MAX = 65504.0


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "off", "no")


def install() -> list[str]:
    """Install the production overrides once per process."""
    if _APPLIED:
        return list(_APPLIED)
    if not _flag("H3_SERVICE_PROD_HOTPATH", "1"):
        _APPLIED.append("prod_hotpath(disabled)")
        return list(_APPLIED)
    if _flag("H3_SERVICE_STRICT_HALF_AUDIT", "0"):
        _APPLIED.append("prod_hotpath(strict-acceptance-audit)")
        return list(_APPLIED)
    try:
        import torch

        import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline
    except Exception as exc:  # pragma: no cover - deployment dependent
        print(f"[h3-prod] hot path not installed: {exc!r}", flush=True)
        return []

    _replace_dit_forward(pipeline, torch)
    return list(_APPLIED)


def _replace_dit_forward(pipeline, torch) -> None:
    cls = pipeline.MiniMaxH3DiTModel
    if getattr(cls.forward, "_h3_production", False):
        _APPLIED.append("dit_step_audit(already)")
        return
    verified = cls.__mro__[1]  # h3_replay123 MiniMaxH3DiTModel (numeric path)

    def forward(self, **kwargs):
        # The compatibility adapter pops geometry-only metadata before the
        # verified forward; the production path keeps exactly that behaviour.
        kwargs.pop("video_token_layout", None)
        video, audio = verified.forward(self, **kwargs)
        for name, value in (("video", video), ("audio", audio)):
            if isinstance(value, torch.Tensor):
                if not bool(torch.isfinite(value).all()):  # pragma: no cover - failure path
                    raise FloatingPointError(f"non-finite {name} prediction in production hot path")
                if float(value.abs().max()) >= _HALF_MAX:  # pragma: no cover - failure path
                    raise FloatingPointError(f"{name} prediction reached the FP16 saturation boundary")
        self._actual_step124 += 1
        return video, audio

    forward._h3_production = True
    cls.forward = forward

    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        removed = 0
        for module in self.modules():
            for key, hook in list(module._forward_hooks.items()):
                if getattr(hook, "__name__", "") == "verify_half_matrix_output":
                    del module._forward_hooks[key]
                    removed += 1
        self._h3_service_half_hooks_removed = removed

    patched_init._h3_production = True
    cls.__init__ = patched_init
    _APPLIED.append("dit_step_audit(stripped:end-to-end-finite-gate)")