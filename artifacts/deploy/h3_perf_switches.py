"""Switchable deployment wrapper for the three validated DiT speedups.

Landing decision (user 2026-11-25): ship the three measured components as a **single
switchable module**, default OFF, so the deployed behaviour is unchanged unless asked for.
Nothing here edits the shared vllm-omni installation; every change is an in-process
monkeypatch applied by the launcher before the model is built.

Enable with:   H3_PERF_COMBO=1
Components (each individually switchable, all default OFF):
    H3_PERF_FA_TILES=1     FA tiles 64/912            (user-approved; relRMS ~2e-4 standalone)
    H3_PERF_FA_TRIM=1      R218 padding-task trim     (user-approved; relRMS 1.02-1.03e-3)
    H3_PERF_GUARD_FAST=1   guard scale==1 fast path   (bitwise identical on that branch)

49-step validated result with all three on: 16.31-16.34 s/step (median 16.32 s), -9.0%
versus the 17.928 s baseline; every recorded step stayed inside the native self-repeat
numerical band (9.84e-04-1.077e-03).

Rollback is immediate: unset H3_PERF_COMBO (or any component flag) and re-launch.
"""
from __future__ import annotations

import math
import os

INSTALLED = {}


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("0", "", "false", "off", "no")


def install(torch, label: str = "") -> dict:
    """Apply the enabled components in-process. Returns a report for the receipt."""
    report = {"label": label, "combo": _flag("H3_PERF_COMBO"), "applied": [], "skipped": []}
    if not report["combo"]:
        report["skipped"].append("H3_PERF_COMBO is off; deployed behaviour unchanged")
        return report

    if _flag("H3_PERF_FA_TILES") or _flag("H3_PERF_FA_TRIM"):
        from vllm_omni.diffusion.attention.backends import fa910a as deployed

        if _flag("H3_PERF_FA_TRIM"):
            import fa218trim as trim_mod  # shipped alongside this module

            deployed.flash_attention_910a = trim_mod.flash_attention_910a
            os.environ.setdefault("H3_FA_TRIM_Q_LIMIT", "41792")
            report["applied"].append("fa_trim:R218_padding_task_trim")
        else:
            report["skipped"].append("fa_trim off")

        if _flag("H3_PERF_FA_TILES"):
            br = int(os.environ.get("H3_FA_BR_CAP", "64"))
            bc = int(os.environ.get("H3_FA_BC_CAP", "912"))
            os.environ["H3_FA_BR_CAP"] = str(br)
            os.environ["H3_FA_BC_CAP"] = str(bc)
            if not _flag("H3_PERF_FA_TRIM"):
                # trim off: keep the deployed kernel builder, only override its tile caps
                real_build = deployed._build

                def build_with_tiles(b, h, n, d, br_cap=br, bc_cap=bc, key_seq_limit=None):
                    return real_build(b, h, n, d, br, bc, key_seq_limit)

                deployed._build = build_with_tiles
            report["applied"].append(f"fa_tiles:{br}x{bc}")
        else:
            report["skipped"].append("fa_tiles off")

    if _flag("H3_PERF_GUARD_FAST"):
        import h3_replay123.transformer as T

        cls = T._MiniMaxH3HalfInputFP32OutputLinearMethod
        original_apply = cls.apply
        stats = {"fast_calls": 0, "fallback_calls": 0, "nonfinite_raise": 0}
        report["guard_stats"] = stats

        def apply_fast(self, layer, x, bias=None):
            if x.device.type not in ("cuda", "npu") or layer.weight.dtype != torch.float16:
                raise TypeError("This trial requires FP16 matrix weights")
            peak = float(x.abs().amax(dim=tuple(range(x.ndim))))
            if not math.isfinite(peak):
                stats["nonfinite_raise"] += 1
                raise FloatingPointError("non-finite guard peak")
            if peak > 32752.0:
                stats["fallback_calls"] += 1
                return original_apply(self, layer, x, bias)
            stats["fast_calls"] += 1
            result = torch.mm(
                x.reshape(-1, x.shape[-1]), layer.weight.t(), out_dtype=torch.float32,
            ).reshape(*x.shape[:-1], layer.weight.shape[0])
            if bias is not None:
                result = result + bias.float()
            return result

        cls.apply = apply_fast
        report["applied"].append("guard_fast:scale1_fastpath")
    else:
        report["skipped"].append("guard_fast off")

    INSTALLED.update(report)
    return report
