#!/usr/bin/env python3
"""R253: three-way combination, one measured step.

  A) R218 FA padding-task trim   (user-approved, minus 0.568 s standalone)
  B) FA tiles 64/912             (user-approved, 1.12x standalone kernel)
  C) guard fast path, scale==1   (bitwise-identical fast path, minus 0.201 s)

All three are applied in-process only; no source file on disk is modified. The FA call site
resolves `fa910a.flash_attention_910a` late, so replacing the module attribute is enough.
Audit stays strict, blocks/steps/dtypes/FP32 parameters/155 checks untouched.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_COMBO253_LABEL", "combo253a")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(local_rank)

    # A + B: R218 trim implementation with the approved 64/912 tiles (env-driven)
    from vllm_omni.diffusion.attention.backends import fa910a as deployed
    import bb218trim as trim_mod

    deployed.flash_attention_910a = trim_mod.flash_attention_910a

    # C: guard fast path
    import h3_replay123.transformer as T

    cls = T._MiniMaxH3HalfInputFP32OutputLinearMethod
    original_apply = cls.apply
    stats = {"fast_calls": 0, "fallback_calls": 0}

    def apply_fast(self, layer, x, bias=None):
        if x.device.type not in ("cuda", "npu") or layer.weight.dtype != torch.float16:
            raise TypeError("This trial requires FP16 matrix weights")
        peak = float(x.abs().amax(dim=tuple(range(x.ndim))))
        if not math.isfinite(peak):
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

    sys.argv = [
        "benchmark_actual_attention126.py",
        "--checkpoint", "/home/mseco/models/MiniMax-H3/FL2VA/transformer",
        "--mode", "baseline", "--label", LABEL,
        "--tp", "2", "--usp", "4", "--offload", "native",
        "--audit-sequence", "strict",
        "--steps", "0", "--warmup", "1", "--repeats", "1",
    ]
    import benchmark_actual_attention126 as bench

    bench.ROOT = ROOT
    bench.main()

    receipt = out / f"candidate_rank{rank}.json"
    data = json.loads(receipt.read_text()) if receipt.exists() else {}
    ch = (data.get("checks") or [{}])[0]
    measured = ch.get("rank_max_elapsed_s")
    video = ch.get("video_vs_R124_baseline") or {}
    data["combo253"] = {
        "components": {"fa_trim": "R218 padding-task trim",
                       "fa_tiles": [int(os.environ.get("H3_FA_BR_CAP", 64)),
                                    int(os.environ.get("H3_FA_BC_CAP", 912))],
                       "guard_fast_path": "scale==1 fast path, bitwise identical"},
        "guard_calls": stats,
        "deployed_source_changed": False,
        "user_approved_components": ["R218 FA trim", "FA tile 64/912"],
        "guard_fast_path_numeric": "bitwise identical to deployed path on the scale==1 "
                                   "branch (synthetic gate 393,216 words, 0 diffs)",
    }
    data["combo253_invariance"] = {
        "measured_rank_max_s": measured, "baseline_s": 17.928,
        "expected_s": 16.30,
        "delta_vs_baseline_s": None if measured is None else round(measured - 17.928, 3),
        "target_s": 15.0,
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank, "step_s": measured, "guard": stats,
                      "video_relRMS": video.get("relative_RMS_vs_R124_NPU_baseline"),
                      "max_abs_diff": video.get("max_abs_difference")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
