#!/usr/bin/env python3
"""R233 driver: in-model A/B of the guarded linear with host sync removed.

H3 (from R232): the guarded layers perform 100 host synchronisations per step
(`out_proj` and `fc2` each 50x). R232 measured `out_proj` at 18.2 ms device time
but 179.6 ms host wall, i.e. the host parks on the guard's `float(peak)` and then
has to catch up; the resulting bubbles appear as the ~3.4 s/step that is still
unattributed.

Variant B keeps the deployed mathematics but computes the power-of-two scale on
the device, so no host read happens inside the layer:

    peak_t    = product.abs().amax(...)              # same reduction
    ratio     = clamp(peak_t / 32752, min=1e-38)
    exponent  = clamp(ceil(log2(ratio)), min=0)      # same rule, device side
    scale_t   = 2 ** exponent                        # power of two, FP32 exact
    half      = (product / scale_t).to(float16)      # exact for scale==1
    result    = mm(half, weight.t(), out_dtype=float32) * scale_t

R221 already showed on synthetic input that this is bit-identical to the deployed
path for the scale==1 branch (112,336,896 words, 0 differences). The nonfinite gate
is preserved by accumulating a device-side mask and reading it once at the end of
the step instead of once per call.

This run is measurement only: no source change, no candidate installed, one step.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_GUARD233_LABEL", "guard233b")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(local_rank)
    import h3_replay123.transformer as T

    cls = T._MiniMaxH3HalfInputFP32OutputLinearMethod
    original_apply = cls.apply
    state = {"calls": 0, "nonfinite_mask": torch.zeros((), dtype=torch.int32, device="npu")}

    def apply_devscale(self, layer, x, bias=None):
        if x.device.type not in ("cuda", "npu") or layer.weight.dtype != torch.float16:
            raise TypeError("This trial requires FP16 matrix weights")
        state["calls"] += 1
        product = x.float()
        peak_t = product.abs().amax(dim=tuple(range(product.ndim)))
        bad = (~torch.isfinite(peak_t)).to(torch.int32)
        state["nonfinite_mask"] = torch.maximum(state["nonfinite_mask"], bad)
        ratio = torch.clamp(peak_t / 32752.0, min=1e-38)
        exponent_t = torch.clamp(torch.ceil(torch.log2(ratio)), min=0.0)
        scale_t = torch.pow(torch.full_like(exponent_t, 2.0), exponent_t)
        half_input = (product / scale_t).to(torch.float16)
        result = torch.mm(
            half_input.reshape(-1, half_input.shape[-1]),
            layer.weight.t(), out_dtype=torch.float32,
        ).reshape(*x.shape[:-1], layer.weight.shape[0])
        result = result * scale_t
        if bias is not None:
            result = result + bias.float()
        return result

    cls.apply = apply_devscale

    sys.argv = [
        "benchmark_actual_attention126.py",
        "--checkpoint", "/home/mseco/models/MiniMax-H3/FL2VA/transformer",
        "--mode", "baseline",
        "--label", LABEL,
        "--tp", "2", "--usp", "4", "--offload", "native",
        "--audit-sequence", "strict",
        "--steps", "0", "--warmup", "1", "--repeats", "1",
    ]
    import benchmark_actual_attention126 as bench

    bench.ROOT = ROOT
    bench.main()

    receipt = out / f"candidate_rank{rank}.json"
    data = json.loads(receipt.read_text()) if receipt.exists() else {}
    data["guard233"] = {
        "variant": "device-side scale, no host sync inside the guarded linear",
        "guarded_calls_seen": state["calls"],
        "nonfinite_peak_seen": bool(int(state["nonfinite_mask"])),
        "measurement_only": True,
        "candidate_installed": False,
        "source_changed_on_disk": False,
        "bitwise_basis": "R221 synthetic proof for the scale==1 branch; in-model "
                         "bitwise comparison across runs is not possible because native "
                         "self-repeat is not deterministic (issue #15)",
    }
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["guard233"]["measured_rank_max_s"] = measured
    data["guard233"]["reference_baseline_s"] = 17.9465
    data["guard233"]["delta_s"] = None if measured is None else round(measured - 17.9465, 3)
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank, "step_s": measured,
                      "guarded_calls": state["calls"],
                      "nonfinite": bool(int(state["nonfinite_mask"]))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())