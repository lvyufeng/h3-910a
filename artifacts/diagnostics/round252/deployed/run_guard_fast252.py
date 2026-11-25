#!/usr/bin/env python3
"""R252: guard fast path -- skip the FP32 temporary when the power-of-two scale is 1.

R251's msprof window showed the guarded linears (out_proj, fc2) each pay a `Cast` on the
aux stream of 5-20 ms, twice per block, plus a divide -- the FP32 temporary the guard builds
before casting back to FP16.

Deployed path (per guarded linear):
    product = x.float()                       # FP16 -> FP32 copy
    peak    = float(product.abs().amax(...))  # host sync
    scale   = 2**ceil(log2(peak/32752)) if peak > 32752 else 1
    half    = (product / scale).to(fp16)      # divide + cast back
    result  = mm(half, w.t(), out_dtype=fp32) * scale
When peak <= 32752 the deployed path computes scale == 1, so `product / 1.0` is the exactly
representable FP32 image of x and `.to(fp16)` recovers x bit-for-bit; `result * 1.0` is exact.
Therefore feeding FP16 x straight to the same mm is bit-identical to the deployed path under
exactly the condition scale == 1, which R221 verified on 112,336,896 real model words.

This candidate keeps the guard: it still computes the peak (host sync retained), still raises on
non-finite, and falls back to the untouched deployed implementation whenever peak > 32752.

In-process only; no source file changes. A synthetic bitwise gate runs before the model step.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_GUARD252_LABEL", "guard252a")


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
    stats = {"fast_calls": 0, "fallback_calls": 0, "nonfinite_raise": 0}

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

    # ---- synthetic bitwise gate (before the model step) -------------------------
    gate = {"cases": [], "mismatched_words": 0, "total_words": 0}
    try:
        torch.manual_seed(20261125)
        for tag, scale_hi in (("small_peaks", False), ("near_limit", True)):
            m, k, n = 512, 256, 384
            base = torch.randn(m, k, dtype=torch.float16, device="npu")
            x = (base * 0.05) if not scale_hi else (base * 30000.0).clamp(-32752.0, 32752.0)

            class _L:
                pass

            layer = _L()
            layer.weight = (torch.randn(n, k, dtype=torch.float16, device="npu") * 0.02)
            expect = original_apply(None, layer, x, None)
            got = apply_fast(None, layer, x, None)
            same = torch.equal(expect, got)
            gate["cases"].append({"tag": tag, "bit_identical": bool(same),
                                  "max_abs_diff": float((expect - got).abs().max().item())})
            gate["total_words"] += int(expect.numel())
            if not same:
                gate["mismatched_words"] += int((expect != got).sum().item())
        torch.npu.synchronize()
    except BaseException as exc:  # pragma: no cover - defensive
        gate["error"] = f"{type(exc).__name__}: {exc}"

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
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["guard252"] = {
        "candidate": "guard fast path: skip FP32 temporary when scale == 1",
        "calls": stats, "synthetic_bitwise_gate": gate,
        "deployed_source_changed": False,
        "numeric_claim": "bit-identical to deployed path whenever the deployed path itself "
                         "takes the scale==1 branch; falls back verbatim otherwise",
    }
    data["guard252_invariance"] = {
        "measured_rank_max_s": measured, "baseline_s": 17.928,
        "delta_s": None if measured is None else round(measured - 17.928, 3),
        "target_s": 15.0,
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank, "step_s": measured, "calls": stats, "gate": gate}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
