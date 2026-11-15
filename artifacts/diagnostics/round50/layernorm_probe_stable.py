#!/usr/bin/env python3
"""Diagnose degenerate LayerNorm rounding and validate stable FP32 normalization.

BF16 input/weights/output remain unchanged. A first-element shift makes the
centered variance reduction invariant to constant offsets before any mean
reduction. Compare to an explicit float64 centered-variance reference as well as
native CPU BF16. This is not an FP16 precision substitution.
"""
import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
import torch_npu


def stable_norm(x, w, b, eps=1e-6):
    shifted = x.float() - x.float()[..., :1]
    centered = shifted - shifted.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + eps)
    if w is not None:
        normalized = normalized * w.float()
    if b is not None:
        normalized = normalized + b.float()
    return normalized.to(x.dtype)


def reference64(x, w, b, eps=1e-6):
    xd = x.double()
    centered = xd - xd.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    out = centered / torch.sqrt(var + eps)
    return (out * w.double() + b.double()).to(x.dtype)


OUT = Path("/tmp/h3_round50_layernorm_probe02.json")
if OUT.exists():
    raise RuntimeError("Refuse to overwrite previous evidence")
torch.npu.set_device(0)
torch.manual_seed(50)
result = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "device": 0,
          "input_output_dtype": "bfloat16", "compute_dtype": "float32", "cases": []}
try:
    for width, rows, mode in ((1152, 64, "normal"), (4608, 16, "normal"),
                              (1152, 64, "constant"), (1152, 64, "high_dynamic"),
                              (1152, 64, "zero"), (4608, 16, "high_dynamic")):
        x = torch.randn(rows, width)
        if mode == "constant":
            x.fill_(13.)
        elif mode == "zero":
            x.zero_()
        elif mode == "high_dynamic":
            x = x * 8192 + 131072
        x = x.to(torch.bfloat16)
        w = (torch.randn(width) * .05 + 1).to(torch.bfloat16)
        b = (torch.randn(width) * .01).to(torch.bfloat16)
        ref = reference64(x, w, b)
        native = F.layer_norm(x, (width,), w, b, 1e-6)
        comp = F.layer_norm(x.float(), (width,), w.float(), b.float(), 1e-6).to(x.dtype)
        xn, wn, bn = [t.to("npu") for t in (x, w, b)]
        simple_npu = F.layer_norm(xn.float(), (width,), wn.float(), bn.float(), 1e-6).to(x.dtype).cpu()
        def run():
            return stable_norm(xn, wn, bn)
        for _ in range(2):
            y = run()
        torch.npu.synchronize()
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            for __ in range(5):
                y = run()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 5)
        yc = y.cpu()
        def metrics(t):
            err = (t.float() - ref.float()).abs()
            rms = err.square().mean().sqrt()
            return {"max_abs_error_vs_float64": err.max().item(), "rms_error": rms.item(),
                    "relative_rms": (rms / ref.float().square().mean().sqrt().clamp_min(1e-12)).item(),
                    "exact": bool(torch.equal(t, ref))}
        case = {"width": width, "rows": rows, "mode": mode, "stable_median_ms": statistics.median(samples),
                "finite": bool(torch.isfinite(yc).all()), "dtype": str(y.dtype),
                "native_cpu_bf16": metrics(native), "simple_cpu_fp32": metrics(comp),
                "simple_npu_fp32": metrics(simple_npu), "stable_npu_fp32": metrics(yc)}
        if mode in ("constant", "zero"):
            case["stable_equals_bias"] = bool(torch.equal(yc, b.expand_as(yc)))
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
        assert case["finite"] and y.dtype == torch.bfloat16
        assert case["stable_npu_fp32"]["relative_rms"] < .001, case
        if mode in ("constant", "zero"):
            assert case["stable_equals_bias"], case
except BaseException as exc:
    result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("Evidence:", OUT, flush=True)
