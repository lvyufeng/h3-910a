#!/usr/bin/env python3
"""Isolated BF16->FP32 LayerNorm compatibility probe for the actual vision widths.

Do not invoke the already-known failing native BF16 LayerNorm on this context.
Compare composed FP32 NPU LayerNorm (BF16 I/O/weights retained) against both CPU
native BF16 LayerNorm and the same CPU FP32 composition. No FP16 substitution.
"""
import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
import torch_npu

OUT = Path("/tmp/h3_round50_layernorm_probe01.json")
if OUT.exists():
    raise RuntimeError("Refuse to overwrite previous probe evidence")
torch.npu.set_device(0)
torch.manual_seed(50)
result = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "device": 0,
          "input_output_dtype": "bfloat16", "compute_dtype": "float32", "cases": []}
try:
    cast_cpu = torch.tensor([-131072, -3.25, -.001, 0., .001, 1., 65536, 131072], dtype=torch.bfloat16)
    cast_npu = cast_cpu.to("npu")
    expanded = cast_npu.float()
    rounded = expanded.to(torch.bfloat16)
    torch.npu.synchronize()
    result["cast_roundtrip_exact"] = bool(torch.equal(cast_cpu, rounded.cpu()))
    result["cast_fp32_exact"] = bool(torch.equal(cast_cpu.float(), expanded.cpu()))
    assert result["cast_roundtrip_exact"] and result["cast_fp32_exact"]
    print("BF16 <-> FP32 cast values are exact on this NPU", flush=True)
    for width, rows, mode in ((1152, 64, "normal"), (4608, 16, "normal"),
                              (1152, 64, "constant"), (1152, 64, "high_dynamic")):
        x = torch.randn(rows, width)
        if mode == "constant":
            x.fill_(13.)
        elif mode == "high_dynamic":
            x = x * 8192 + 131072
        x = x.to(torch.bfloat16)
        weight = (torch.randn(width) * .05 + 1).to(torch.bfloat16)
        bias = (torch.randn(width) * .01).to(torch.bfloat16)
        native_cpu = F.layer_norm(x, (width,), weight, bias, 1e-6)
        composed_cpu = F.layer_norm(x.float(), (width,), weight.float(), bias.float(), 1e-6).to(torch.bfloat16)
        xn, wn, bn = [t.to("npu") for t in (x, weight, bias)]
        def run():
            return F.layer_norm(xn.float(), (width,), wn.float(), bn.float(), 1e-6).to(torch.bfloat16)
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
        y_cpu = y.cpu()
        error = (y_cpu.float() - native_cpu.float()).abs()
        relative_rms = (error.square().mean().sqrt() / native_cpu.float().square().mean().sqrt().clamp_min(1e-12)).item()
        case = {"width": width, "rows": rows, "mode": mode, "median_ms": statistics.median(samples),
                "finite": bool(torch.isfinite(y_cpu).all()),
                "cpu_native_vs_composed_exact": bool(torch.equal(native_cpu, composed_cpu)),
                "npu_vs_cpu_native_exact": bool(torch.equal(y_cpu, native_cpu)),
                "max_abs_error": error.max().item(), "relative_rms": relative_rms,
                "dtype": str(y.dtype)}
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
        assert case["finite"] and y.dtype == torch.bfloat16
        assert relative_rms < .005, case
except BaseException as exc:
    result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("Evidence:", OUT, flush=True)
