#!/usr/bin/env python3
"""Test exact local layout conversion before the failing VAE reflection pad.

Each strategy runs in its own process. No global internal-format setting changes.
The native failure is expected and checked; ND and NCDHW paths must match CPU
padding and preserve every input/output value and dtype.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
import torch_npu

ap = argparse.ArgumentParser()
ap.add_argument("strategy", choices=["native", "nd", "ncdhw"])
ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
a = ap.parse_args()
root = Path("/tmp/h3_round51_pad_probe01")
root.mkdir(exist_ok=True)
out = root / f"{a.strategy}_{a.dtype}.json"
if out.exists():
    raise RuntimeError("Refuse to overwrite prior probe")
torch.npu.set_device(0)
torch.manual_seed(51)
dtype = getattr(torch, a.dtype)
result = {"strategy": a.strategy, "dtype": a.dtype, "source_format": 32,
          "global_allow_internal_format": getattr(torch.npu.config, "allow_internal_format", None),
          "global_settings_changed": False,
          "torch": torch.__version__, "torch_npu": torch_npu.__version__, "cases": []}
try:
    for shape in ((1, 3, 2, 7, 9), (1, 16, 1, 8, 10), (2, 17, 3, 8, 10)):
        cpu = torch.randn(shape).to(dtype)
        packed = torch_npu.npu_format_cast(cpu.to("npu"), 32)
        initial = int(torch_npu.get_npu_format(packed))
        assert initial == 32, initial
        pad = (0, 1, 0, 1, 0, 0)
        ref = F.pad(cpu.float(), pad, mode="reflect").to(dtype)
        if a.strategy == "native":
            try:
                y = F.pad(packed, pad, mode="reflect")
                torch.npu.synchronize()
            except RuntimeError as exc:
                assert "reflection_pad3d do not support internal format" in str(exc), str(exc)
                result["expected_native_failure"] = str(exc)
                print("Verified native packed-input reflection-pad failure", flush=True)
                break
            else:
                raise AssertionError("Expected the known internal-format failure")
        target_format = 2 if a.strategy == "nd" else 30
        converted = torch_npu.npu_format_cast(packed, target_format)
        actual = int(torch_npu.get_npu_format(converted))
        assert actual in (2, 30), actual
        assert torch.equal(converted.cpu(), cpu), "Format conversion changed values"
        def run():
            base = torch_npu.npu_format_cast(packed, target_format)
            return F.pad(base, pad, mode="reflect")
        for _ in range(2):
            y = run()
        torch.npu.synchronize()
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            for __ in range(5):
                y = run()
            torch.npu.synchronize()
            timings.append((time.perf_counter() - start) * 1000 / 5)
        yc = y.cpu()
        case = {"input_shape": list(shape), "output_shape": list(y.shape), "source_format": initial,
                "converted_format": actual, "output_format": int(torch_npu.get_npu_format(y)),
                "conversion_exact": bool(torch.equal(converted.cpu(), cpu)),
                "output_exact": bool(torch.equal(yc, ref)), "output_dtype": str(y.dtype),
                "median_convert_plus_pad_ms": statistics.median(timings)}
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
        assert case["output_exact"] and y.dtype == dtype
except BaseException as exc:
    result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    raise
finally:
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("Evidence:", out, flush=True)
