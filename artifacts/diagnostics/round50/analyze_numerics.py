#!/usr/bin/env python3
"""Summarize raw norm and vision probes without inferring model-level losslessness."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
cases = json.loads((root / "layernorm_probe02_result.json").read_text())["cases"]
vision = json.loads((root / "vision_block_probe04.json").read_text())
tests = {}
for stem in ("norm", "mask", "attn"):
    text = (root / f"{stem}_tests_final.log").read_text()
    count = int(re.search(r"Ran (\d+) tests", text).group(1))
    assert text.rstrip().endswith("OK")
    tests[stem] = count
result = {"normalization_cases": len(cases), "bf16_input_output_preserved": all(x["dtype"] == "torch.bfloat16" for x in cases),
          "max_relative_rms_vs_float64": max(x["stable_npu_fp32"]["relative_rms"] for x in cases),
          "max_absolute_error_vs_float64": max(x["stable_npu_fp32"]["max_abs_error_vs_float64"] for x in cases),
          "constant_and_zero_exact_bias": all(x["stable_equals_bias"] for x in cases if x["mode"] in ("constant", "zero")),
          "vision_block_relative_rms_vs_cpu_bf16": vision["relative_rms_vs_cpu_bf16"],
          "vision_block_finite": vision["finite"], "passing_regressions": tests,
          "scope": "operator/component numerical checks, not whole-model quality or official lossless validation"}
out = root / "numerical_summary.json"
if out.exists():
    raise RuntimeError("Refuse to overwrite prior summary")
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
