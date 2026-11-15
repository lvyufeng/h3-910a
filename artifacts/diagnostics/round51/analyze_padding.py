#!/usr/bin/env python3
"""Summarize exact padding evidence without extrapolating whole-VAE speed."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cases = []
native_failures = []
for p in sorted((root / "pad_probes").glob("*.json")):
    report = json.loads(p.read_text())
    assert "failure" not in report, p
    if report["strategy"] == "native":
        assert "expected_native_failure" in report, p
        native_failures.append({"dtype": report["dtype"], "message": report["expected_native_failure"]})
    else:
        cases.extend(report["cases"])
assert len(cases) == 12 and len(native_failures) == 2
assert all(x["conversion_exact"] and x["output_exact"] for x in cases)
summary = {"conversion_plus_pad_cases": len(cases), "original_failure_dtypes": [x["dtype"] for x in native_failures],
           "exact_padding_outputs": True, "dtype_preserved": True,
           "minimum_small_case_ms": min(x["median_convert_plus_pad_ms"] for x in cases),
           "maximum_small_case_ms": max(x["median_convert_plus_pad_ms"] for x in cases),
           "scope": "small CPU/NPU padding checks; not a whole-VAE speedup or quality claim"}
out = root / "padding_summary.json"
if out.exists():
    raise RuntimeError("Refuse to overwrite existing summary")
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
