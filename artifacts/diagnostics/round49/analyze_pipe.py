#!/usr/bin/env python3
"""Summarize hardware counters, preserving their overlapped-pipeline semantics."""
import csv
import json
from pathlib import Path
import statistics
import sys

root = Path(__file__).resolve().parent
with (root / "pipe_profile/kernel_details.csv").open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 3, "Expected exactly 3 profiled raw-kernel launches"
assert all(int(x["Block Num"]) == 30 for x in rows)
result = {"calls": len(rows), "block_dim": 30, "kernel": rows[0]["Name"],
          "semantic_note": "Counters describe overlapped pipeline activity; ratios are not additive stage percentages.",
          "duration_ms": {"min": min(float(x["Duration(us)"]) for x in rows) / 1000,
                          "max": max(float(x["Duration(us)"]) for x in rows) / 1000,
                          "mean": statistics.mean(float(x["Duration(us)"]) for x in rows) / 1000},
          "pipes": {}}
for name, prefix in (("vector", "vec"), ("cube_mac", "mac"), ("scalar", "scalar"),
                     ("MTE1", "mte1"), ("MTE2", "mte2"), ("MTE3", "mte3")):
    result["pipes"][name] = {"average_time_ms": statistics.mean(float(x[prefix + "_time(us)"]) for x in rows) / 1000,
                            "average_ratio": statistics.mean(float(x[prefix + "_ratio"]) for x in rows)}
unroll = json.loads((root / "broadcast_unroll_result.json").read_text())
base_ms = statistics.mean(x["median_ms"] for x in unroll["runs"] if x["variant"] == "baseline")
new_ms = statistics.mean(x["median_ms"] for x in unroll["runs"] if x["variant"] == "unroll")
result["static_broadcast_unroll_ab"] = {"baseline_ms": base_ms, "candidate_ms": new_ms,
                                        "latency_reduction_percent": (1 - new_ms / base_ms) * 100,
                                        "bit_exact": unroll["bit_exact"], "deployed": False}
print(json.dumps(result, indent=2))
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    if path.exists():
        raise RuntimeError("Refuse to overwrite existing summary")
    path.write_text(json.dumps(result, indent=2) + "\n")
