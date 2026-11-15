#!/usr/bin/env python3
"""Summarize final masked wrapper measurements without an end-to-end claim."""
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
r=json.loads((root/"validation_result05.json").read_text())
assert "failure" not in r
old=statistics.mean(x["median_ms"] for x in r["timing_runs"] if x["variant"]=="old_small_tiles")
new=statistics.mean(x["median_ms"] for x in r["timing_runs"] if x["variant"]=="new_internal_masked_padding")
assert len(r["timing_runs"])==4
assert r["cpu_pad_buffer_reuse_and_tail_zero"]
assert all(x["zero_tail"] and x["finite"] for x in r["length_change_cases"])
s={"old_wrapper_ms":old,"new_wrapper_ms":new,"speedup":old/new,"latency_reduction_percent":(1-new/old)*100,
   "max_pair_relative_rms":max(x["pair_relative_rms"] for x in r["numerical_cases"]),
   "max_new_sampled_fp32_reference_relative_rms":max(x["reference"]["new_internal_masked_padding"]["sampled_relative_rms"] for x in r["numerical_cases"]),
   "cpu_pad_reuse_validated":True,"npu_key_length_changes_validated":r["length_change_cases"],
   "negative_score_case":r["negative_score_case"],"finite_minimum_score_case":r["finite_minimum_score_case"],
   "masked_kernel":r["masked_kernel"],
   "scope":"Isolated full wrapper including copies/clearing/trimming; sampled reference checks are not whole-model lossless validation."}
p=root/"wrapper_summary.json"
if p.exists(): raise RuntimeError("Refuse to overwrite an existing summary")
p.write_text(json.dumps(s,indent=2)+"\n")
print(json.dumps(s,indent=2))
