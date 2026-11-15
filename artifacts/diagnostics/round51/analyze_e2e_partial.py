#!/usr/bin/env python3
"""Keep incomplete real-workload measurements separate from steady performance."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
r = json.loads((root / "e2e01/result.json").read_text())
assert r["failure"]["type"] == "TimeoutError"
text = (root / "e2e01/serve.log").read_text(errors="replace")
dsteps = [{"index": int(i), "prep_s": float(a), "model_s": float(b), "update_s": float(c)}
          for i, a, b, c in re.findall(r"\[H3PERF\] dstep (\d+) prep=([0-9.]+)s model=([0-9.]+)s update=([0-9.]+)s", text)]
assert len(dsteps) == 8 and all(x["index"] == 1 for x in dsteps)
summary = {"completed_video": False, "stopped_by": "manager's 900s request deadline", "steady_step_measured": False,
           "encode_prepare_reported_s": sorted(set(r["stage_times"]["encode"])),
           "completed_dstep_reports": dsteps, "first_iteration_progressbar_s": 351.31,
           "actual_attention": {"B": 1, "H": 7, "N": 41792, "D": 128, "Br": 64, "Bc": 64,
                                "block_dim": 30, "ranks_confirmed": len(r["fa_core_rows"])},
           "qualification": "Cold first-step and partial second-step evidence; do not infer a full steady step or attribute residual time by subtraction."}
p = root / "e2e_partial_summary.json"
if p.exists():
    raise RuntimeError("Refuse to overwrite an existing summary")
p.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
