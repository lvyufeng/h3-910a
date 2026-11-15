#!/usr/bin/env python3
"""Deploy only the inspected three masked_fill substitutions, with checked backup."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--expected-sha256", required=True)
args = ap.parse_args()
base = Path("/tmp/h3_round49_mask_validation01")
target = Path("/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py")
assert target.resolve(strict=True) == target, "Unexpected resolved deployment target"
old = target.read_bytes()
assert hashlib.sha256(old).hexdigest() == args.expected_sha256, "Current encoder differs from inspected snapshot"
new = (base / "candidate_encoder.py").read_bytes()
ast.parse(new.decode("utf-8"))
backup = Path(str(target) + ".pre_maskedfill_20261115.bak")
with backup.open("xb") as f:
    f.write(old)
target.write_bytes(new)
assert target.read_bytes() == new
record = {"target": str(target), "backup": str(backup), "before_sha256": args.expected_sha256,
          "after_sha256": hashlib.sha256(new).hexdigest()}
(base / "deployment.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
