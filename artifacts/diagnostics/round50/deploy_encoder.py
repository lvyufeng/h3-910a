#!/usr/bin/env python3
"""Deploy only on a hash-matched encoder, preserving a unique rollback copy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--expected-sha256", required=True)
a = ap.parse_args()
root = Path("/tmp/h3_round50_norm_validation01")
target = Path("/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py")
assert target.resolve(strict=True) == target, "Unexpected resolved target"
original = target.read_bytes()
assert hashlib.sha256(original).hexdigest() == a.expected_sha256, "Current source differs from inspected baseline"
incoming = (root / "candidate_encoder.py").read_bytes()
ast.parse(incoming.decode())
backup = Path(str(target) + ".pre_stable_bf16norm_20261115.bak")
with backup.open("xb") as f:
    f.write(original)
target.write_bytes(incoming)
assert target.read_bytes() == incoming
receipt = {"target": str(target), "backup": str(backup), "before_sha256": a.expected_sha256,
           "after_sha256": hashlib.sha256(incoming).hexdigest(), "scope": "vision LayerNorm and unmasked attention; BF16 I/O and weights retained"}
(root / "deployment.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
