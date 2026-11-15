#!/usr/bin/env python3
"""Deploy only if the current wrapper matches the inspected snapshot exactly."""
import argparse
import ast
import hashlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--expected-sha256", required=True)
a = ap.parse_args()
target = Path("/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py")
resolved = target.resolve(strict=True)
if resolved != target:
    raise RuntimeError(f"Unexpected deployment target: {resolved}")
original = target.read_bytes()
if hashlib.sha256(original).hexdigest() != a.expected_sha256:
    raise RuntimeError("Current remote file differs from inspected snapshot; do not overwrite")
incoming = Path("/tmp/h3_candidate_fa910a.py").read_bytes()
ast.parse(incoming.decode())
backup = Path(str(target) + ".pre_corebudget_20261115.bak")
with backup.open("xb") as f:
    f.write(original)
target.write_bytes(incoming)
assert target.read_bytes() == incoming
print("Deployed checked core-budget patch; original preserved at", backup, flush=True)
print("sha256:", hashlib.sha256(incoming).hexdigest(), flush=True)
