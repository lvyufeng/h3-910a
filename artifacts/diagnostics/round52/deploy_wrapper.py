#!/usr/bin/env python3
"""Deploy only a hash-verified FA wrapper, preserving the original source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--expected-sha256",required=True)
a=p.parse_args()
root=Path("/tmp/h3_round52_wrapper_validation01")
target=Path("/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py")
assert target.resolve(strict=True)==target,"Unexpected resolved target"
old=target.read_bytes()
assert hashlib.sha256(old).hexdigest()==a.expected_sha256,"Current wrapper differs from inspected source"
new=(root/"candidate_fa910a.py").read_bytes()
ast.parse(new.decode())
backup=Path(str(target)+".pre_masked_tilepad_20261115.bak")
with backup.open("xb") as f:
    f.write(old)
target.write_bytes(new)
assert target.read_bytes()==new
receipt={"target":str(target),"backup":str(backup),"before_sha256":a.expected_sha256,
         "after_sha256":hashlib.sha256(new).hexdigest(),
         "scope":"bounded tile padding; compiled added-key exclusion before probability sum; original caller padding contract unchanged"}
(root/"deployment.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2),flush=True)
