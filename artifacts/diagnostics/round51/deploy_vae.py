#!/usr/bin/env python3
"""Apply the tested local-format padding fix to verified canonical/cached sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--expected-sha256", required=True)
a = ap.parse_args()
root = Path("/tmp/h3_round51_vae_validation01")
targets = [Path("/home/mseco/models/MiniMax-H3/FL2VA/video_vae/vae_cnn.py"),
           Path("/home/mseco/.cache/huggingface/modules/transformers_modules/video_vae/16568b9a798a2948/vae_cnn.py")]
incoming = (root / "candidate_vae_cnn.py").read_bytes()
ast.parse(incoming.decode())
originals = {}
backups = {}
for target in targets:
    assert target.resolve(strict=True) == target, f"Unexpected resolved target: {target}"
    originals[target] = target.read_bytes()
    assert hashlib.sha256(originals[target]).hexdigest() == a.expected_sha256, f"Source changed: {target}"
    backups[target] = Path(str(target) + ".pre_npu_pad_20261115.bak")
    assert not backups[target].exists(), f"Backup exists; reconcile previous deployment: {backups[target]}"
for target in targets:
    with backups[target].open("xb") as f:
        f.write(originals[target])
for target in targets:
    target.write_bytes(incoming)
    assert target.read_bytes() == incoming
receipt = {"targets": [str(x) for x in targets], "backups": [str(backups[x]) for x in targets],
           "before_sha256": a.expected_sha256, "after_sha256": hashlib.sha256(incoming).hexdigest(),
           "scope": "Downsample3D reflection padding only; dtype and global format policy unchanged"}
(root / "deployment.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
