#!/usr/bin/env python3
"""Patch verified canonical and known cache Conv3D padding targets only."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path("/tmp/h3_round51_vae_validation01")
EXPECTED = "b3adee35f27e5d372543242aa86ce5e0138f13b794b6c81f919001e3e2346723"
targets = [
    Path("/home/mseco/models/MiniMax-H3/FL2VA/video_vae/conv.py"),
    Path("/home/mseco/.cache/huggingface/modules/transformers_modules/video_vae/16568b9a798a2948/conv.py"),
    Path("/home/mseco/.cache/huggingface/modules/transformers_modules/video_vae/7d66428023c52353/conv.py"),
]
incoming = (ROOT / "candidate_conv.py").read_bytes()
ast.parse(incoming.decode())
originals = {}
backups = {}
for target in targets:
    assert target.resolve(strict=True) == target, f"Unexpected resolved target: {target}"
    originals[target] = target.read_bytes()
    assert hashlib.sha256(originals[target]).hexdigest() == EXPECTED, f"Source changed: {target}"
    backups[target] = Path(str(target) + ".pre_npu_pad_20261115.bak")
    assert not backups[target].exists(), f"Backup exists; reconcile previous deployment: {backups[target]}"
for target in targets:
    with backups[target].open("xb") as f:
        f.write(originals[target])
for target in targets:
    target.write_bytes(incoming)
    assert target.read_bytes() == incoming
receipt = {"targets": [str(x) for x in targets], "backups": [str(backups[x]) for x in targets],
           "before_sha256": EXPECTED, "after_sha256": hashlib.sha256(incoming).hexdigest(),
           "scope": "local base-format reflection padding and no-op zero-padding removal; dtype unchanged"}
(ROOT / "conv_deployment.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
