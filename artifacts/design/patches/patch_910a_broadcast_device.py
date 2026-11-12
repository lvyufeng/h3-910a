#!/usr/bin/env python3
"""Give vLLM-Omni's object broadcasts an explicit device for HCCL.

``torch.distributed.broadcast_object_list`` without ``device=`` builds a CPU
ByteTensor and tries to push it through the process group. On HCCL that is
invalid, and the failure surfaces far away as

    SymIntArrayRef expected to contain only concrete integers

which points at the shape machinery rather than at the real culprit. Passing the
NPU device makes the collective use an on-device staging buffer.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py")

# (old, new) pairs, applied in order; each must exist unless already patched.
REPLACEMENTS = [
    (
        "    dist.broadcast_object_list(payload, src=0, group=group)",
        "    dist.broadcast_object_list(\n"
        "        payload, src=0, group=group,\n"
        "        device=(\n"
        "            torch.device(\"npu\", torch.npu.current_device())\n"
        "            if hasattr(torch, \"npu\") and torch.npu.is_available()\n"
        "            else None\n"
        "        ),\n"
        "    )",
    ),
    (
        "        dist.broadcast_object_list(header, src=0, group=group)",
        "        dist.broadcast_object_list(header, src=0, group=group, device=self.device)",
    ),
    (
        "            dist.broadcast_object_list(reuse_text, src=0, group=group)",
        "            dist.broadcast_object_list(reuse_text, src=0, group=group, device=self.device)",
    ),
]

MARKER = "device=self.device)"


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if text.count(MARKER) >= 2 and "torch.npu.current_device()" in text:
        print("already patched")
        return 0

    changed = 0
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
            changed += 1

    if not changed:
        raise SystemExit("no anchors matched; upstream changed")

    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {changed} broadcast site(s) in {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())