#!/usr/bin/env python3
"""Remove the fp16 ``nan_to_num`` call from the MiniMax-H3 video VAE.

910A ships no ``NanToNum`` kernel for fp16 -- fp32 and bf16 have one:

    No supported Ops kernel and engine are found for [NanToNum542],
    optype [NanToNum]

The video VAE decoder calls ``.nan_to_num(0.0)`` on its attention output, so the
request fails during VAE decode, after the denoise loop has already succeeded.

Why this patches the model directory and not the HF cache
--------------------------------------------------------
``transformers`` copies a dynamic module out of the model directory into
``~/.cache/huggingface/modules/transformers_modules/<name>/<hash>/`` on every
load, keyed by content hash. Editing the cached copy is silently discarded on
the next start, which makes the bug look unfixable. The edit has to land in the
model's own source tree.

Note that registering an ``aten::nan_to_num`` dispatcher fallback does *not*
work here: for fp16 the Ascend backend kernel is still selected, and a
CompositeExplicitAutograd implementation is bypassed.

Idempotent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD = "    return out.transpose(1, 2).nan_to_num(0.0)"

NEW = '''    out = out.transpose(1, 2)
    # 910A has no fp16 NanToNum kernel; scrub non-finite values with primitives
    # that do exist on this SoC.
    return torch.where(torch.isfinite(out), out, torch.zeros((), dtype=out.dtype, device=out.device))'''

MARKER = "torch.isfinite(out)"


def patch_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already patched"
    if OLD not in text:
        raise SystemExit(f"{path}: expected line not found; is this the H3 video VAE?")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        help="H3 model directory, e.g. /home/mseco/models/MiniMax-H3-4step",
    )
    args = parser.parse_args()

    root = Path(args.model_dir)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    targets = sorted(root.rglob("flash.py"))
    vae_targets = [p for p in targets if "video_vae" in p.parts]
    if not vae_targets:
        raise SystemExit(f"no video_vae/flash.py under {root}")

    for path in vae_targets:
        print(f"{path}: {patch_file(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())