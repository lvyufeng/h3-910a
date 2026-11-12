#!/usr/bin/env python3
"""Create a patched ComfyUI H3 workflow without modifying the source file.

Examples:
  python3 scripts/make_instance_workflow.py \
    --source workflows/fl2va_4step_极速档.json \
    --output /tmp/h3-8801-864x480.json \
    --width 864 --height 480 --frames 124 --steps 4 --npu-count 4

The tool accepts both ComfyUI UI-format and /prompt API-format workflows.
It locates nodes by class/type and follows graph links, so it does not rely on
the node IDs in the bundled upstream workflow. Model paths come from
H3_MODEL_* environment variables when command-line values are omitted:

  H3_MODEL_DIFFUSION    default model filename
  H3_MODEL_CLIP         text encoder filename
  H3_MODEL_VIDEO_VAE    video VAE filename
  H3_MODEL_AUDIO_VAE    audio VAE filename
  H3_MODEL_LORA         few-step LoRA filename
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from h3_workflow import (
    Overrides,
    WorkflowError,
    apply_overrides,
    dump_workflow,
    load_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an instance-specific MiniMax H3 ComfyUI workflow."
    )
    parser.add_argument("--source", required=True, help="UI or API workflow JSON")
    parser.add_argument("--output", required=True, help="output workflow JSON")
    parser.add_argument("--model", default=os.getenv("H3_MODEL_DIFFUSION"))
    parser.add_argument("--clip", default=os.getenv("H3_MODEL_CLIP"))
    parser.add_argument("--video-vae", default=os.getenv("H3_MODEL_VIDEO_VAE"))
    parser.add_argument("--audio-vae", default=os.getenv("H3_MODEL_AUDIO_VAE"))
    parser.add_argument("--lora", default=os.getenv("H3_MODEL_LORA"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--npu-count", type=int, choices=(1, 2, 4))
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--first-image",
        help="LoadImage filename, or 'none' to disconnect the first frame",
    )
    parser.add_argument(
        "--last-image",
        help="LoadImage filename, or 'none' to disconnect the last frame",
    )
    parser.add_argument("--filename-prefix")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a machine-readable summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if source == output:
        raise WorkflowError("--output must differ from --source")

    overrides = Overrides(
        model=args.model,
        clip=args.clip,
        video_vae=args.video_vae,
        audio_vae=args.audio_vae,
        lora=args.lora,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        npu_count=args.npu_count,
        prompt=args.prompt,
        seed=args.seed,
        first_image=args.first_image,
        last_image=args.last_image,
        filename_prefix=args.filename_prefix,
    )
    workflow = load_workflow(source)
    patched, summary = apply_overrides(workflow, overrides)
    dump_workflow(patched, output)
    summary["source"] = str(source)
    summary["output"] = str(output)
    if args.summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            f"wrote {output} ({summary['format']}, "
            f"video node {summary['video_node']})"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
