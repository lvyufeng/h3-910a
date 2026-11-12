#!/usr/bin/env python3
"""Fuse the FlashGen 4-step distill LoRA into the FL2VA transformer on disk.

Why offline: vLLM-Omni's DLO fast path (direct checkpoint mmap, one shared host
copy for all ranks) refuses to run when a distilled LoRA is supplied -- the
loader logs "distilled LoRA requires the ordinary loader" and every rank then
materialises its own 66 GB copy, which OOMs a 754 GB host at 8 ranks.

Fusing first turns the deployment into a plain 4-step checkpoint, so the mmap
path is available again and the 4-step quality/speed is preserved.

Layout produced under --out:
  transformer/           merged shards (same file names as the source)
  <everything else>      symlinks to the source model

Usage:
  python merge_distill_lora.py --model /home/mseco/models/MiniMax-H3 \
      --lora /home/mseco/models/loras/flashgen_4step.safetensors \
      --out /home/mseco/models/MiniMax-H3-4step
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

LORA_SUFFIX_A = ".lora_A.default.weight"
LORA_SUFFIX_B = ".lora_B.default.weight"
PREFIX = "transformer."


def load_lora(path: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
        a_keys = {k[: -len(LORA_SUFFIX_A)]: k for k in keys if k.endswith(LORA_SUFFIX_A)}
        b_keys = {k[: -len(LORA_SUFFIX_B)]: k for k in keys if k.endswith(LORA_SUFFIX_B)}
        pairs = {}
        for module, a_key in a_keys.items():
            b_key = b_keys.get(module)
            if b_key is None:
                raise ValueError(f"LoRA module {module} has A but no B")
            pairs[module] = (handle.get_tensor(a_key), handle.get_tensor(b_key))
    return pairs, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/mseco/models/MiniMax-H3")
    parser.add_argument("--lora", default="/home/mseco/models/loras/flashgen_4step.safetensors")
    parser.add_argument("--out", default="/home/mseco/models/MiniMax-H3-4step")
    parser.add_argument("--variant", default="FL2VA", help="model subdirectory to fuse")
    parser.add_argument("--scale", type=float, default=None,
                        help="override alpha/rank; defaults to metadata alpha/rank")
    args = parser.parse_args(argv)

    model_dir = Path(args.model)
    out_dir = Path(args.out)
    src = model_dir / args.variant / "transformer"
    dst = out_dir / args.variant / "transformer"
    if not src.is_dir():
        raise SystemExit(f"no transformer dir at {src}")

    print(f"loading LoRA {args.lora}", flush=True)
    pairs, metadata = load_lora(Path(args.lora))
    alpha = float(metadata.get("lora_alpha", 0) or 0)
    rank = float(metadata.get("lora_rank", 0) or 0)
    scale = args.scale if args.scale is not None else (alpha / rank if rank else 1.0)
    print(f"lora: {len(pairs)} modules, alpha={alpha} rank={rank} -> scale={scale}", flush=True)
    print(f"schedule={metadata.get('base_schedule')} tasks={metadata.get('tasks')}", flush=True)

    # Precompute deltas once; the LoRA is small (1.2 GB) compared with the base.
    deltas: dict[str, torch.Tensor] = {}
    for module, (a, b) in pairs.items():
        name = module[len(PREFIX):] if module.startswith(PREFIX) else module
        deltas[name + ".weight"] = (b.float() @ a.float()) * scale
    print(f"precomputed {len(deltas)} deltas", flush=True)

    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.safetensors"))
    used: set[str] = set()
    for index, shard in enumerate(shards, 1):
        with safe_open(str(shard), framework="pt") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            metadata_shard = handle.metadata() or {}
        fused = 0
        for key, tensor in list(tensors.items()):
            delta = deltas.get(key)
            if delta is None:
                continue
            if delta.shape != tensor.shape:
                raise SystemExit(f"shape mismatch for {key}: {tuple(delta.shape)} vs {tuple(tensor.shape)}")
            merged = tensor.float() + delta.to(tensor.device)
            tensors[key] = merged.to(tensor.dtype)
            used.add(key)
            fused += 1
        target = dst / shard.name
        save_file(tensors, str(target), metadata=metadata_shard)
        print(f"[{index}/{len(shards)}] {shard.name}: fused {fused} tensors -> {target.stat().st_size/1e9:.2f} GB", flush=True)
        del tensors

    missing = sorted(set(deltas) - used)
    if missing:
        print(f"WARNING: {len(missing)} LoRA targets never matched, e.g. {missing[:3]}", flush=True)

    # Everything except the fused transformer is shared with the source model.
    for entry in sorted((model_dir / args.variant).iterdir()):
        if entry.name == "transformer":
            continue
        link = out_dir / args.variant / entry.name
        if link.exists() or link.is_symlink():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(entry, link)
    top_index = model_dir / "model_index.json"
    if top_index.exists():
        target = out_dir / "model_index.json"
        if not target.exists():
            os.symlink(top_index, target)

    marker = out_dir / "fused_lora.json"
    marker.write_text(json.dumps({
        "source_model": str(model_dir),
        "lora": str(args.lora),
        "lora_metadata": metadata,
        "scale": scale,
        "fused_tensors": len(used),
        "steps": len(str(metadata.get("base_schedule", "")).split(",")) - 1,
    }, indent=2), encoding="utf-8")
    print(f"done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())