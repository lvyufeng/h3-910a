"""FastH3 FP8 -> FP16 dequantizer (CPU only).

The FastH3-Omni-FP8 export stores each quantized linear as
``<layer>.weight`` in float8_e4m3fn plus ``<layer>.weight_scale`` in float32
with shape [N, 1] (one scale per output channel). Everything else is BF16/F32.
Dequantization is therefore exactly ``w_fp16 = (w_fp8.float() * scale).half()``.

Usage:
  python dequant_fasth3.py --src DIR --dst DIR [--shards 1,2] [--validate]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dtype_of(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shards", default="")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    index_path = src / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    # weight name lookup for every scale tensor
    scale_to_weight: dict[str, str] = {}
    for name in weight_map:
        if name.endswith(".weight_scale"):
            scale_to_weight[name] = name[: -len(".weight_scale")] + ".weight"
    missing = [n for n in weight_map if n.endswith(".weight_scale") and scale_to_weight[n] not in weight_map]
    if missing:
        raise SystemExit(f"scale tensors without weight: {missing[:3]}")

    shards = sorted({v for v in weight_map.values()})
    only = {s.strip() for s in args.shards.split(",") if s.strip()}
    if only:
        shards = [s for s in shards if any(o in s for o in only)]

    out_index = {"metadata": {}, "weight_map": {}}
    for shard in shards:
        t0 = time.time()
        spath = src / shard
        with safe_open(str(spath), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            out: dict[str, torch.Tensor] = {}
            stats = {"fp8": 0, "bf16": 0, "f32": 0, "f16": 0, "other": 0, "scale": 0}
            for name in keys:
                if name.endswith(".weight_scale"):
                    stats["scale"] += 1
                    continue
                t = f.get_tensor(name)
                dt = dtype_of(t)
                if dt == "float8_e4m3fn":
                    scale_name = name[: -len(".weight")] + ".weight_scale"
                    if scale_name not in keys:
                        raise SystemExit(f"{name}: fp8 weight without scale in same shard")
                    scale = f.get_tensor(scale_name).float().reshape(-1, 1)
                    if scale.shape[0] != t.shape[0]:
                        raise SystemExit(f"{name}: scale {tuple(scale.shape)} vs weight {tuple(t.shape)}")
                    deq = (t.float() * scale).half()
                    if args.validate:
                        back = (deq.float() / scale).to(torch.float8_e4m3fn)
                        nz = back.float() != t.float()
                        bad = int(nz.sum())
                        if bad:
                            print(f"  [warn] {name}: {bad} roundtrip mismatches", flush=True)
                    out[name] = deq
                    stats["fp8"] += 1
                    del t, scale, deq
                elif dt == "bfloat16":
                    out[name] = t.half()
                    stats["bf16"] += 1
                elif dt == "float32":
                    out[name] = t.half()
                    stats["f32"] += 1
                elif dt == "float16":
                    out[name] = t
                    stats["f16"] += 1
                else:
                    stats["other"] += 1
                    out[name] = t
            outname = f"fp16-{shard}"
            save_file(out, str(dst / outname), metadata={"format": "pt", "source": str(spath)})
            for name, tensor in out.items():
                out_index["weight_map"][name] = outname
            sizes = {k: v for k, v in stats.items() if v}
            print(f"{shard}: {len(out)} tensors -> {outname} in {time.time()-t0:.0f}s {sizes}", flush=True)
            del out
    (dst / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(out_index, indent=2) + "\n")
    cfg = src / "config.json"
    if cfg.exists():
        (dst / "config.json").write_text(cfg.read_text())
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
