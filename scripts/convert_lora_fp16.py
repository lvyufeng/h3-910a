#!/usr/bin/env python3
"""Convert a BF16 LoRA to FP16 for Ascend 910A.

910A cannot compile SoftmaxV2 for bf16 inputs ("Unsupported cast type!
src_dtype: bfloat16, dst_dtype: float32"), and merging a bf16 LoRA drags the
whole sampling graph into bf16.  Converting the adapter to fp16 keeps the graph
in the only precision 910A supports.

    python convert_lora_fp16.py in.safetensors out.safetensors
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def header_dtypes(path: Path) -> collections.Counter:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return collections.Counter(
        value["dtype"] for key, value in header.items() if key != "__metadata__"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args(argv)

    source, target = Path(args.source), Path(args.target)
    before = header_dtypes(source)
    print(f"source dtypes: {dict(before)}")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(source, framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            elif tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            tensors[key] = tensor

    metadata.setdefault("h3_910a_converted_from", str(source.name))
    save_file(tensors, str(target), metadata=metadata)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    print(f"target dtypes: {dict(header_dtypes(target))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())