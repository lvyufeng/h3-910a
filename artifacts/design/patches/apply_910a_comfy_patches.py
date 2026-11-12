#!/usr/bin/env python3
"""Apply the 910A ComfyUI patches that the community deployment guide lists.

Source: the "MiniMax-H3 昇腾 910A 完整部署使用指南" in
``ComfyUI-Ascend910A-Adaptation``.  Four source patches are needed before
ComfyUI can run MiniMax-H3 on 910A:

1. ``comfy/model_management.py`` — ``is_oom`` must recognise NPU OOM, otherwise
   the attention retry/cache-clearing path never triggers.
2. ``comfy/ldm/modules/attention.py`` — ``attention_split`` rounds the chunk
   size up; the original floor division makes a long video sequence allocate a
   ~26 GB attention matrix in one go.
3. ``comfy/sd.py`` — a quantised UNet must respect ``unet_dtype`` instead of
   forcing a manual cast to bf16 (bf16 softmax does not compile on 910A).
4. ``main.py`` — disable torch_npu internal format (reflection_pad3d) and
   replace ``torch.nan_to_num`` (NanToNum is unsupported).

Patch (4) is installed as ``sitecustomize.py`` on the launcher's PYTHONPATH
instead of editing ``main.py``: it does not depend on ComfyUI's version and
also covers worker subprocesses.

Usage:
    python apply_910a_comfy_patches.py /path/to/ComfyUI [--revert]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

OOM_ANCHOR = """    if isinstance(e, ACCELERATOR_ERROR) and (getattr(e, 'error_code', None) == 2 or "out of memory" in str(e).lower()):
        discard_cuda_async_error()
        return True
    return False
"""

OOM_PATCHED = """    if isinstance(e, ACCELERATOR_ERROR) and (getattr(e, 'error_code', None) == 2 or "out of memory" in str(e).lower()):
        discard_cuda_async_error()
        return True
    if is_ascend_npu() and isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        discard_cuda_async_error()
        return True
    return False
"""

SPLIT_ANCHOR = "slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]"
SPLIT_PATCHED = "slice_size = math.ceil(q.shape[1] / steps)"

SD_ANCHOR = """    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
"""
SD_PATCHED = """    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
"""

#: 910A has no usable bf16: SoftmaxV2 fails to compile with
#: "Unsupported cast type! src_dtype: bfloat16, dst_dtype: float32".  ComfyUI's
#: unet_dtype() still picks bf16 on NPU, which drags the whole sampling graph
#: into bf16, so pin the inference dtype to fp16 unless the user forced fp32.
FP16_ANCHOR = """    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)"""
FP16_PATCHED = """    if model_management.is_ascend_npu() and unet_dtype != torch.float32:
        unet_dtype = torch.float16
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)"""


def patch_file(path: Path, anchor: str, replacement: str, label: str, *,
               count: int = 1, optional: bool = True) -> bool:
    if not path.exists():
        print(f"SKIP {label}: {path} missing")
        return False
    text = path.read_text(encoding="utf-8")
    found = text.count(anchor)
    if found == 0:
        marker = replacement.splitlines()[0]
        if marker in text:
            print(f"OK   {label}: already patched")
            return True
        if optional:
            print(f"SKIP {label}: anchor not found (upstream text changed?)")
            return False
        print(f"FAIL {label}: anchor not found", file=sys.stderr)
        return False
    backup = path.with_suffix(path.suffix + ".before-910a")
    if not backup.exists():
        shutil.copy2(path, backup)
    text = text.replace(anchor, replacement, count if count else -1)
    path.write_text(text, encoding="utf-8")
    print(f"OK   {label}: patched {found} site(s)")
    return True


def revert(path: Path, label: str) -> None:
    backup = path.with_suffix(path.suffix + ".before-910a")
    if backup.exists():
        shutil.copy2(backup, path)
        print(f"REVERTED {label}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", help="ComfyUI checkout")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.comfyui)
    targets = [
        (root / "comfy/model_management.py", OOM_ANCHOR, OOM_PATCHED, "model_management.is_oom"),
        (root / "comfy/ldm/modules/attention.py", SPLIT_ANCHOR, SPLIT_PATCHED, "attention_split ceil"),
        (root / "comfy/sd.py", SD_ANCHOR, SD_PATCHED, "sd.py manual_cast dtype"),
    ]
    if args.revert:
        for path, _anchor, _new, label in targets:
            revert(path, label)
        return 0

    ok = True
    for path, anchor, replacement, label in targets:
        ok &= patch_file(path, anchor, replacement, label)
    # Both load_state_dict paths need the fp16 pin; count=0 means "every site".
    ok &= patch_file(root / "comfy/sd.py", FP16_ANCHOR, FP16_PATCHED,
                     "sd.py ascend fp16 dtype", count=0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())