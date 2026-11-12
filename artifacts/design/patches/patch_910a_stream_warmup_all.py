#!/usr/bin/env python3
"""Warm every vLLM-Omni offloader stream so Ascend binds an allocator to it.

A fresh ``torch.npu.Stream`` has no allocator registered, and the first op that
runs on it fails with

    aclrtAllocatorGetByStream failed. Parameter stream is invalid.
    Reason: The stream is not registered with any allocator.

That failure then cascades: later ops report unrelated names (SwiGlu, RmsNorm,
HcclAllreduce) because the device context is already broken. The earlier patch
only covered the DLO hook; the backend, the module-residency helper and the
plain layerwise backend create their own streams too.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

HELPER = '''

def _warm_910a_stream(stream, device=None):
    """Touch a device stream once so the Ascend allocator binds to it."""
    try:
        import torch
        torch.npu.current_stream().wait_stream(stream)
        with torch.npu.stream(stream):
            probe = torch.empty(8, dtype=torch.float16, device=device or torch.npu.current_device())
            del probe
        stream.synchronize()
        return stream
    except Exception:  # noqa: BLE001 - warm-up must never break start-up
        return stream
'''

SITES = [
    (
        "vllm_omni/diffusion/offloader/distributed_layerwise_backend.py",
        "        self.copy_stream = current_omni_platform.Stream()\n"
        "        self.comm_stream = current_omni_platform.Stream()\n",
        "        self.copy_stream = _warm_910a_stream(current_omni_platform.Stream(), self.device)\n"
        "        self.comm_stream = _warm_910a_stream(current_omni_platform.Stream(), self.device)\n",
    ),
    (
        "vllm_omni/diffusion/offloader/module_residency.py",
        "        self.copy_stream = copy_stream if copy_stream is not None else current_omni_platform.Stream()\n",
        "        self.copy_stream = (\n"
        "            copy_stream if copy_stream is not None\n"
        "            else _warm_910a_stream(current_omni_platform.Stream())\n"
        "        )\n",
    ),
    (
        "vllm_omni/diffusion/offloader/layerwise_backend.py",
        "        self.copy_stream = current_omni_platform.Stream()\n",
        "        self.copy_stream = _warm_910a_stream(current_omni_platform.Stream())\n",
    ),
]


def _inject_helper(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "_warm_910a_stream" in text:
        return
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            insert_at = index + 1
    lines.insert(insert_at, HELPER)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    patched = 0
    for rel, old, new in SITES:
        path = Path(rel)
        if not path.exists():
            print(f"skip missing {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if "_warm_910a_stream" in text:
            print(f"already {rel}")
            continue
        if old not in text:
            print(f"anchor missing in {rel}")
            continue
        _inject_helper(path)
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        patched += 1
        print(f"patched {rel}")
    print(f"total {patched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())