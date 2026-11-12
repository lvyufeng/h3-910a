#!/usr/bin/env python3
"""Replace vLLM-Omni's object broadcasts with a tensor-backed implementation.

``torch.distributed.broadcast_object_list`` allocates its staging tensor from a
size tensor and calls ``.item()`` on it. On this NPU stack that path raises

    SymIntArrayRef expected to contain only concrete integers

The module already ships ``_broadcast_tensor``, which works here, so serialise to
bytes, broadcast as uint8 through that helper, and deserialise on the other side.
Same semantics, no shape machinery involved.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py")

HELPER = '''

def _broadcast_object_bytes(obj, group, rank, world_size, device):
    """Broadcast an arbitrary Python object as a uint8 tensor (910A-safe)."""
    import pickle

    if world_size == 1:
        return obj
    payload = pickle.dumps(obj)
    size = torch.tensor([len(payload)], dtype=torch.long, device=device)
    size = _broadcast_tensor(size, dtype=torch.long, device=device)
    length = int(size.item())
    if rank == 0:
        data = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)
    else:
        data = torch.empty(length, dtype=torch.uint8, device=device)
    data = _broadcast_tensor(data, dtype=torch.uint8, device=device)
    return obj if rank == 0 else pickle.loads(bytes(data.cpu().tolist()))
'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_broadcast_object_bytes" in text:
        print("already patched")
        return 0

    # Insert the helper right before the first use site's enclosing class.
    anchor = "def _broadcast_rank0_exception("
    if anchor not in text:
        raise SystemExit("anchor function not found")
    text = text.replace(anchor, HELPER.strip("\n") + "\n\n\n" + anchor, 1)

    replacements = [
        (
            "    dist.broadcast_object_list(payload, src=0, group=group)",
            "    payload = [_broadcast_object_bytes(\n"
            "        payload[0], group, rank, world_size,\n"
            "        torch.device(\"npu\", torch.npu.current_device()),\n"
            "    )]",
        ),
        (
            "        dist.broadcast_object_list(header, src=0, group=group)",
            "        header = [_broadcast_object_bytes(header[0], group, rank, world_size, self.device)]",
        ),
        (
            "            dist.broadcast_object_list(reuse_text, src=0, group=group)",
            "            reuse_text = [_broadcast_object_bytes(reuse_text[0], group, rank, world_size, self.device)]",
        ),
    ]
    changed = 0
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed += 1
        elif old.splitlines()[0].strip() in text:
            # Already in another form; leave it.
            pass

    if not changed:
        raise SystemExit("no broadcast sites matched")

    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {changed} site(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())