#!/usr/bin/env python3
"""Register vLLM-Omni's DLO copy/comm streams with the Ascend allocator.

On Ascend a freshly created ``torch.npu.Stream`` has no allocator bound to it,
so the first HCCL collective issued on it dies with

    aclrtAllocatorGetByStream failed. Parameter stream is invalid.
    Reason: The stream is not registered with any allocator.

which surfaces as ``HcclAllreduce`` failing mid-generation. Running a small
allocation inside the stream context is enough to register it; after that the
collectives work.

Idempotent: re-running is a no-op.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/offloader/distributed_layerwise_backend.py")

OLD = """        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()
"""

NEW = """        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        # Ascend (910A) has no allocator bound to a fresh stream, so the first
        # HCCL collective on it fails with aclrtAllocatorGetByStream. Touching
        # each stream once registers it. Cheap, and a no-op on CUDA where the
        # caching allocator already covers every stream.
        self._register_stream_with_allocator(self.copy_stream)
        self._register_stream_with_allocator(self.comm_stream)
"""

HELPER = '''
    @staticmethod
    def _register_stream_with_allocator(stream) -> None:
        """Warm a stream so the device allocator binds to it (Ascend 910A)."""
        try:
            with current_omni_platform.stream(stream):
                probe = torch.empty(8, dtype=torch.float16,
                                    device=current_omni_platform.current_device())
                del probe
            stream.synchronize()
        except Exception:  # noqa: BLE001 - never block start-up on a warm-up
            logger.debug("Could not warm stream %s", stream, exc_info=True)
'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_register_stream_with_allocator" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found; upstream changed")

    text = text.replace(OLD, NEW, 1)

    # Insert the helper right after the class body starts (after the class docstring).
    class_marker = "class "
    idx = text.index("class DistributedLayerwise")
    brace = text.index("\n", text.index(":", idx))
    text = text[: brace + 1] + HELPER + text[brace + 1 :]

    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())