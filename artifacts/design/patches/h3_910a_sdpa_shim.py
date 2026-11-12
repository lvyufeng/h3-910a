# SPDX-License-Identifier: Apache-2.0
"""910A runtime shims for vLLM-Omni.

``torch.nn.functional.scaled_dot_product_attention`` has to be replaced on this
hardware: the 910A generation has no usable attention kernel. SDPA raises
``ERR01007 can not cast format when output is input``, while
``npu_fusion_attention`` and ``PromptFlashAttention`` both fail to launch.
Several H3 modules call the functional entry point directly (the NPU platform
even swaps ``encoder._scaled_dot_product_attention`` for its own copy), so
patching the diffusion backend registry alone is not enough.

Imported for side effects from ``vllm_omni/__init__.py``.
"""

from __future__ import annotations

import os

import torch

_ORIGINAL = torch.nn.functional.scaled_dot_product_attention
_CHUNK_DEFAULT = 2048
_APPLIED = False


def _chunk() -> int:
    try:
        return max(1, int(os.environ.get("VLLM_OMNI_910A_ATTN_CHUNK", _CHUNK_DEFAULT)))
    except (TypeError, ValueError):
        return _CHUNK_DEFAULT


def _chunk_rows() -> int:
    """Query rows per attention tile.

    Sized so one ``[heads, chunk, keys]`` score tile stays well under a gigabyte
    in fp16 and only three kernels are launched per tile (QK, softmax, PV).
    A fully tiled implementation is slower here: every extra elementwise op is a
    separate kernel launch, and with hundreds of tiles that overhead dominates.
    """
    try:
        return max(16, int(os.environ.get("VLLM_OMNI_910A_ATTN_CHUNK", 128)))
    except (TypeError, ValueError):
        return 128


def _matmul_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False, **kwargs):
    """Chunked matmul/softmax attention with SDPA's tensor contract.

    The 910A generation has no usable fused attention kernel, so this keeps the
    per-tile kernel count minimal: one QK matmul, one softmax, one PV matmul.
    Scores are upcast only for the softmax, because Ascend's SoftmaxV2 rejects
    bfloat16 outright ("Unsupported cast type! src_dtype: bfloat16").
    """
    if enable_gqa and query.shape[1] != key.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    head_dim = query.shape[-1]
    scale = (1.0 / (head_dim ** 0.5)) if scale is None else scale
    queries = query.shape[-2]
    keys = key.shape[-2]
    key_t = key.transpose(-1, -2).contiguous()

    step = min(_chunk_rows(), queries)
    chunks = []
    for start in range(0, queries, step):
        stop = min(start + step, queries)
        scores = torch.matmul(query[..., start:stop, :], key_t) * scale

        if is_causal:
            rows = torch.arange(start, stop, device=query.device).view(-1, 1)
            cols = torch.arange(keys, device=query.device).view(1, -1)
            _NEG = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(cols > rows, _NEG)
        if attn_mask is not None:
            mask = attn_mask if attn_mask.dtype == torch.bool else attn_mask.bool()
            if mask.dim() >= 3 and mask.shape[-2] != 1:
                mask = mask[..., start:stop, :]
            if mask.device != scores.device:
                mask = mask.to(scores.device)
            _NEGM = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, _NEGM)

        # fp16 dot products over 128 dims overflow easily (max 65504), and an
        # inf reaching softmax makes Ascend emit its NanToNum fallback, which
        # has no kernel on 910A. Softmax subtracts the row max internally, so
        # clamping to a magnitude no fp16 score reaches is lossless here.
        scores = scores.clamp(-50000.0, 50000.0)
        weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        chunks.append(torch.matmul(weights, value))

    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-2)


def _shim(query, key, value, *args, **kwargs):
    try:
        on_npu = query.device.type == "npu"
    except AttributeError:
        return _ORIGINAL(query, key, value, *args, **kwargs)
    if not on_npu:
        return _ORIGINAL(query, key, value, *args, **kwargs)

    names = ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
    bound = dict(zip(names, args))
    bound.update({k: v for k, v in kwargs.items() if k in names})
    bound.setdefault("attn_mask", None)
    bound.setdefault("dropout_p", 0.0)
    bound.setdefault("is_causal", False)
    bound.setdefault("scale", None)
    bound.setdefault("enable_gqa", False)
    return _matmul_sdpa(query, key, value, **bound)


def apply() -> bool:
    """Install the attention shim once; True when this call installed it."""
    global _APPLIED
    if _APPLIED:
        return False
    if os.environ.get("VLLM_OMNI_910A_SDPA_SHIM", "1") == "0":
        return False
    torch.nn.functional.scaled_dot_product_attention = _shim
    _APPLIED = True
    return True


_STREAM_FACTORY_PATCHED = False


def _warm(stream):
    """Touch a stream once so the Ascend allocator binds to it."""
    try:
        torch.npu.current_stream().wait_stream(stream)
        with torch.npu.stream(stream):
            probe = torch.empty(8, dtype=torch.float16, device=torch.npu.current_device())
            del probe
        stream.synchronize()
    except Exception:  # noqa: BLE001
        pass
    return stream


def patch_stream_factory() -> bool:
    """Wrap the platform's Stream factory so every stream is allocator-bound.

    ``torch_npu.Stream`` is a C-extension type, so its ``__init__`` cannot be
    replaced from Python. ``current_omni_platform`` is an instance, so an
    instance attribute shadows the class attribute cleanly instead.
    """
    global _STREAM_FACTORY_PATCHED
    if _STREAM_FACTORY_PATCHED:
        return False
    if os.environ.get("VLLM_OMNI_910A_STREAM_WARMUP", "1") == "0":
        return False
    try:
        from vllm_omni.platforms import current_omni_platform as platform
    except Exception:  # noqa: BLE001
        return False

    original = getattr(platform, "Stream", None)
    if original is None:
        return False

    def warmed(*args, **kwargs):
        # 910A: an allocator is only bound to streams created on the thread that
        # owns them. The DLO worker issues its collectives from a different
        # thread, so a private stream there dies with aclrtAllocatorGetByStream.
        # Reusing the caller's current stream keeps every collective on a stream
        # this thread already owns. Overlap is lost; correctness is not.
        if os.environ.get("VLLM_OMNI_910A_SHARED_STREAM", "1") != "0":
            return torch.npu.current_stream()
        return _warm(original(*args, **kwargs))

    try:
        platform.Stream = warmed
    except Exception:  # noqa: BLE001
        return False
    _STREAM_FACTORY_PATCHED = True
    return True


try:
    patch_stream_factory()
except Exception:  # noqa: BLE001
    pass


apply()