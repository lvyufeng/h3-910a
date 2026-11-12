# SPDX-License-Identifier: Apache-2.0
"""Ascend 910A-safe diffusion attention backend.

The 910A generation NPU cannot run either attention kernel the upstream NPU
path depends on:

* ``torch.nn.functional.scaled_dot_product_attention`` fails with
  ``ERR01007 can not cast format when output is input`` for every shape we
  tried (with or without masks, fp16 or bf16).
* ``torch_npu.npu_fusion_attention`` fails inside
  ``aclnnFlashAttentionScore``.

A plain matmul + softmax formulation does run and matches CPU numerically, so
this backend keeps the SDPA class's interface, mask handling and platform
dispatch, and only replaces the kernel call on NPU. Queries are processed in
blocks so the (heads, queries, keys) score tensor never has to exist in full,
which is what lets a long H3 video latent fit in 32 GB.

Select it with ``--diffusion-attention-backend TORCH_SDPA``; the backend is
installed as that enum member's implementation.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import (
    SDPABackend,
    SDPAImpl,
    _maybe_reshape_attn_mask,
)

logger = init_logger(__name__)

_DEFAULT_CHUNK = 2048


def _attn_chunk(queries: int) -> int:
    raw = os.environ.get("VLLM_OMNI_910A_ATTN_CHUNK", str(_DEFAULT_CHUNK))
    try:
        chunk = int(raw)
    except (TypeError, ValueError):
        chunk = _DEFAULT_CHUNK
    return max(1, min(chunk, queries))


class H3910AAttentionImpl(SDPAImpl):
    """SDPA implementation whose NPU kernel is chunked matmul/softmax."""

    def _matmul_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scale = self.softmax_scale
        if scale is None:
            scale = 1.0 / (query.shape[-1] ** 0.5)
        keys = key.shape[-2]
        # A non-contiguous K^T costs ~24% on this SoC: matmul has to gather
        # strided rows for every chunk, and the same K^T is reused across all
        # of them. Pay one copy up front instead.
        key_t = key.transpose(-1, -2).contiguous()
        queries = query.shape[-2]
        step = _attn_chunk(queries)
        causal = bool(self.causal)
        # Fold the softmax scale into Q once. Q is S x D while the scores are
        # S x N, so scaling here replaces one full pass over the score matrix
        # with a pass over a far smaller tensor.
        query = query * scale
        # Ascend's softmax rejects infinities (it lowers to a NanToNum fallback
        # that has no 910A kernel), so mask with the most negative finite value.
        neg = torch.finfo(query.dtype).min
        # SoftmaxV2 has native fp16 and fp32 kernels but rejects bf16
        # ("Unsupported cast type! src_dtype: bfloat16"), so only bf16 needs the
        # fp32 round trip.
        upcast_softmax = query.dtype == torch.bfloat16

        chunks: list[torch.Tensor] = []
        for start in range(0, queries, step):
            stop = min(start + step, queries)
            # (B, H, block, D) @ (B, H, D, K) -> (B, H, block, K)
            scores = torch.matmul(query[..., start:stop, :], key_t)
            if causal:
                rows = torch.arange(start, stop, device=query.device).view(-1, 1)
                cols = torch.arange(keys, device=query.device).view(1, -1)
                scores = scores.masked_fill(cols > rows, neg)
            if attention_mask is not None:
                mask = attention_mask
                if mask.dim() >= 3 and mask.shape[-2] != 1:
                    mask = mask[..., start:stop, :]
                if mask.dtype != torch.bool:
                    mask = mask.bool()
                if mask.device != scores.device:
                    mask = mask.to(scores.device)
                scores = scores.masked_fill(~mask, neg)
            # Keeping the native dtype avoids a fp32 copy of the whole score
            # tile; that upcast doubled peak attention memory and is what made
            # 768p OOM on 32 GiB cards.
            if upcast_softmax:
                weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            else:
                weights = torch.softmax(scores, dim=-1)
            chunks.append(torch.matmul(weights, value))

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=-2)

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        attention_mask = None
        if attn_metadata:
            attention_mask = _maybe_reshape_attn_mask(
                query, key, attn_metadata.attn_mask, mask_mode="broadcast_k"
            )

        # (B, S, H, D) -> (B, H, S, D)
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        if query.shape[1] != key.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}."
                )
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        out = self._matmul_attention(query, key, value, attention_mask)
        return out.permute(0, 2, 1, 3).contiguous()

class H3910AAttentionBackend(SDPABackend):
    """Backend entry point that serves :class:`H3910AAttentionImpl`."""

    @staticmethod
    def get_name() -> str:
        return "H3_910A"

    @staticmethod
    def get_impl_cls() -> type[H3910AAttentionImpl]:
        return H3910AAttentionImpl
