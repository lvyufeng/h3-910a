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
# 910A has no fused attention op: aclnnFlashAttentionScore ships without an
# inferShape, so every fallback (SDPA, npu_fusion_attention v1/v2/v3,
# prompt/incremental FA, fused_infer_attention_score) fails registration.
# What is left is matmul + softmax, and measured on the real shape
# (H=7, S=73408, D=128) the split is softmax 64% / matmul 36%. Softmax is
# slow because it streams the whole S x S tile several times: for one query
# chunk the score tile is already 1.05 GB, and the full matrix would be
# 75 GB at fp16.
_DEFAULT_KBLOCK = 4096
_DEFAULT_DENSITY = 1.0


def _attn_chunk(queries: int) -> int:
    raw = os.environ.get("VLLM_OMNI_910A_ATTN_CHUNK", str(_DEFAULT_CHUNK))
    try:
        chunk = int(raw)
    except (TypeError, ValueError):
        chunk = _DEFAULT_CHUNK
    return max(1, min(chunk, queries))


def _key_block(keys: int) -> int:
    raw = os.environ.get("VLLM_OMNI_910A_ATTN_KBLOCK", str(_DEFAULT_KBLOCK))
    try:
        block = int(raw)
    except (TypeError, ValueError):
        block = _DEFAULT_KBLOCK
    return max(256, min(block, max(256, keys)))


def _density(step: int | None) -> float:
    """Fraction of key blocks to keep. 1.0 keeps the dense path.

    ``VLLM_OMNI_910A_ATTN_DENSITY_STEPS`` may override per denoise step, e.g.
    ``0:1.0,1:0.2,2:0.2,3:1.0`` to keep the first and last steps dense, which
    is how the RainFusion recipe protects quality.
    """
    raw = os.environ.get("VLLM_OMNI_910A_ATTN_DENSITY", str(_DEFAULT_DENSITY))
    if step is not None:
        per = os.environ.get("VLLM_OMNI_910A_ATTN_DENSITY_STEPS", "")
        if per:
            for item in per.split(","):
                k, _, val = item.partition(":")
                if k.strip().isdigit() and int(k.strip()) == step:
                    raw = val.strip()
                    break
    try:
        d = float(raw)
    except (TypeError, ValueError):
        d = _DEFAULT_DENSITY
    return min(1.0, max(0.01, d))


def _select_by_importance(qc, key, kblock, keys, nblocks, keep, stride):
    """Rank key blocks by how much they actually matter for this query block.

    A fixed local window is wrong here on two counts: the packed sequence starts
    with the text conditioning rows, so a window far from index 0 would drop the
    prompt entirely, and video content does not necessarily attend locally. So
    score every key block with a subsample of the queries (``stride`` rows),
    which costs ``1/stride`` of the full matmul, and keep the top ``keep``.
    """
    qp = qc[..., ::stride, :]
    imp = torch.empty(nblocks, dtype=torch.float32, device=qc.device)
    for b in range(nblocks):
        ks = b * kblock
        ke = min(ks + kblock, keys)
        kt = key[..., ks:ke, :].transpose(-1, -2).contiguous()
        s = torch.matmul(qp, kt)
        # A whole-tensor fp16 amax() fails on 910A for some shapes (ReduceMax
        # reports an inner error, e.g. on (1,2,256,256) while (1,7,32,4096)
        # works), so reduce one axis at a time; the keepdim form is reliable.
        imp[b] = s.amax(dim=-1).amax()
    # The first block carries the text/conditioning rows; never drop it.
    imp[0] = float("inf")
    if keep >= nblocks:
        return list(range(nblocks))
    top = torch.topk(imp, keep).indices.tolist()
    return sorted(int(b) for b in top)


def _select_key_blocks(centre: int, kblock: int, nblocks: int, keep: int) -> list[int]:
    """Pick ``keep`` key blocks: a local window plus evenly strided global ones.

    The packed H3 sequence is frame-major then row-major, so a block index
    distance is a spatio-temporal distance and a local window is the
    meaningful prior. The strided part keeps long-range reach, which a pure
    window would lose.
    """
    if keep >= nblocks:
        return list(range(nblocks))
    home = min(centre // kblock, nblocks - 1)
    n_local = max(1, int(round(keep * 0.75)))
    n_global = max(0, keep - n_local)
    chosen: list[int] = []
    for d in range(nblocks):
        for cand in (home - d, home + d):
            if 0 <= cand < nblocks and cand not in chosen:
                chosen.append(cand)
                if len(chosen) >= n_local:
                    break
        if len(chosen) >= n_local:
            break
    if n_global and nblocks > len(chosen):
        stride = max(1, nblocks // n_global)
        cand = list(range(0, nblocks, stride))
        add = [b for b in cand if b not in chosen]
        need = min(n_global, len(add))
        if need:
            pick = [add[int(i * (len(add) - 1) / max(1, need - 1))] for i in range(need)] if need > 1 else add[:1]
            for b in pick:
                if b not in chosen:
                    chosen.append(b)
    return sorted(chosen)


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

    def _sparse_flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        step: int | None,
    ) -> torch.Tensor:
        """Block-sparse attention with online softmax.

        Two wins over the dense path. It skips ``1 - density`` of the key
        blocks, and because it never materialises the full S x K score tile the
        per-block tile is megabytes rather than the 1.05 GB that made softmax
        bandwidth bound. Accumulation is fp32: fp16 over many blocks loses too
        much in the running sum.
        """
        scale = self.softmax_scale
        if scale is None:
            scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale
        b, h, s, d = query.shape
        keys = key.shape[-2]
        kblock = _key_block(keys)
        nblocks = (keys + kblock - 1) // kblock
        density = _density(step)
        keep = max(1, min(nblocks, int(round(nblocks * density))))
        chunk = _attn_chunk(s)
        mode = os.environ.get("VLLM_OMNI_910A_ATTN_SELECT", "importance").strip().lower()
        try:
            proxy_stride = max(1, int(os.environ.get("VLLM_OMNI_910A_ATTN_PROXY_STRIDE", "32")))
        except (TypeError, ValueError):
            proxy_stride = 32

        out = torch.empty((b, h, s, d), dtype=query.dtype, device=query.device)
        for qs in range(0, s, chunk):
            qe = min(qs + chunk, s)
            qc = query[..., qs:qe, :]
            if mode == "importance":
                sel = _select_by_importance(
                    qc, key, kblock, keys, nblocks, keep, proxy_stride
                )
            else:
                sel = _select_key_blocks((qs + qe) // 2, kblock, nblocks, keep)
            run_max = None
            run_sum = None
            run_acc = None
            for bi in sel:
                ks = bi * kblock
                ke = min(ks + kblock, keys)
                kt = key[..., ks:ke, :].transpose(-1, -2).contiguous()
                scores = torch.matmul(qc, kt)
                blk_max = scores.amax(dim=-1, keepdim=True)
                new_max = blk_max if run_max is None else torch.maximum(run_max, blk_max)
                # scores - new_max <= 0 by construction, so exp cannot overflow
                # even in fp16 and no masking sentinel is needed. Doing the exp
                # in the native dtype matters: a float round trip here costs two
                # extra passes over a 16 MB tile per block.
                scores.sub_(new_max).exp_()
                weights = scores
                part = torch.matmul(weights, value[..., ks:ke, :])
                blk_sum = weights.sum(dim=-1, keepdim=True)
                if run_acc is None:
                    run_acc = part.float()
                    run_sum = blk_sum.float()
                else:
                    # corr <= 1 so this rescale is safe in fp32.
                    corr = torch.exp((run_max - new_max).float())
                    run_acc = run_acc * corr + part.float()
                    run_sum = run_sum * corr + blk_sum.float()
                run_max = new_max
            out[..., qs:qe, :] = (run_acc / run_sum).to(query.dtype)
        return out

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
        if os.environ.get("VLLM_OMNI_910A_SHAPE_LOG", "0") != "0":
            import sys as _s
            print("[SHAPE] q=%s k=%s v=%s dtype=%s" % (
                tuple(query.shape), tuple(key.shape), tuple(value.shape), query.dtype),
                file=_s.stderr, flush=True)
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

        step = None
        raw_step = os.environ.get("VLLM_OMNI_910A_DENOISE_STEP")
        if raw_step is not None and raw_step.isdigit():
            step = int(raw_step)
        density = _density(step)
        use_sparse = (
            density < 1.0
            and attention_mask is None
            and not bool(self.causal)
        )
        if use_sparse:
            out = self._sparse_flash_attention(query, key, value, step)
        else:
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
