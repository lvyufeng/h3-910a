#!/usr/bin/env python3
"""Numerical check for the 910A attention backend (NPU vs CPU reference).

Runs the backend's forward_npu directly, so it needs no weights or server.
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.h3_910a_attn import H3910AAttentionImpl

DEVICE = "npu:0"
DTYPE = torch.float16


def reference(q, k, v, mask=None, causal=False):
    """Float32 matmul attention on CPU: q/k/v in (B, S, H, D)."""
    q, k, v = (x.float().permute(0, 2, 1, 3) for x in (q, k, v))
    if q.shape[1] != k.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        n, m = scores.shape[-2], scores.shape[-1]
        tri = torch.ones(n, m, dtype=torch.bool).tril()
        scores = scores.masked_fill(~tri, float("-inf"))
    if mask is not None:
        m = mask if mask.dtype == torch.bool else mask.bool()
        while m.dim() < scores.dim():
            m = m.unsqueeze(1)
        scores = scores.masked_fill(~m, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out.permute(0, 2, 1, 3)


def check(name, *, batches=1, seq_q=256, seq_k=256, heads=8, kv_heads=8, dim=64,
          causal=False, masked=False, chunk=None):
    import os

    if chunk:
        os.environ["VLLM_OMNI_910A_ATTN_CHUNK"] = str(chunk)

    torch.manual_seed(0)
    q = torch.randn(batches, seq_q, heads, dim, dtype=DTYPE)
    k = torch.randn(batches, seq_k, kv_heads, dim, dtype=DTYPE)
    v = torch.randn(batches, seq_k, kv_heads, dim, dtype=DTYPE)
    mask = None
    if masked:
        mask = torch.ones(batches, seq_k, dtype=torch.bool)
        mask[:, seq_k // 2:] = False

    backend = H3910AAttentionImpl(
        num_heads=heads, head_size=dim, softmax_scale=dim ** -0.5, causal=causal,
        num_kv_heads=kv_heads,
    )
    metadata = AttentionMetadata(attn_mask=mask) if mask is not None else None
    with torch.no_grad():
        out = backend.forward_npu(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), metadata)
    torch.npu.synchronize()

    ref = reference(q, k, v, mask=mask, causal=causal)
    got = out.float().cpu()
    if got.shape != ref.shape:
        print(f"SHAPE {name}: {tuple(got.shape)} vs {tuple(ref.shape)}")
        return False
    diff = (got - ref).abs().max().item()
    finite = bool(torch.isfinite(got).all())
    ok = finite and diff < 5e-2
    print(f"{'OK  ' if ok else 'BAD '} {name:34s} max|diff|={diff:.5f} finite={finite} "
          f"ref_mean={ref.mean().item():+.4f}")
    return ok


def main():
    results = [
        check("mha_basic", seq_q=256, seq_k=256),
        check("mha_long_chunked", seq_q=4096, seq_k=4096, chunk=1024),
        check("gqa_8x2", heads=8, kv_heads=2),
        check("causal", causal=True),
        check("masked_k", masked=True),
        check("batched", batches=2, seq_q=512, seq_k=512, heads=4, kv_heads=4),
        check("single_chunk_tiny", seq_q=16, seq_k=16, heads=2, kv_heads=2, dim=32),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main()