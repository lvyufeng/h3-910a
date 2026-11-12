#!/usr/bin/env python3
"""Patch ComfyUI's SDPA fallback for Ascend 910A.

On this box every ``torch.nn.functional.scaled_dot_product_attention`` call
fails with ``can not cast format when output is input`` (ERR01007), and
``torch_npu.npu_fusion_attention`` fails in ``aclnnFlashAttentionScore``.
A plain matmul/softmax formulation does run, so this replaces the fallback
implementation with that formulation while keeping the public signature.

Usage:
    python apply_910a_attention.py /path/to/comfy/ops.py [--revert]

The script is idempotent, and writes ``ops.py.before-910a-attention`` next to
the target the first time it changes anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ORIGINAL = '''def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
    if kwargs.get("enable_gqa", False) and attn_mask is not None:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
        kwargs["enable_gqa"] = False
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
'''

PATCHED = '''def _h3_910a_attn_chunk(queries):
    value = os.environ.get("H3_910A_ATTN_CHUNK", "2048")
    try:
        chunk = int(value)
    except (TypeError, ValueError):
        chunk = 2048
    return max(1, min(chunk, queries))


def _h3_910a_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                        scale=None, enable_gqa=False):
    """Chunked matmul/softmax attention for Ascend 910A.

    SDPA is unsupported here (ERR01007) and the fused Ascend kernel fails in
    aclnnFlashAttentionScore, so this reproduces SDPA semantics directly.
    Queries are processed in blocks so the (heads, queries, keys) score tensor
    never has to exist in full: a 22k-token video latent cannot hold it.
    """
    if enable_gqa and q.shape[-3] != k.shape[-3]:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
    factor = scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5)
    keys = k.shape[-2]
    transposed = k.transpose(-1, -2)
    block = _h3_910a_attn_chunk(q.shape[-2])
    chunks = []
    for start in range(0, q.shape[-2], block):
        stop = min(start + block, q.shape[-2])
        scores = torch.matmul(q[..., start:stop, :], transposed) * factor
        if is_causal:
            rows = torch.arange(start, stop, device=q.device).view(-1, 1)
            cols = torch.arange(keys, device=q.device).view(1, -1)
            scores = scores.masked_fill(cols > rows, float("-inf"))
        if attn_mask is not None:
            mask = attn_mask[..., start:stop, :]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask.to(scores.dtype)
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        if dropout_p:
            weights = torch.nn.functional.dropout(weights, p=dropout_p, training=True)
        chunks.append(torch.matmul(weights, v))
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-2)


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
    if comfy.model_management.is_ascend_npu():
        return _h3_910a_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.0),
            is_causal=args[2] if len(args) > 2 else kwargs.get("is_causal", False),
            scale=kwargs.get("scale"),
            enable_gqa=kwargs.get("enable_gqa", False),
        )
    if kwargs.get("enable_gqa", False) and attn_mask is not None:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
        kwargs["enable_gqa"] = False
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="path to comfy/ops.py")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.target)
    text = path.read_text(encoding="utf-8")
    backup = path.with_name("ops.py.before-910a-attention")

    if args.revert:
        if not backup.exists():
            print(f"no backup at {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"reverted {path} from {backup}")
        return 0

    if "_h3_910a_attention" in text:
        print("already patched")
        return 0
    if ORIGINAL not in text:
        print("current ops.py does not match the expected SDPA fallback", file=sys.stderr)
        return 2

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup written to {backup}")
    path.write_text(text.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())