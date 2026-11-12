#!/usr/bin/env python3
"""Use index_copy_ instead of index_add_ for H3's packed embeddings.

``minimax_h3_transformer`` builds the packed embedding table with three
``index_add_`` calls. The three index sets (text / image / audio local positions)
are disjoint and the table starts as zeros, so ``index_copy_`` is exactly
equivalent -- but it lowers to a different Ascend op than InplaceIndexAdd, which
the 910A aclnn build rejects in this pipeline.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py")

OLD = """        embeddings.index_add_(
            0,
            text_local_pos,
            text_embed.to(_BF16_DTYPE)[: text_local_pos.shape[0]],
        )
        embeddings.index_add_(
            0,
            img_local_pos,
            video_embed.to(_BF16_DTYPE)[: img_local_pos.shape[0]],
        )
        embeddings.index_add_(
            0,
            audio_local_pos,
            audio_embed.to(_BF16_DTYPE)[: audio_local_pos.shape[0]],
        )"""

NEW = """        # 910A rejects InplaceIndexAdd on this path. The three position sets are
        # disjoint and `embeddings` starts at zero, so index_copy_ (a different
        # Ascend kernel) is equivalent.
        embeddings.index_copy_(
            0,
            text_local_pos,
            text_embed.to(_BF16_DTYPE)[: text_local_pos.shape[0]],
        )
        embeddings.index_copy_(
            0,
            img_local_pos,
            video_embed.to(_BF16_DTYPE)[: img_local_pos.shape[0]],
        )
        embeddings.index_copy_(
            0,
            audio_local_pos,
            audio_embed.to(_BF16_DTYPE)[: audio_local_pos.shape[0]],
        )"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "a different\n        # Ascend kernel) is equivalent" in text or "index_copy_(\n            0,\n            text_local_pos" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())