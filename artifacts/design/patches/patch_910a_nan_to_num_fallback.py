#!/usr/bin/env python3
"""DEAD END -- kept only as a record. Does not fix the 910A NanToNum failure.

Measured outcome: registering ``aten::nan_to_num`` on the
CompositeExplicitAutograd key is bypassed for fp16, because the Ascend backend
kernel is still selected for that dtype. fp16 kept failing exactly as before;
only fp32/bf16 went through the Python implementation.

The fix that works edits the model's ``video_vae/flash.py`` so it never calls
``nan_to_num`` on an fp16 tensor at all. Use
``patch_model_vae_nan_to_num.py`` instead.

--- original notes below (the reasoning that turned out to be wrong) ---

Provide a working ``aten::nan_to_num`` for 910A.

910A ships no NanToNum kernel for fp16 (fp32 and bf16 do have one):

    torch.nan_to_num(fp16) -> NPU out of memory / "Inner error"
    No supported Ops kernel and engine are found for [NanToNum542]

The video VAE reaches it during decode (``F.rms_norm`` and the remote module's
``flash.py`` both route through it), which fails the whole request after the
denoise loop has already succeeded.

Rather than chase every call site, register the op itself: a Python kernel for
``aten::nan_to_num`` intercepts at the dispatcher, before the missing Ascend
kernel is selected, and implements the documented semantics with primitives that
do have kernels.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/h3_910a_sdpa_shim.py")

BLOCK = '''# --- aten::nan_to_num fallback for 910A --------------------------------------
# 910A has no fp16 NanToNum kernel. Register a Python implementation so the
# dispatcher never reaches the missing one; the video VAE needs it during decode.
import os as _os_n2n_fallback

if _os_n2n_fallback.environ.get("VLLM_OMNI_910A_N2N_FALLBACK", "1") != "0":
    try:
        import torch.library as _tlib_n2n

        _N2N_FALLBACK_LIB = _tlib_n2n.Library(
            "vllm_omni_910a_n2n", "IMPL", "CompositeExplicitAutograd"
        )

        @_tlib_n2n.impl(_N2N_FALLBACK_LIB, "nan_to_num", "CompositeExplicitAutograd")
        def _nan_to_num_fallback_910a(self, nan=None, posinf=None, neginf=None):
            nan_v = 0.0 if nan is None else float(nan)
            pos_v = None if posinf is None else float(posinf)
            neg_v = None if neginf is None else float(neginf)

            finite = torch.isfinite(self)
            out = torch.where(finite, self, torch.full_like(self, nan_v))

            if pos_v is not None or neg_v is not None:
                is_pos_inf = self == float("inf")
                is_neg_inf = self == float("-inf")
                if pos_v is not None:
                    out = torch.where(is_pos_inf, torch.full_like(self, pos_v), out)
                if neg_v is not None:
                    out = torch.where(is_neg_inf, torch.full_like(self, neg_v), out)
            return out
    except Exception:
        pass

'''

MARKER = "# --- hard-disable torch.compile"


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_nan_to_num_fallback_910a" in text:
        print("already patched")
        return 0
    if MARKER not in text:
        raise SystemExit("anchor not found")
    text = text.replace(MARKER, BLOCK + MARKER, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("patched: aten::nan_to_num fallback registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())