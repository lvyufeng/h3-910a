#!/usr/bin/env python3
"""Trace every aten::nan_to_num dispatch with a Python stack.

910A has no NanToNum kernel, so the video VAE fails during decode with

    No supported Ops kernel and engine are found for [NanToNum542]

Patching ``torch.Tensor.nan_to_num`` does nothing: it is a method on a C
extension type, so assignment is silently ignored. This registers a dispatcher
override for ``aten::nan_to_num`` instead, which sees every call regardless of
how it is spelled, and prints the Python stack that requested it.

Debug only: remove or gate behind an env var before trusting timings.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/h3_910a_sdpa_shim.py")

BLOCK = '''# --- trace aten::nan_to_num ------------------------------------------------
# Assignment to torch.Tensor.nan_to_num is ignored (C extension type), so hook
# the dispatcher instead to find who requests the unsupported 910A op.
import os as _os_n2n

if _os_n2n.environ.get("VLLM_OMNI_910A_TRACE_N2N", "0") != "0":
    try:
        import traceback as _tb_n2n

        import torch.library as _tlib

        _N2N_LIB = _tlib.Library("vllm_omni_910a_dbg", "IMPL", "CompositeExplicitAutograd")

        @_tlib.impl(_N2N_LIB, "nan_to_num", "CompositeExplicitAutograd")
        def _traced_nan_to_num_impl(self, nan=None, posinf=None, neginf=None):
            import sys as _s_n2n

            _s_n2n.stderr.write(
                "[N2NTRACE] " + "".join(_tb_n2n.format_stack(limit=18)) + "\\n"
            )
            _s_n2n.stderr.flush()
            nan = 0.0 if nan is None else nan
            return torch.where(
                torch.isfinite(self),
                self,
                torch.where(self != self, torch.full_like(self, float(nan)), self),
            )
    except Exception:
        pass

'''

MARKER = "# --- hard-disable torch.compile"


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "N2NTRACE" in text:
        print("already patched")
        return 0
    if MARKER not in text:
        raise SystemExit("anchor not found")
    text = text.replace(MARKER, BLOCK + MARKER, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("patched dispatcher trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())