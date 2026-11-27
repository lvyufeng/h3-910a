"""Speedups for the MiniMax-H3 910A service, installable in *any* process.

Everything here is either bit-identical to the verified R124/R126 path or is the
user-approved FA V1 change (relRMS 1.29e-4, one FP16 ulp). Each item is gated by
an environment flag so a deployment can disable it without touching code:

  H3_SERVICE_FA_V1         FA V1 kernels: PV before the next QK matmul, 128/384 tiles
  H3_SERVICE_GUARD_FAST    fast FP16-in/FP32-out linear guard (bitwise identical)
  H3_SERVICE_FAST_GEMM     fp32-output aclnn form for fc1/qkv (torch.equal verified)

``install()`` is idempotent and safe to call from a sitecustomize hook in every
spawned worker process.
"""
from __future__ import annotations

import math
import os
import threading

_APPLIED: list[str] = []
_LOCK = threading.Lock()


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "off", "no")


def applied() -> list[str]:
    return list(_APPLIED)


def install(verbose: bool = False) -> list[str]:
    """Install every enabled speedup once per process."""
    with _LOCK:
        if _APPLIED:
            return list(_APPLIED)
        try:
            import torch  # noqa: F401
        except Exception:
            return []
        if _flag("H3_SERVICE_FAST_GEMM"):
            _install_fast_gemm()
        if _flag("H3_SERVICE_GUARD_FAST"):
            _install_guard_fast()
        if _flag("H3_SERVICE_FA_V1"):
            _install_fa_v1()
        if verbose:
            print(f"[h3-opt] applied: {','.join(_APPLIED)}", flush=True)
        """remember that at least one attempt ran so the lock short-circuits"""
        if not _APPLIED:
            _APPLIED.append("none")
    return list(_APPLIED)


# ---------------------------------------------------------------------------
def _install_guard_fast() -> None:
    """Bitwise-identical fast path for the FP16-input / FP32-output linear."""
    import torch

    import h3_replay123.transformer as T

    cls = T._MiniMaxH3HalfInputFP32OutputLinearMethod
    if getattr(cls.apply, "_h3_fast_guard", False):
        _APPLIED.append("guard_fast(already)")
        return
    original_apply = cls.apply

    def apply_fast(self, layer, x, bias=None):
        if x.device.type not in ("cuda", "npu") or layer.weight.dtype != torch.float16:
            raise TypeError("service requires FP16 matrix weights on the accelerator")
        peak = float(x.abs().amax(dim=tuple(range(x.ndim))))
        if not math.isfinite(peak):
            raise FloatingPointError("non-finite guard peak")
        if peak > 32752.0:
            return original_apply(self, layer, x, bias)
        result = torch.mm(
            x.reshape(-1, x.shape[-1]), layer.weight.t(), out_dtype=torch.float32
        ).reshape(*x.shape[:-1], layer.weight.shape[0])
        if bias is not None:
            result = result + bias.float()
        return result

    apply_fast._h3_fast_guard = True
    cls.apply = apply_fast
    _APPLIED.append("guard_fast")


def _install_fast_gemm() -> None:
    """Bit-identical (torch.equal) fp32-output form for the two column-parallel GEMMs."""
    import torch

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod as _UQ

    if getattr(_UQ.apply, "_h3_fast_gemm", False):
        _APPLIED.append("fast_gemm(already)")
        return
    original_apply = _UQ.apply

    def fast_apply(self, layer, x, bias=None):
        try:
            import torch_npu  # noqa: F401

            w = layer.weight
            if (bias is None and w.dtype == torch.float16 and x.dtype == torch.float16
                    and w.dim() == 2 and w.shape[1] == 5376
                    and w.shape[0] in (14336, 10752)):
                nz = getattr(layer, "_h3_nz_weight", None)
                if w.shape[0] == 14336 and nz is None:
                    try:
                        nz = torch_npu.npu_format_cast(w.t().contiguous(), 2)
                    except Exception:
                        nz = None
                    layer._h3_nz_weight = nz
                b = nz if (w.shape[0] == 14336 and nz is not None) else w.t()
                out = torch.mm(x.reshape(-1, w.shape[1]), b, out_dtype=torch.float32)
                return out.half().reshape(*x.shape[:-1], w.shape[0])
        except Exception:
            pass
        return original_apply(self, layer, x, bias)

    fast_apply._h3_fast_gemm = True
    _UQ.apply = fast_apply
    _APPLIED.append("fast_gemm")


def _install_fa_v1() -> None:
    """User-approved FA V1 kernels behind the same trim wrapper as the reference runs."""
    os.environ.setdefault("H3_FA_TRIM_Q_LIMIT", "41792")
    os.environ.setdefault("H3_FA215_PKG", "/tmp/fa_pkg_v1")
    os.environ.setdefault("H3_FA215_KERNEL_DIR", "/tmp/fa910a_kernels")
    os.environ.setdefault("H3_FA_BR_CAP", "128")
    os.environ.setdefault("H3_FA_BC_CAP", "384")
    try:
        from vllm_omni.diffusion.attention.backends import fa910a as deployed

        import bb218trim as trim_mod

        deployed.flash_attention_910a = trim_mod.flash_attention_910a
        _APPLIED.append("fa_v1_128x384")
    except Exception as exc:  # pragma: no cover - deployment dependent
        print(f"[h3-opt] FA V1 not installed: {exc!r}", flush=True)