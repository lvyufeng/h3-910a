"""Runtime torch patches for Ascend 910A, applied at interpreter start-up.

Loaded automatically by ``site`` when ``scripts/`` is on ``PYTHONPATH`` (the H3
launcher puts it there).  Doing it here instead of editing ``main.py`` keeps the
patch independent of ComfyUI's version and also covers worker subprocesses.

Only active when ``H3_910A_RUNTIME_PATCH=1`` (set by ``start-h3-instance.sh``),
so ordinary Python processes on this box are unaffected.

Two 910A gaps are covered:

* ``torch.nan_to_num`` — the NanToNum operator does not exist on 910A.
* conv3d internal format — ``torch_npu`` converts conv3d output to
  NDC1HWC0/ACL_FRACTAL_Z_3D by default, which ``reflection_pad3d`` rejects, so
  the video VAE encode crashes.  Turning it off keeps tensors in ND layout.
"""

from __future__ import annotations

import logging
import os

LOG = logging.getLogger("h3_910a_runtime")


def _patch_nan_to_num(torch):
    if getattr(torch, "_h3_910a_nan_to_num", False):
        return

    def _nan_to_num(x, nan=0.0, posinf=None, neginf=None, *, out=None):
        if not torch.is_tensor(x) or not x.dtype.is_floating_point:
            return x if out is None else out.copy_(x)
        value = x
        if nan is not None:
            value = torch.where(torch.isnan(value), torch.full_like(value, nan), value)
        if posinf is not None:
            value = torch.where(torch.isposinf(value), torch.full_like(value, posinf), value)
        if neginf is not None:
            value = torch.where(torch.isneginf(value), torch.full_like(value, neginf), value)
        if out is not None:
            out.copy_(value)
            return out
        return value

    torch.nan_to_num = _nan_to_num
    torch.Tensor.nan_to_num = lambda self, *a, **k: _nan_to_num(self, *a, **k)
    torch.Tensor.nan_to_num_ = lambda self, *a, **k: self.copy_(_nan_to_num(self, *a, **k))
    torch._h3_910a_nan_to_num = True
    LOG.info("patched torch.nan_to_num for Ascend 910A (NanToNum unsupported)")


def _patch_internal_format(torch_npu):
    config = getattr(torch_npu, "config", None)
    if config is None or not hasattr(config, "allow_internal_format"):
        return
    if getattr(config, "allow_internal_format", False):
        config.allow_internal_format = False
        LOG.info("disabled torch_npu internal format (reflection_pad3d needs ND layout)")


def _apply():
    try:
        import torch  # noqa: F401
        import torch_npu
    except Exception:  # noqa: BLE001 - not every process on this box has NPU libs
        return
    try:
        if torch.npu.is_available():
            _patch_internal_format(torch_npu)
            _patch_nan_to_num(torch)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("910A runtime patch skipped: %s", exc)


if os.environ.get("H3_910A_RUNTIME_PATCH", "0").strip().lower() in {"1", "true", "yes", "on"}:
    _apply()