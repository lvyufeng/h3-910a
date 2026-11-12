#!/usr/bin/env python3
"""Compare eager operators on NPU against CPU to find silent-wrong results.

The 910A box completed a full H3 sampling run but produced a black video, which
means something computed fine (no exception) yet numerically wrong.  This probe
feeds identical fp16 inputs to both devices and reports the max deviation.
"""
from __future__ import annotations

import torch
import torch_npu  # noqa: F401

DEVICE = "npu:0"
DTYPE = torch.float16
TOL = 2e-2


def check(name, fn, *tensors):
    cpu = [t.float().cpu() for t in tensors]
    npu = [t.to(DEVICE) for t in tensors]
    try:
        ref = fn(*cpu)
        out = fn(*npu)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        return
    ref_f = ref.float()
    out_f = out.float().cpu()
    if ref_f.shape != out_f.shape:
        print(f"SHAPE {name}: {tuple(ref_f.shape)} vs {tuple(out_f.shape)}")
        return
    diff = (ref_f - out_f).abs().max().item()
    flag = "OK  " if diff <= TOL else "BAD "
    print(f"{flag}{name:22s} max|diff|={diff:.5f} ref_mean={ref_f.mean().item():+.3f} out_mean={out_f.mean().item():+.3f}")


def main():
    torch.manual_seed(0)
    a = torch.randn(4, 64, 64, dtype=DTYPE)
    b = torch.randn(4, 64, 64, dtype=DTYPE)
    c = torch.randn(64, 64, dtype=DTYPE)

    check("matmul", lambda x, y: torch.matmul(x, y), a[0], c)
    check("bmm", lambda x, y: torch.bmm(x, y), a, b)
    check("baddbmm", lambda x, y, z: torch.baddbmm(x, y, z), a, b, b)

    s = torch.randn(4, 64, 64, dtype=DTYPE) * 3
    check("softmax_fp16", lambda x: torch.softmax(x, dim=-1), s)
    check("softmax_fp32cast", lambda x: torch.softmax(x.float(), dim=-1), s)

    check("exp", lambda x: torch.exp(x), s)
    check("silu", lambda x: torch.nn.functional.silu(x), a)
    check("gelu", lambda x: torch.nn.functional.gelu(x), a)

    x = torch.randn(2, 8, 32, 32, dtype=DTYPE)
    w = torch.randn(8, 8, 3, 3, dtype=DTYPE)
    check("conv2d", lambda i, k: torch.nn.functional.conv2d(i, k, padding=1), x, w)

    g = torch.randn(2, 8, 32, 32, dtype=DTYPE)
    check("group_norm", lambda t: torch.nn.functional.group_norm(t, 8), g)

    ln = torch.randn(2, 32, 128, dtype=DTYPE)
    check("layer_norm", lambda t: torch.nn.functional.layer_norm(t, (128,)), ln)

    q = torch.randn(2, 8, 32, 16, dtype=DTYPE)
    k2 = torch.randn(2, 8, 32, 16, dtype=DTYPE)
    check("chunked_attn_bmm", lambda qq, kk: torch.softmax(
        torch.matmul(qq, kk.transpose(-1, -2)) * (1 / qq.shape[-1] ** 0.5), dim=-1), q, k2)


if __name__ == "__main__":
    main()