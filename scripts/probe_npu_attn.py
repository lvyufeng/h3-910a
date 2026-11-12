#!/usr/bin/env python3
"""Find an attention implementation that actually runs on this 910A box.

``torch.nn.functional.scaled_dot_product_attention`` fails for every shape we
tried ("can not cast format when output is input"), so this probes the two
realistic replacements: a plain matmul/softmax formulation, and the fused
Ascend kernel.
"""
import time
import traceback

import torch
import torch_npu  # noqa: F401

DEVICE = "npu:0"


def manual_attention(q, k, v, mask=None):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def fused_attention(q, k, v, heads):
    out = torch_npu.npu_fusion_attention(
        q, k, v, heads, "BNSD", scale=1.0 / (q.shape[-1] ** 0.5),
        keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
    )
    return out[0]


def run(name, fn):
    try:
        started = time.time()
        out = fn()
        torch.npu.synchronize()
        elapsed = time.time() - started
        finite = bool(torch.isfinite(out).all())
        print(f"OK   {name:34s} {elapsed*1000:8.1f} ms shape={tuple(out.shape)} finite={finite}")
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name:34s} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        return None


def main():
    for seq in (128, 2048):
        b, h, kv, d = 1, 32, 8, 64
        q = torch.randn(b, h, seq, d, dtype=torch.float16, device=DEVICE)
        k = torch.randn(b, kv, seq, d, dtype=torch.float16, device=DEVICE)
        v = torch.randn(b, kv, seq, d, dtype=torch.float16, device=DEVICE)
        n = h // kv
        ke = k.repeat_interleave(n, dim=1)
        ve = v.repeat_interleave(n, dim=1)

        print(f"--- seq={seq}")
        ref = run(f"manual_{seq}", lambda: manual_attention(q, ke, ve))
        run(f"fused_{seq}", lambda: fused_attention(q, k, v, h))
        if ref is not None:
            manual = manual_attention(q, ke, ve)
            precision_error = (manual.float() - ref.float()).abs().max().item()
            print(f"     manual repeatability max|diff|={precision_error:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()