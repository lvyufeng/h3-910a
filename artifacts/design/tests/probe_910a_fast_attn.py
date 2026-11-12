#!/usr/bin/env python3
"""Hunt for a usable fast-attention kernel on Ascend 910A.

The chunked matmul fallback is numerically correct but O(S^2) in eager mode,
which cannot reach a 10s-video budget.  This probes every fused variant the
installed torch_npu/CANN exposes and reports which ones actually launch.
"""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

DEVICE = "npu:0"


def report(name, fn):
    try:
        start = time.time()
        out = fn()
        torch.npu.synchronize()
        elapsed = time.time() - start
        finite = bool(torch.isfinite(out).all()) if torch.is_tensor(out) else True
        print(f"OK   {name:38s} {elapsed*1000:8.1f} ms finite={finite} shape={tuple(out.shape)}")
        return out
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0][:110]
        print(f"FAIL {name:38s} {type(exc).__name__}: {msg}")
        return None


def main():
    B, H, KV, S, D = 1, 32, 8, 4096, 128
    q = torch.randn(B, H, S, D, dtype=torch.float16, device=DEVICE)
    k = torch.randn(B, KV, S, D, dtype=torch.float16, device=DEVICE)
    v = torch.randn(B, KV, S, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / (D ** 0.5)

    print("== fused attention variants ==")
    report("fusion_attention BNSD gqa", lambda: torch_npu.npu_fusion_attention(
        q, k, v, H, "BNSD", scale=scale, keep_prob=1.0,
        pre_tockens=2147483647, next_tockens=2147483647)[0])
    report("fusion_attention BNSD no gqa", lambda: torch_npu.npu_fusion_attention(
        q, q.clone(), q.clone(), H, "BNSD", scale=scale, keep_prob=1.0,
        pre_tockens=2147483647, next_tockens=2147483647)[0])
    report("fusion_attention BNSD + mask", lambda: torch_npu.npu_fusion_attention(
        q, k, v, H, "BNSD",
        atten_mask=torch.ones(2048, 2048, dtype=torch.bool, device=DEVICE).triu(1),
        scale=scale, keep_prob=1.0, pre_tockens=2147483647,
        next_tockens=2147483647)[0])
    report("fusion_attention BSH", lambda: torch_npu.npu_fusion_attention(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(), H, "BSH", scale=scale, keep_prob=1.0,
        pre_tockens=2147483647, next_tockens=2147483647)[0])

    for attr in ("npu_prompt_flash_attention", "npu_incre_flash_attention"):
        fn = getattr(torch_npu, attr, None)
        print(f"   {attr}: {'present' if fn else 'absent'}")

    report("sdpa fp16", lambda: torch.nn.functional.scaled_dot_product_attention(
        q[:, :8], k, v, enable_gqa=True))

    print("\n== raw matmul ceiling ==")
    a = torch.randn(4096, 4096, dtype=torch.float16, device=DEVICE)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=DEVICE)
    for label, fn in (
        ("mm 4096^3 (warm)", lambda: torch.matmul(a, b)),
        ("mm 4096^3 (again)", lambda: torch.matmul(a, b)),
    ):
        start = time.time()
        out = fn()
        torch.npu.synchronize()
        elapsed = time.time() - start
        tflops = 2 * 4096 ** 3 / elapsed / 1e12
        print(f"OK   {label:38s} {elapsed*1000:8.1f} ms ~{tflops:6.1f} TFLOPS")

    kt = k.transpose(-1, -2)
    start = time.time()
    torch.matmul(q[:, :8], kt)
    torch.npu.synchronize()
    elapsed = time.time() - start
    tflops = 2 * 8 * S * S * D / elapsed / 1e12
    print(f"OK   {'qk^T 8 heads S=4096':38s} {elapsed*1000:8.1f} ms ~{tflops:6.1f} TFLOPS")


if __name__ == "__main__":
    main()