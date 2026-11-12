#!/usr/bin/env python3
"""Probe which scaled_dot_product_attention shapes work on this NPU."""
import traceback

import torch
import torch_npu  # noqa: F401

DEVICE = "npu:0"
B, H, KV, S, D = 1, 32, 8, 128, 64


def make(dtype=torch.float16):
    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, KV, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, KV, S, D, dtype=dtype, device=DEVICE)
    return q, k, v


def expand(k, v, h):
    n = h // k.shape[1]
    return k.repeat_interleave(n, dim=1), v.repeat_interleave(n, dim=1)


def run(name, fn):
    try:
        out = fn()
        torch.npu.synchronize()
        print(f"OK   {name} shape={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")


def main():
    q, k, v = make()

    run("sdpa_gqa_true_no_mask",
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True))

    run("sdpa_gqa_false_broadcast",
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    ke, ve = expand(k, v, H)
    run("sdpa_expanded_no_mask",
        lambda: torch.nn.functional.scaled_dot_product_attention(q, ke, ve))

    mask = torch.zeros(B, 1, S, S, dtype=torch.float16, device=DEVICE)
    run("sdpa_expanded_float_mask",
        lambda: torch.nn.functional.scaled_dot_product_attention(q, ke, ve, attn_mask=mask))

    bool_mask = torch.ones(B, 1, S, S, dtype=torch.bool, device=DEVICE)
    run("sdpa_expanded_bool_mask",
        lambda: torch.nn.functional.scaled_dot_product_attention(q, ke, ve, attn_mask=bool_mask))

    # The text encoder runs bf16 on some paths; check the same shapes there.
    qb, kb, vb = make(torch.bfloat16)
    keb, veb = expand(kb, vb, H)
    run("sdpa_expanded_bf16",
        lambda: torch.nn.functional.scaled_dot_product_attention(qb, keb, veb))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()