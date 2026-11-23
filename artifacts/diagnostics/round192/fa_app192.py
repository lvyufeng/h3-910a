#!/usr/bin/env python3
"""R192: minimal repeated FA call, for msopprof to profile.

Profiles the isolated real FA shape so msopprof can report cube/vector/MTE pipe utilisation.
This decides whether the 6.00 us per-tile cost is pipeline bubbles (R190's inference).
"""
from __future__ import annotations
import os
import sys

B, H, N, D = 1, 7, 41792, 128
CALLS = int(os.environ.get('H3_FA_CALLS', '6'))


def main():
    import torch
    import torch_npu  # noqa: F401
    sys.path.insert(0, '/tmp/h3_fa191')
    import fa910a_sweep as fa

    torch.npu.set_device(0)
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')

    print('warmup build', flush=True)
    fa.flash_attention_910a(q, k, v)
    torch.npu.synchronize()

    for _ in range(CALLS):
        out = fa.flash_attention_910a(q, k, v)
    torch.npu.synchronize()
    print('done calls', CALLS, 'finite', bool(torch.isfinite(out).all().item()), flush=True)


if __name__ == '__main__':
    main()
