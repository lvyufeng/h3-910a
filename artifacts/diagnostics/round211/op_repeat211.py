#!/usr/bin/env python3
"""R211: which single operator is not bit-reproducible on this 910B?

R210 showed HCCL all-reduce is bit-reproducible, so the native run-to-run variation cannot come from
the collectives. This probes the operators the model actually uses (matmul, FA, norms, residual add,
softmax) on identical inputs, single card, single stream, repeated, bit-compared.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

OUT = Path('/tmp/h3_oprep211/op_repeat211.json')
REPEATS = 10


def word_hash(t) -> str:
    return hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:16]


def probe(name, fn, repeats=REPEATS):
    """Run fn() repeatedly and report how many distinct bit patterns appear."""
    hashes = []
    for _ in range(repeats):
        out = fn()
    # do it again collecting hashes (the warmup pass above lets kernels settle)
    hashes = [word_hash(fn()) for _ in range(repeats)]
    uniq = len(set(hashes))
    row = {'op': name, 'repeats': repeats, 'distinct_results': uniq,
           'reproducible': uniq == 1, 'hashes': hashes[:3]}
    print(json.dumps(row), flush=True)
    return row


def main() -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    dev = 'npu:0'
    rows = []

    def fixed(shape, dtype=torch.float16, seed=0):
        torch.manual_seed(seed)
        return torch.randn(*shape, dtype=dtype, device=dev)

    # --- 1. matmul at the model's real GEMM shapes (fp16 in, fp32 accumulate as the model does)
    for tag, m, n, k in (('qkv', 41792, 5120, 5120), ('mlp_fc1', 41792, 25600, 5120),
                         ('mlp_fc2', 41792, 5120, 25600), ('out_proj', 41792, 5120, 5120)):
        a = fixed((m, k), seed=1)
        b = fixed((k, n), seed=2)
        rows.append(probe(f'matmul_{tag}', lambda a=a, b=b: torch.matmul(a, b)))

    # --- 2. reductions and normalisations
    x = fixed((41792, 5120), seed=3)
    rows.append(probe('sum_dim_-1', lambda x=x: x.float().sum(dim=-1)))
    rows.append(probe('max_dim_-1', lambda x=x: x.float().max(dim=-1).values))
    w = fixed((5120,), seed=4)
    rows.append(probe('rmsnorm_fp32', lambda x=x, w=w: (
        x.float() / torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + 1e-6) * w.float()
    ).half()))

    # --- 3. residual add / elementwise
    y = fixed((41792, 5120), seed=5)
    rows.append(probe('residual_add_fp16', lambda x=x, y=y: x + y))

    # --- 4. softmax (the FA inner op)
    s = fixed((128, 304), seed=6).float()
    rows.append(probe('softmax_fp32', lambda s=s: torch.softmax(s, dim=-1)))

    # --- 5. the 910A FA kernel itself, if importable
    try:
        sys.path.insert(0, '/tmp/fa_pkg')
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'fa910a', '/tmp/h3_perf12601/native_sources/fa910a.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        head_dim, heads, seq = 128, 7, 43776
        q = fixed((1, seq, heads, head_dim), seed=7)
        k = fixed((1, seq, heads, head_dim), seed=8)
        v = fixed((1, seq, heads, head_dim), seed=9)
        rows.append(probe('fa_910a_kernel', lambda q=q, k=k, v=v: mod.flash_attention_910a(q, k, v),
                          repeats=4))
    except Exception as exc:  # noqa: BLE001
        rows.append({'op': 'fa_910a_kernel', 'error': repr(exc)[:300]})
        print(json.dumps(rows[-1]), flush=True)

    out = {'scope': 'R211: single-operator bit-repeat probe on 910B, single card single stream',
           'device': 'npu:0', 'repeats': REPEATS, 'rows': rows,
           'nondeterministic_ops': [r['op'] for r in rows if r.get('reproducible') is False],
           'all_reproducible': all(r.get('reproducible', True) for r in rows)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print('VERDICT', 'all_reproducible' if out['all_reproducible']
          else 'non-det: ' + ','.join(out['nondeterministic_ops']), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
