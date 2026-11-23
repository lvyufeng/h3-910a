#!/usr/bin/env python3
"""R212: is the 910A FA kernel bit-reproducible? (R211b failed on layout: the interface wants B,H,N,D)"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

OUT = Path('/tmp/h3_oprep212/fa_repeat212.json')


def word_hash(t):
    return hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:16]


def main():
    import torch, torch_npu  # noqa: F401
    torch.npu.set_device(0)
    sys.path.insert(0, '/tmp/fa_pkg')
    spec = importlib.util.spec_from_file_location('fa910a',
                                                  '/tmp/h3_perf12601/native_sources/fa910a.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = []
    # layout is (B, H, N, D); the real per-rank shape from R159 is H=7 heads, N=41792, D=128
    for tag, b, h, n, d, reps in (('small', 1, 7, 4096, 128, 6), ('real', 1, 7, 41792, 128, 4)):
        torch.manual_seed(11)
        q = torch.randn(b, h, n, d, dtype=torch.float16, device='npu:0')
        k = torch.randn(b, h, n, d, dtype=torch.float16, device='npu:0')
        v = torch.randn(b, h, n, d, dtype=torch.float16, device='npu:0')
        try:
            for _ in range(2):
                out = mod.flash_attention_910a(q, k, v)
            torch.npu.synchronize()
            hashes = [word_hash(mod.flash_attention_910a(q, k, v)) for _ in range(reps)]
            uniq = len(set(hashes))
            row = {'op': f'fa_{tag}', 'shape': [b, h, n, d], 'repeats': reps,
                   'distinct_results': uniq, 'reproducible': uniq == 1, 'hashes': hashes[:3]}
        except Exception as exc:  # noqa: BLE001
            row = {'op': f'fa_{tag}', 'shape': [b, h, n, d], 'error': repr(exc)[:300]}
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = {'scope': 'R212: 910A FA kernel bit-repeat with the correct B,H,N,D layout',
           'rows': rows,
           'all_reproducible': all(r.get('reproducible', False) for r in rows if 'error' not in r)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print('VERDICT', 'reproducible' if out['all_reproducible'] else 'check rows', flush=True)


if __name__ == '__main__':
    main()
