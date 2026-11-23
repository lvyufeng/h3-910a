#!/usr/bin/env python3
"""R211b: is the 910A FA kernel itself bit-reproducible? (the all-operator probe failed on shape)"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

OUT = Path('/tmp/h3_oprep211/fa_repeat211.json')


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
    # the real shapes from R159: input_N=41792, kernel_N=43776 (both multiples of 16)
    for tag, heads, seq, hd in (('real_7x128', 7, 41792, 128), ('small_7x128', 7, 4096, 128)):
        torch.manual_seed(11)
        q = torch.randn(1, seq, heads, hd, dtype=torch.float16, device='npu:0')
        k = torch.randn(1, seq, heads, hd, dtype=torch.float16, device='npu:0')
        v = torch.randn(1, seq, heads, hd, dtype=torch.float16, device='npu:0')
        try:
            out = mod.flash_attention_910a(q, k, v)
            for _ in range(2):
                out = mod.flash_attention_910a(q, k, v)
            hashes = [word_hash(mod.flash_attention_910a(q, k, v)) for _ in range(6)]
            uniq = len(set(hashes))
            row = {'op': f'fa_{tag}', 'seq': seq, 'heads': heads, 'distinct_results': uniq,
                   'reproducible': uniq == 1, 'hashes': hashes[:3]}
        except Exception as exc:  # noqa: BLE001
            row = {'op': f'fa_{tag}', 'seq': seq, 'error': repr(exc)[:300]}
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = {'scope': 'R211b: 910A FA kernel bit-repeat', 'rows': rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
