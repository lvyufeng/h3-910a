#!/usr/bin/env python3
"""Patch only measurement hooks/metadata in the pinned existing replay script."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=HERE/'benchmark_source218.py'
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='0f2bf37c646257bb3ae015ac4468e77a2de94d24106e1be429b5377e8f392648'
text=raw.decode()
def replace(old,new):
    global text
    assert text.count(old)==1,(old[:80],text.count(old))
    text=text.replace(old,new,1)
replace("ROOT = Path('/tmp/h3_budget149')", "ROOT = Path('/tmp/h3_path218')")
replace("                    from actual_attention_profile126 import ActualAttentionProfile126\n                    event_profiler = ActualAttentionProfile126(model)",
        "                    from path_profile218 import PathProfile218\n                    event_profiler = PathProfile218(model, audit, native_hooks)")
replace("        'checks': [],", "        'checks': [],\n        'measurement_scope218': 'same-process eight-rank DiT replay, NOT full pipeline or 49-step acceptance',\n        'live_encoder_executed218': False,\n        'candidate_installed218': False,")
# Hash raw values outside all timing; existing stats already moves to host.
replace("    def stats(x):\n        cpu = x.detach().cpu()\n        return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),",
        "    def stats(x):\n        cpu = x.detach().cpu().contiguous()\n        return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),\n                'raw_sha256': hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),")
ast.parse(text)
(HERE/'benchmark_path218.py').write_text(text)
print('generated path replay',hashlib.sha256(text.encode()).hexdigest())
