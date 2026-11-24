#!/usr/bin/env python3
"""Create an isolated, single-candidate clean->leafgate->trim->restored replay."""
import ast
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'benchmark_source218.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='0f2bf37c646257bb3ae015ac4468e77a2de94d24106e1be429b5377e8f392648'
text=raw.decode()
def rep(old,new):
 global text
 assert text.count(old)==1,(old[:80],text.count(old))
 text=text.replace(old,new,1)
rep("ROOT = Path('/tmp/h3_budget149')","ROOT = Path('/tmp/h3_trim_model218')")
rep("assert set(audit_modes) <= {'strict','deferred','device','profile'}","assert set(audit_modes) <= {'strict','trim'}")
rep("            event_profiler = None\n            active_case = None","            event_profiler = None\n            from trim_stage218 import TrimStage218\n            trim_stage = TrimStage218()\n            active_case = None")
rep("                    audit.mode = 'strict' if audit_mode in ('strict','profile') else 'deferred'","                    trim_stage.restore()\n                    audit.mode = 'strict'\n                    if audit_mode == 'trim':\n                        trim_stage.begin_gate()\n                        audit.begin()\n                        gate_video, gate_audio = model(**fk)\n                        gate_safety = audit.end()\n                        torch.npu.synchronize()\n                        gate = trim_stage.end_gate()\n                        del gate_video, gate_audio\n                        failed = torch.tensor([int(not gate['passed'])], dtype=torch.int32, device=device)\n                        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)\n                        gate['all_rank_passed'] = int(failed.cpu()) == 0\n                        gate['strict_checks'] = gate_safety\n                        report['trim_gate218'] = gate\n                        publish()\n                        if not gate['all_rank_passed']:\n                            trim_stage.restore()\n                            blocked_cases.add(case_index)\n                            active_case = case_index\n                            continue\n                        trim_stage.enable_measurement()")
# Baseline runners keep original profiling code unreachable (audit choices strict/trim only).
rep("        report['completed'] = True","        trim_stage.restore()\n        report['trim_stage218'] = trim_stage.report()\n        trim_stage.close()\n        report['completed'] = True")
rep("        'checks': [],","        'checks': [],\n        'scope218': 'single FA trim candidate, eight-rank isolated DiT step25 replay, not full pipeline',\n        'live_encoder_executed218': False,\n        'combination_with_other_candidates218': False,")
rep("    def stats(x):\n        cpu = x.detach().cpu()","    def stats(x):\n        cpu = x.detach().cpu().contiguous()")
rep("return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),","return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),\n                'raw_sha256': hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),")
ast.parse(text)
(HERE/'benchmark_trim_model218.py').write_text(text)
print('generated independent model trim candidate',hashlib.sha256(text.encode()).hexdigest())
