#!/usr/bin/env python3
"""Generate gate-only evidence replay: model forwards always return original FA outputs."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE.parent/'round218'/'benchmark_source218.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='0f2bf37c646257bb3ae015ac4468e77a2de94d24106e1be429b5377e8f392648'
text=raw.decode()
def rep(old,new):
    global text
    assert text.count(old)==1,(old[:100],text.count(old))
    text=text.replace(old,new,1)
rep("ROOT = Path('/tmp/h3_budget149')","ROOT = Path('/tmp/h3_gate219')")
rep("assert set(audit_modes) <= {'strict','deferred','device','profile'}","assert set(audit_modes) <= {'strict','gate'}")
rep("            event_profiler = None\n            active_case = None",
    "            event_profiler = None\n            from gate_stage219 import GateStage219\n            gate_stage = GateStage219()\n            active_case = None")
rep("                    audit.mode = 'strict' if audit_mode in ('strict','profile') else 'deferred'",
    "                    audit.mode = 'strict'")
rep("                if audit_mode == 'profile' and event_profiler is None:",
    "                if audit_mode == 'gate':\n"
    "                    gate_stage.begin_gate(step)\n"
    "                    try:\n"
    "                        audit.begin()\n"
    "                        gate_video, gate_audio = model(**fk)\n"
    "                        gate_safety = audit.end()\n"
    "                        torch.npu.synchronize()\n"
    "                        gate = gate_stage.end_gate()\n"
    "                        del gate_video, gate_audio\n"
    "                    finally:\n"
    "                        gate_stage.restore()\n"
    "                    failed = torch.tensor([int(not gate['passed'])], dtype=torch.int32, device=device)\n"
    "                    torch.distributed.all_reduce(failed,op=torch.distributed.ReduceOp.MAX)\n"
    "                    gate['all_rank_passed'] = int(failed.cpu()) == 0\n"
    "                    gate_stage.failed = gate_stage.failed or not gate['all_rank_passed']\n"
    "                    gate['strict_checks'] = gate_safety\n"
    "                    report.setdefault('gate_rows219', []).append(gate)\n"
    "                    publish()\n"
    "                if audit_mode == 'profile' and event_profiler is None:")
rep("        report['completed'] = True",
    "        gate_stage.restore()\n        report['gate_stage219'] = gate_stage.report()\n        gate_stage.close()\n        report['completed'] = True")
rep("        if torch.distributed.is_initialized():",
    "        if 'gate_stage' in locals():\n            gate_stage.close()\n        if torch.distributed.is_initialized():")
rep("        'checks': [],", "        'checks': [],\n"
    "        'gate_scope219': 'step0/48 gate-only isolated DiT replay; no candidate result used by model',\n"
    "        'live_encoder_executed219': False,\n"
    "        'candidate_adopted219': False,\n"
    "        'no_candidate_timing219': True,")
rep("    def stats(x):\n        cpu = x.detach().cpu()",
    "    def stats(x):\n        cpu = x.detach().cpu().contiguous()")
rep("return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),",
    "return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),\n"
    "                'raw_sha256': hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),")
ast.parse(text)
(HERE/'benchmark_gate219.py').write_text(text)
print('Generated gate-only replay',hashlib.sha256(text.encode()).hexdigest())
