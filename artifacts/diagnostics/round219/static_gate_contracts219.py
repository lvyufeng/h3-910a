#!/usr/bin/env python3
"""Static fail-closed source and token-coverage contracts for gate-only tests."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
stage=ast.parse((HERE/'gate_stage219.py').read_text())
cls=next(n for n in stage.body if isinstance(n,ast.ClassDef) and n.name=='GateStage219')
attention=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_attention')
returns=[n for n in ast.walk(attention) if isinstance(n,ast.Return)]
assert returns and all(isinstance(n.value,ast.Name) and n.value.id=='reference' for n in returns)
assert not any(n.name=='enable_measurement' for n in cls.body if isinstance(n,ast.FunctionDef))
trim=HERE.parent/'round218'/'fa218trim.py'
assert hashlib.sha256(trim.read_bytes()).hexdigest()=='52365cbf64180e9766b4dfc4c3a7bff5f311b6b8f8af7fc08b340a2bf8487b73'
for source_n,pad,br,bc,kernel_n in [(41792,23,128,304,43776),(2048,256,128,256,2048),(2048,257,128,256,2048)]:
    tr=(source_n+br-1)//br;tc=(source_n-pad+bc-1)//bc
    assert tr*br>=source_n
    assert tc*bc>=source_n-pad
    assert (tr-1)*br<source_n
    assert (tc-1)*bc<source_n-pad
    assert tr*br<=kernel_n and tc*bc<=kernel_n
record={'gate_function_returns_reference_only':True,
        'no_enable_candidate_measurement_method':True,
        'candidate_source_identical_to_R218':True,
        'all_real_query_and_valid_key_tiles_kept':True,
        'not_a_runtime_precision_proof':True}
(HERE/'static_gate_contracts219.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
