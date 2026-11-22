#!/usr/bin/env python3
"""CPU-only observer smoke for mutation snapshots/returns/exception restoration."""
from pathlib import Path
import json
import torch
from projection_observer130 import ProjectionObserver130

ROOT=Path('/tmp/h3_time13001')
path=ROOT/'observer_unit130.json';assert not path.exists()
torch.set_num_threads(1);old_mm=torch.mm;old_add=torch.Tensor.add_
a=torch.tensor([[1.,2.],[3.,4.]]);b=torch.eye(2);bias=torch.ones(2,2)
observer=ProjectionObserver130(torch)
with observer.observe():
    output=torch.mm(a,b);identity=output
    result=output.add_(bias,alpha=0.5)
    assert result is identity
assert torch.mm is old_mm and torch.Tensor.add_ is old_add
assert torch.equal(observer.rows[0]['output'],a)
assert torch.equal(observer.rows[1]['input_a'],a)
assert torch.equal(observer.rows[1]['output'],a+0.5)
try:
    with ProjectionObserver130(torch).observe():raise RuntimeError('Injected observer failure')
except RuntimeError:pass
assert torch.mm is old_mm and torch.Tensor.add_ is old_add
result={'completed':True,'model_or_NPU_inference':False,'original_mutable_return_identity_preserved':True,
    'pre_add_snapshots_not_changed_by_later_inplace_add':True,'exception_restores_function_objects':True}
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
