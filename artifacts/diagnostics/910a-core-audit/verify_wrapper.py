#!/usr/bin/env python3
"""Check the actual runtime-budgeted wrapper against its unchanged 32-core backup."""
import importlib.util
from importlib.machinery import SourceFileLoader
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu

BASE = Path("/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py")

def load(path, name):
    s = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

torch.npu.set_device(0)
old = load(str(BASE) + ".pre_corebudget_20261115.bak", "old_fa")
new = load(BASE, "new_fa")
q, k, v = [torch.randn((1, 7, 63232, 128), dtype=torch.float16, device="npu") * .05 for _ in range(3)]
result = {"runtime_core_budget": new._runtime_core_budget()}
for label, mod in [("old32", old), ("new30", new)]:
    obj, name, bd = mod._build(1, 7, 63232, 128)
    result[label] = {"object": obj, "kernel": name, "block_dim": bd}
    for _ in range(2):
        out = mod.flash_attention_910a(q, k, v)
    torch.npu.synchronize()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        for __ in range(3):
            out = mod.flash_attention_910a(q, k, v)
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) / 3)
    result[label]["groups_ms"] = [t * 1000 for t in times]
    result[label]["median_ms"] = statistics.median(times) * 1000
    if label == "old32":
        reference = out.clone()
    else:
        result["same_bits"] = bool(torch.equal(out, reference))
        result["max_abs_diff"] = (out.float() - reference.float()).abs().max().item()
assert result["new30"]["block_dim"] == 30
assert result["same_bits"], result
Path("/tmp/h3_wrapper_corebudget_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
