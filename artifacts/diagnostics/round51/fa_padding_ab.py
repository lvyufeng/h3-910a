#!/usr/bin/env python3
"""Same valid attention input: exact-divisor tiles vs padded large tiles.

The real FL2VA shape N=41792 forces Br/Bc=64 in the current divisor heuristic.
Compare its existing wrapper to N=43776 (=18*lcm(128,304)) with known-zero K/V
rows and the existing LSE padding correction. No production source changes.
"""
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
import torch
import torch_npu

OUT = Path("/tmp/h3_round51_fa_padding_ab01.json")
if OUT.exists():
    raise RuntimeError("Refuse to overwrite earlier A/B evidence")
torch.npu.set_device(0)
spec = importlib.util.spec_from_file_location("fa51", "/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)
b, h, n, d = 1, 7, 41792, 128
pn = math.ceil(n / math.lcm(128, 304)) * math.lcm(128, 304)
assert pn == 43776
scale = d ** -.5
result = {"valid_shape": [b, h, n, d], "padded_n": pn, "extra_zero_keys": pn-n,
          "core_budget": fa._runtime_core_budget(), "cases": [], "runs": []}
for size in (n, pn):
    obj, symbol, bd = fa._build(b,h,size,d)
    assert bd == 30
    result[str(size)] = {"object": obj, "block_dim": bd,
                         "br": fa.pick_tile(size,128), "bc": fa.pick_tile(size,304)}
torch.manual_seed(51)
for amplitude in (.05, .5):
    q, k, v = [torch.randn(b,h,n,d,dtype=torch.float16,device="npu") * amplitude for _ in range(3)]
    qp, kp, vp = [torch.zeros(b,h,pn,d,dtype=torch.float16,device="npu") for _ in range(3)]
    for padded, original in zip((qp,kp,vp),(q,k,v)):
        padded[:,:,:n,:].copy_(original)
    funcs = {"small_tiles": lambda: fa.flash_attention_910a(q,k,v,scale=scale),
             "padded_large_tiles": lambda: fa.flash_attention_910a(qp,kp,vp,scale=scale,n_pad=pn-n)[:,:,:n,:]}
    outputs = {}
    for order in (("small_tiles","padded_large_tiles"),("padded_large_tiles","small_tiles")):
        for label in order:
            for _ in range(2):
                out = funcs[label]()
            torch.npu.synchronize()
            if amplitude == .05:
                groups = []
                for _ in range(3):
                    start = time.perf_counter()
                    for __ in range(3):
                        out = funcs[label]()
                    torch.npu.synchronize()
                    groups.append((time.perf_counter()-start)*1000/3)
                row = {"variant": label,"groups_ms": groups,"median_ms": statistics.median(groups)}
                result["runs"].append(row)
                print(json.dumps(row),flush=True)
            outputs[label] = out.clone().cpu()
    difference = outputs["padded_large_tiles"].float() - outputs["small_tiles"].float()
    base_rms = outputs["small_tiles"].float().square().mean().sqrt().clamp_min(1e-12)
    case = {"amplitude": amplitude,"pair_max_abs_diff": difference.abs().max().item(),
            "pair_relative_rms": (difference.square().mean().sqrt()/base_rms).item(),
            "pair_bit_exact": bool(torch.equal(outputs["small_tiles"],outputs["padded_large_tiles"])),
            "reference": {}}
    rows = torch.tensor([0,1,63,64,127,n//2,n-2,n-1])
    qs = (q.cpu().float()*scale).half().float()[0,0,rows]
    kk, vv = k.cpu().float()[0,0], v.cpu().float()[0,0]
    reference = torch.softmax(qs @ kk.T,dim=-1) @ vv
    for label, output in outputs.items():
        sampled = output.float()[0,0,rows]
        err = sampled-reference
        case["reference"][label] = {"max_abs_error": err.abs().max().item(),
                                    "relative_rms": (err.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-12)).item(),
                                    "finite": bool(torch.isfinite(output).all())}
    result["cases"].append(case)
    print(json.dumps(case),flush=True)
    assert case["pair_relative_rms"] < .005, case
    assert all(x["relative_rms"] < .005 and x["finite"] for x in case["reference"].values()), case
result["production_changed"] = False
OUT.write_text(json.dumps(result,indent=2)+"\n")
print("Evidence:",OUT,flush=True)
