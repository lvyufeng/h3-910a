#!/usr/bin/env python3
"""Validate the real new wrapper, including copies, clipping, reuse and numerics.

Runs on one otherwise-idle NPU. No production source changes. All numerical
references use FP32 full-key attention after matching FP16 input scaling.
"""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch_npu

ROOT = Path("/tmp/h3_round52_wrapper_validation01")
OUT = ROOT / os.environ.get("H3_VALIDATION_OUT", "result.json")
if OUT.exists():
    raise RuntimeError("Refuse to overwrite an earlier validation result")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = load(ROOT / "source/fa910a.py", "fa52_old")
new = load(ROOT / "candidate_fa910a.py", "fa52_new")
result = {"timing_runs": [], "numerical_cases": [], "production_changed": False}


def reference_metrics(out, q, k, v, scale, original_pad=0):
    n = q.shape[2]
    rows = torch.tensor([0,1,63,64,127,n//2,n-original_pad-2,n-original_pad-1])
    qs = (q.detach().cpu().float() * scale).half().float()[0,0,rows]
    end = n-original_pad
    kk = k.detach().cpu().float()[0,0,:end]
    vv = v.detach().cpu().float()[0,0,:end]
    ref = torch.softmax(qs @ kk.T,dim=-1) @ vv
    error = out.float()[0,0,rows] - ref
    return {"sampled_max_abs_error": error.abs().max().item(),
            "sampled_relative_rms": (error.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item(),
            "finite": bool(torch.isfinite(out).all())}


try:
    # CPU check of shared target-size buffers, independent of kernel execution.
    pointers = None
    for n in (48,32,48):
        xs = [torch.full((1,2,n,16), float(i+1),dtype=torch.float16) for i in range(3)]
        pads = new._pad_qkv(*xs,64)
        ptrs = [x.data_ptr() for x in pads]
        if pointers is not None:
            assert ptrs == pointers
        pointers = ptrs
        for src,dst in zip(xs,pads):
            assert torch.equal(src,dst[:,:,:n,:]) and not bool(dst[:,:,n:,:].count_nonzero())
    result["cpu_pad_buffer_reuse_and_tail_zero"] = True

    torch.npu.set_device(0)
    result["core_budget"] = new._runtime_core_budget()
    obj,symbol,bd = new._build(1,7,43776,128,key_seq_limit=41792)
    assert bd == 30
    meta = json.loads(Path(obj).with_suffix(".json").read_text())
    assert meta["fa910a_key_seq_limit"] == 41792 and meta["fa910a_kernel_version"] == 2
    result["masked_kernel"] = {"object":obj,"symbol":symbol,"block_dim":bd,"key_seq_limit":41792}
    torch.manual_seed(52)
    scale = 128 ** -.5
    for amplitude, original_pad in ((.05,0),(.5,0),(.1,64)):
        q,k,v = [torch.randn(1,7,41792,128,dtype=torch.float16,device="npu") * amplitude for _ in range(3)]
        if original_pad:
            # Do not exercise the driver's nonzero-offset zero_ warning in the
            # harness when preparing the caller's known-zero original padding.
            kp=k.cpu(); vp=v.cpu()
            kp[:,:,-original_pad:,:].zero_(); vp[:,:,-original_pad:,:].zero_()
            k=kp.to("npu"); v=vp.to("npu")
        if amplitude == .05:
            # Exercise the old missing-isrc bug: first unscaled, then scaled.
            first = new.flash_attention_910a(q,k,v,scale=1.)
            torch.npu.synchronize()
            result["initial_scale_one"] = reference_metrics(first.cpu(),q,k,v,1.)
            assert result["initial_scale_one"]["sampled_relative_rms"] < .005
        outputs = {}
        funcs = {"old_small_tiles":lambda:old.flash_attention_910a(q,k,v,scale=scale,n_pad=original_pad),
                 "new_internal_masked_padding":lambda:new.flash_attention_910a(q,k,v,scale=scale,n_pad=original_pad)}
        for order in (("old_small_tiles","new_internal_masked_padding"),("new_internal_masked_padding","old_small_tiles")):
            for label in order:
                for _ in range(2):
                    y = funcs[label]()
                torch.npu.synchronize()
                if amplitude == .05:
                    groups=[]
                    for _ in range(3):
                        start=time.perf_counter()
                        for __ in range(3):
                            y=funcs[label]()
                        torch.npu.synchronize()
                        groups.append((time.perf_counter()-start)*1000/3)
                    item={"variant":label,"groups_ms":groups,"median_ms":statistics.median(groups)}
                    result["timing_runs"].append(item)
                    print(json.dumps(item),flush=True)
                assert y.shape == q.shape and y.dtype == q.dtype
                outputs[label]=y.clone().cpu()
        difference=outputs["new_internal_masked_padding"].float()-outputs["old_small_tiles"].float()
        case={"amplitude":amplitude,"original_pad":original_pad,
              "pair_max_abs_diff":difference.abs().max().item(),
              "pair_relative_rms":(difference.square().mean().sqrt()/outputs["old_small_tiles"].float().square().mean().sqrt().clamp_min(1e-12)).item(),
              "reference":{label:reference_metrics(value,q,k,v,scale,original_pad) for label,value in outputs.items()}}
        result["numerical_cases"].append(case)
        print(json.dumps(case),flush=True)
        assert case["pair_relative_rms"] < .005
        assert all(x["finite"] and x["sampled_relative_rms"] < .005 for x in case["reference"].values())

    # Same target allocation, different key-clipping boundaries, then shrink again.
    for n in (41920,41792):
        q,k,v=[torch.randn(1,7,n,128,dtype=torch.float16,device="npu")*.05 for _ in range(3)]
        y=new.flash_attention_910a(q,k,v,scale=scale)
        torch.npu.synchronize()
        case=reference_metrics(y.cpu(),q,k,v,scale)
        assert case["finite"] and case["sampled_relative_rms"] < .005
        pads=new._PAD_BUFFERS[(1,7,43776,128,q.device)]
        assert all(not bool(p[:,:,n:,:].detach().cpu().count_nonzero()) for p in pads)
        result.setdefault("length_change_cases",[]).append({"input_n":n,**case,"zero_tail":True})

    # All valid scores are strongly negative. Extra zero-key LSE subtraction is
    # unsafe here; -inf score masking must keep the uniform value output intact.
    q=torch.full((1,7,41792,128),2.,dtype=torch.float16,device="npu")
    k=torch.full_like(q,-2.); v=torch.ones_like(q)
    safe=new.flash_attention_910a(q,k,v,scale=scale)
    torch.npu.synchronize()
    safe_cpu=safe.cpu()
    max_error=(safe_cpu.float()-1.).abs().max().item()
    assert bool(torch.isfinite(safe_cpu).all()) and max_error < .002
    pq,pk,pv=new._pad_qkv(q,k,v,43776)
    unsafe=old.flash_attention_910a(pq,pk,pv,scale=scale,n_pad=1984)[:,:,:41792,:]
    torch.npu.synchronize()
    unsafe_cpu=unsafe.cpu()
    unsafe_finite=bool(torch.isfinite(unsafe_cpu).all())
    unsafe_error=(unsafe_cpu.float()-1.).abs().max().item() if unsafe_finite else None
    result["negative_score_case"]={"valid_score":-45.25,"masked_finite":True,"masked_max_error_to_uniform_one":max_error,
                                   "zero_key_lse_finite":unsafe_finite,"zero_key_lse_max_error_to_uniform_one":unsafe_error}
    assert not unsafe_finite or unsafe_error > .5
    print(json.dumps(result["negative_score_case"]),flush=True)
    # Finite-minimum rowmax tie: a finite sentinel alone would leave probability
    # mass on the padding here. Explicit zeroing after exp must remove it.
    q.zero_(); k.zero_()
    q[:,:,:,0].fill_(1.)
    k[:,:,:,0].fill_(-65504.)
    minimum=new.flash_attention_910a(q,k,v,scale=1.)
    torch.npu.synchronize()
    minimum_cpu=minimum.cpu()
    minimum_error=(minimum_cpu.float()-1.).abs().max().item()
    assert bool(torch.isfinite(minimum_cpu).all()) and minimum_error < .002
    result["finite_minimum_score_case"]={"valid_score":-65504.,"finite":True,
                                          "max_error_to_uniform_one":minimum_error}
    print(json.dumps(result["finite_minimum_score_case"]),flush=True)
except BaseException as exc:
    result["failure"]={"type":type(exc).__name__,"message":str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("Evidence:",OUT,flush=True)
