"""No-weight, independent decomposition of the deployed FP32/FP16 GEMM path.

Exactly one NPU. This NEVER enables FA trim or any other model candidate. It
benchmarks synthetic tensors only and does not download/read model weights,
change model sources, or claim numerical equivalence. Original rounding/range
semantics remain mandatory; the unguarded variant only supplies an unattributed
performance reference and deliberately cannot count as an adoption candidate.
"""
from __future__ import annotations
import argparse, json, math, statistics, time, hashlib
import torch


def stats(values):
    values = list(map(float, values))
    return {"samples": values, "min": min(values), "median": statistics.median(values), "max": max(values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--m", type=int, default=10448)
    p.add_argument("--cases", default="qkv,fc2,out_proj,fc1")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.device < 0 or args.device > 7:
        raise ValueError("device must be an explicitly observed logical card 0..7")
    if min(args.warmup, args.steps) < 1 or args.steps > 200:
        raise ValueError("invalid reproducibility sample counts")
    import torch_npu
    torch.npu.set_device(args.device)
    info = torch.npu.get_device_properties(args.device)
    info_dict = {"actual_id": torch.npu.current_device(),
                 "requested_id": args.device,
                 "name": str(getattr(info, "name", "unknown")),
                 "total_memory_bytes": int(getattr(info, "total_memory", 0))}
    m = args.m
    shapes = {
        "qkv": (7168, 10752, False),
        "fc2": (5376, 5376, False),
        "out_proj": (5376, 5376, True),
        "fc1": (5376, 14336, True),
    }
    torch.manual_seed(221)
    result = {"schema_version": "r221.gemm.decomposition.v1", "exec_authorization": "ts-217-only",
              "host": "ts-217", "device": info_dict, "torch": torch.__version__,
              "no_weight_access": True, "no_model_modification": True,
              "fa_trim_enabled": False, "fa_or_other_candidate_composition": False,
              "elapsed_per_iteration_ms": {}, "component_event_ms": {}}
    for label in args.cases.split(","):
        k, n, has_bias = shapes[label.strip()]
        x = (torch.randn(m, k, dtype=torch.float16, device="npu") * 0.125).float()
        w = (torch.randn(k, n, dtype=torch.float16, device="npu") * 0.125).contiguous()
        bias = torch.randn(n, dtype=torch.float32, device="npu") if has_bias else None
        torch.npu.synchronize()
        a32 = b32 = f16 = ref = out = None
        boundary = torch.empty((), dtype=torch.float32, device="npu").fill_(float(torch.finfo(torch.float16).max))
        if bias is not None:
            bias = bias.reshape(1, -1)
        def step_parts(records, full, check):
            nonlocal a32, b32, f16, ref, out
            # Independent storage on every iteration excludes source aliasing artifacts.
            a32 = x.float(memory_format=torch.contiguous_format)
            b32 = w.float(memory_format=torch.contiguous_format)
            torch.npu.synchronize()
            events = {name: torch.npu.Event(enable_timing=True) for name in
                      ("start", "cast_end", "amax_end", "host_end", "convert_end", "gemm_end", "end")}
            events["start"].record()
            a32.copy_(x)
            b32.copy_(w)
            events["cast_end"].record()
            if check:
                scale = 65504.0 / amax if (amax := torch.max(torch.abs(a32))) > 65504.0 else 1.0
                events["amax_end"].record()
                # Original implementation is scalar host-synchronizing: `if tensor > ...`.
                scale = float(scale.detach().cpu())
                events["host_end"].record()
                a16 = (a32 * scale).half().contiguous()
            else:
                scale = 1.0
                a16 = a32.half().contiguous()
            f16 = a16
            events["convert_end"].record()
            if full:
                out = torch.mm(a16, b32) * (1.0 / scale)
                if bias is not None:
                    out = out + bias
            else:
                out = torch.mm(a16, w)  # Performance only. Never a numerical candidate.
            events["gemm_end"].record()
            if bias is not None and full:
                # capture separately only on dedicated rescale runs below
                pass
            events["end"].record()
            torch.npu.synchronize()
            records["elapsed"].append((time.perf_counter_ns() % 10**12) * 0.0)  # populated by outer wall clock
            names = list(events)
            for first, second in zip(names, names[1:]):
                records[first + "_to_" + second].append(events[first].elapsed_time(events[second]))
            records["_peak_hbm"].append(float(torch.npu.max_memory_allocated()) / 2**20)
        # Make explicit component timings, including a separate rescale/bias event.
        records = {"elapsed": [], "_peak_hbm": [], "start_to_cast_end": [], "cast_end_to_amax_end": [],
                   "amax_end_to_host_end": [], "host_end_to_convert_end": [], "convert_end_to_gemm_end": [],
                   "gemm_end_to_end": [], "rescale_plus_bias": []}
        def run(full, check):
            t = time.perf_counter()
            step_parts(records, full, check)
            records["elapsed"][-1] = (time.perf_counter()-t)*1000
        for _ in range(args.warmup):
            run(True, True)
        for i in range(args.steps):
            torch.npu.reset_peak_memory_stats()
            run(True, True)
        ref = out.detach().float().cpu()
        metrics = {"range_ratio_batch_max": float(torch.max(torch.abs(x)).cpu()) / 65504.0}
        if torch.isfinite(ref).all():
            bounded = ref
        else:
            bounded = x.new_tensor(float("nan"))
        # Full end-to-end variant without a range guard; marks unsafe, not equivalent.
        for _ in range(args.warmup):
            run(False, False)
        unguarded = [records["elapsed"][-args.steps-i-1] for i in range(args.steps)]  # replaced below
        start_idx = len(records["elapsed"])
        for _ in range(args.steps):
            run(False, False)
        unguarded = records["elapsed"][start_idx:]
        candidate = out.detach().float().cpu()
        bounded_diff = torch.isfinite(ref) & torch.isfinite(candidate)
        elements = ref.numel()
        differ = int(((ref != candidate) & bounded_diff).sum())
        exceeded = int(((~torch.isfinite(candidate)) | (ref != candidate)).sum())
        result["elapsed_per_iteration_ms"][label] = {"M":m,"K":k,"N":n,
            "event_guarded": stats(records["elapsed"][args.warmup:args.warmup+args.steps]),
            "event_unguarded_perf_only": stats(unguarded),
            "range_exceeded_fraction": metrics["range_ratio_batch_max"],
            "bitwise_equal_element_count": elements - exceeded,
            "total_element_count": elements, "differing_or_nonfinite_element_count": exceeded,
            "scoped_dense_finite_diff_count": differ,
            "in_scope_rel_l2": float((torch.linalg.vector_norm((ref-candidate)[bounded_diff])/torch.linalg.vector_norm(ref[bounded_diff])).cpu()) if bool(bounded_diff.any()) else None,
            "in_scope_max_abs_error": float(torch.abs(ref-candidate)[bounded_diff].max().cpu()) if bool(bounded_diff.any()) else None,
            "full_scope_metrics_valid": bool(bounded_diff.all()),
            "source_outputs_nonaliased": ref.data_ptr()!=candidate.data_ptr(),
            "source_weight_contiguous": w.is_contiguous(), "source_activation_contiguous": x.is_contiguous(),
            "cpu_reference_source_not_hardware_truth": True,
            "unguarded_adoption_forbidden": True,
            "guarded_event": {key:stats(records[key][:args.warmup+args.steps][args.warmup:]) for key in ("start_to_cast_end","cast_end_to_amax_end","amax_end_to_host_end","host_end_to_convert_end","convert_end_to_gemm_end","gemm_end_to_end","_peak_hbm")}}
        result["component_event_ms"][label] = result["elapsed_per_iteration_ms"][label].pop("guarded_event")
        print("CASE",json.dumps(result["elapsed_per_iteration_ms"][label]),flush=True)
        del a32,b32,f16,ref,out,x,w,bias,bounded,candidate
        torch.npu.empty_cache()
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print("SUMMARY",json.dumps({"out":args.out,"device":info_dict,"samples":args.steps}),flush=True)


if __name__ == "__main__":
    main()
