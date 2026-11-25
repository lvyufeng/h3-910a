#!/usr/bin/env python3
"""R221: exact-semantics decomposition of the two guarded DiT linears.

No weights are loaded, no model is imported, no source is modified. This replays
the op sequence of ``_MiniMaxH3HalfInputFP32OutputLinearMethod`` (deployed copy
``transformer.py`` sha256 5d94ba09..., lines 495-517) on synthetic tensors at the
true in-model shapes measured by R146:

  attn.out_proj : x FP16 (M,3584) -> guard -> mm(FP16 x FP16, FP32 out) -> M,5376
  mlp.fc2       : x FP32 (M,7168) -> guard -> mm(FP16 x FP16, FP32 out) -> M,5376

M = 10448 tokens/rank. Every component is separated by device events with a
single synchronize per sample, so no cross-stream drain is folded into a span.

Timing variants (none is adopted, none grants numerical approval):
  mm_only        bare mm, FP16 cast of x + FP32 output included (lower bound)
  amax_cast_mm   upcast -> abs/amax -> host sync -> cast -> mm, no /scale, no *scale
  full_guard     deployed sequence verbatim on a scale==1 input
  full_guard_big deployed sequence on a forced peak>32752 input (exponent>0)

A bitwise comparison of ``full_guard`` vs ``amax_cast_mm`` is recorded for the
scale==1 branch only. It is evidence about a hypothesis, not an approval.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

M = 10448
PEAK_FP16_TF = 310.0  # R146 comparison convention; an assumption, not a measurement

CASES = [
    # name, K, N, input dtype, R147 in-model median ms
    ("attn_out_proj_guarded", 3584, 5376, "float16", 18.306),
    ("mlp_fc2_guarded", 7168, 5376, "float32", 29.934),
]

EV_NAMES = ("start", "upcast", "amax", "sync", "div", "cast", "mm", "mul", "end")


def pct(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(values):
    values = [float(v) for v in values]
    if not values:
        return None
    return {"n": len(values), "min": min(values), "p10": pct(values, 0.10),
            "median": statistics.median(values), "p90": pct(values, 0.90),
            "max": max(values), "mean": statistics.fmean(values)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not 0 <= args.device <= 7:
        raise SystemExit("device must be an observed logical card 0..7")
    if not 0 <= args.warmup <= 50 or not 1 <= args.steps <= 200:
        raise SystemExit("invalid sample counts")

    import torch
    import torch_npu as _tn

    torch.npu.set_device(args.device)
    props = torch.npu.get_device_properties(args.device)
    result = {
        "schema_version": "r221.guarded_linear.decomposition.v1",
        "host": "ts-217",
        "authorized_device_scope": "ts-217 only; single card, TP2/USP4 untouched",
        "measurement_only": True,
        "no_model_weights_loaded": True,
        "no_model_import": True,
        "no_source_modification": True,
        "fa_trim_enabled": False,
        "candidate_installed": False,
        "numerical_approval_granted": False,
        "deployed_semantics": {
            "file": "transformer.py (h3_replay123)",
            "sha256": "5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b",
            "class": "_MiniMaxH3HalfInputFP32OutputLinearMethod",
            "lines": "495-517",
        },
        "device": {
            "requested_id": args.device,
            "actual_id": int(torch.npu.current_device()),
            "name": str(getattr(props, "name", "unknown")),
            "total_memory_bytes": int(getattr(props, "total_memory", 0)),
        },
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "torch_npu": getattr(_tn, "__version__", "unknown")},
        "m_tokens": M,
        "peak_assumed_FP16_TFLOPS": PEAK_FP16_TF,
        "r147_in_model_median_ms": {c[0]: c[4] for c in CASES},
        "cases": {},
    }

    for name, k, n, in_dtype, in_model_ms in CASES:
        dtype = torch.float16 if in_dtype == "float16" else torch.float32
        torch.manual_seed(221)
        x = (torch.randn(M, k, dtype=dtype, device="npu") * 0.05)
        w = (torch.randn(n, k, dtype=torch.float16, device="npu") * 0.05)
        w_t = w.t()  # non-contiguous view, exactly as layer.weight.t() in-model
        x_seen = x.float()
        alias = x_seen.data_ptr() == x.data_ptr() if dtype == torch.float32 else None
        del x_seen
        # Forced-large magnitude operand for the exponent>0 branch (finite, > 32752).
        x_big = (torch.rand(M, k, dtype=torch.float32, device="npu") * 50000.0).to(dtype)
        del w

        def full_guard_once(x_in, collect):
            e = {key: torch.npu.Event(enable_timing=True) for key in EV_NAMES}
            e["start"].record()
            product = x_in.float()
            e["upcast"].record()
            peak = float(product.abs().amax(dim=tuple(range(product.ndim))))
            e["amax"].record()
            if not math.isfinite(peak):
                raise FloatingPointError("nonfinite peak")
            exponent = max(0, math.ceil(math.log2(peak / 32752.0))) if peak > 32752.0 else 0
            scale = float(2 ** exponent)
            e["sync"].record()
            half_input = (product / scale).to(torch.float16)
            e["div"].record()
            e["cast"].record()
            out = torch.mm(half_input.reshape(-1, half_input.shape[-1]), w_t,
                           out_dtype=torch.float32)
            e["mm"].record()
            out = out * scale
            e["mul"].record()
            e["end"].record()
            torch.npu.synchronize()
            if collect is not None:
                for first, second in zip(EV_NAMES, EV_NAMES[1:]):
                    collect[first + "_to_" + second].append(e[first].elapsed_time(e[second]))
            return out, peak, scale

        def amax_cast_mm_once(collect):
            e = {key: torch.npu.Event(enable_timing=True) for key in EV_NAMES}
            e["start"].record()
            product = x.float()
            e["upcast"].record()
            peak = float(product.abs().amax(dim=tuple(range(product.ndim))))
            e["amax"].record()
            e["sync"].record()
            half_input = product.to(torch.float16)
            e["div"].record()
            e["cast"].record()
            out = torch.mm(half_input.reshape(-1, half_input.shape[-1]), w_t,
                           out_dtype=torch.float32)
            e["mm"].record()
            e["mul"].record()
            e["end"].record()
            torch.npu.synchronize()
            if collect is not None:
                for first, second in zip(EV_NAMES, EV_NAMES[1:]):
                    collect[first + "_to_" + second].append(e[first].elapsed_time(e[second]))
            return out, peak

        def mm_only_once():
            half_input = x.to(torch.float16)
            e = {key: torch.npu.Event(enable_timing=True) for key in EV_NAMES}
            e["mm"].record()
            out = torch.mm(half_input.reshape(-1, half_input.shape[-1]), w_t,
                           out_dtype=torch.float32)
            e["end"].record()
            torch.npu.synchronize()
            return e["mm"].elapsed_time(e["end"])

        def clock(fn):
            t0 = torch.npu.Event(enable_timing=True)
            t1 = torch.npu.Event(enable_timing=True)
            t0.record()
            value = fn()
            t1.record()
            torch.npu.synchronize()
            return t0.elapsed_time(t1), value

        variants = {}
        checks = {"range_exceeded_iterations": 0, "scale_not_one_iterations": 0}

        def run_variant(label, call):
            for _ in range(args.warmup):
                call(None)
            collect = {f"{a}_to_{b}": [] for a, b in zip(EV_NAMES, EV_NAMES[1:])}
            samples = []
            last = None
            for _ in range(args.steps):
                ms, value = clock(lambda: call(collect))
                samples.append(ms)
                last = value
                torch.npu.empty_cache()
            variants[label] = {
                "end_to_end_ms": summarize(samples),
                "components_ms": {key: summarize(values) for key, values in collect.items()},
            }
            return last

        ref_out, ref_peak, ref_scale = run_variant("full_guard", lambda c: full_guard_once(x, c))
        fast_out, _ = run_variant("amax_cast_mm", amax_cast_mm_once)
        checks["scale_not_one_iterations"] = 0 if ref_scale == 1.0 else 1
        checks["range_exceeded_iterations"] = 1 if ref_peak > 32752.0 else 0
        ref_view = ref_out.view(torch.int16) if ref_out.numel() else ref_out
        fast_view = fast_out.view(torch.int16)
        checks["bitwise_same_full_vs_amax_cast"] = bool(torch.equal(ref_view, fast_view))
        checks["differing_words_full_vs_amax_cast"] = int((ref_view != fast_view).sum())
        checks["word_count"] = int(ref_view.numel())
        checks["observed_peak_scale1_input"] = ref_peak
        checks["scale1_semantics_note"] = (
            "when exponent==0 the deployed guard divides by 1.0 and multiplies by 1.0; "
            "this comparison is evidence for that branch only, not an approval")
        del ref_out, fast_out, ref_view, fast_view

        # mm-only lower bound (FP16 cast inside the timed call)
        for _ in range(args.warmup):
            mm_only_once()
        mm_samples = []
        for _ in range(args.steps):
            mm_samples.append(mm_only_once())
            torch.npu.empty_cache()
        variants["mm_only"] = {"end_to_end_ms": summarize(mm_samples),
                               "note": "includes the FP32->FP16 cast of x inside the timed call"}

        # exponent>0 branch
        big_samples = []
        big_peaks = []
        for _ in range(2):
            full_guard_once(x_big, None)
        for _ in range(max(5, args.steps // 3)):
            ms, value = clock(lambda: full_guard_once(x_big, None))
            big_samples.append(ms)
            big_peaks.append((value[1], value[2]))
            torch.npu.empty_cache()
        variants["full_guard_big"] = {
            "end_to_end_ms": summarize(big_samples),
            "observed_peak_scale_pairs": [{"peak": p, "scale": s} for p, s in big_peaks[:3]],
            "note": "input pre-scaled so peak>32752; exponent>0 branch, rarely hit in-model",
        }

        flops = 2 * M * k * n
        mm_ms = variants["mm_only"]["end_to_end_ms"]["median"]
        result["cases"][name] = {
            "K": k, "N": n, "input_dtype": in_dtype, "m": M, "flops": flops,
            "mm_only_TFLOPS_per_card": round(flops / (mm_ms * 1e-3) / 1e12, 1),
            "mm_only_percent_of_assumed_peak": round(
                100 * flops / (mm_ms * 1e-3) / (PEAK_FP16_TF * 1e12), 1),
            "x_float_is_alias": alias,
            "weight_t_is_contiguous": bool(w_t.is_contiguous()),
            "weight_mm_layout": "layer.weight.t() non-contiguous view",
            "variants": variants,
            "checks": checks,
            "in_model_ms_from_R147": in_model_ms,
            "limits": ("synthetic tensors; single card; no TP all-reduce, no RowParallelLinear "
                       "wrapper, no rope/silu/residual/AdaLN, no weight-stream neighbours; "
                       "device-event spans exclude host-side python overhead"),
        }
        print("CASE", json.dumps({
            "case": name,
            "mm_only_ms": round(mm_ms, 3),
            "amax_cast_mm_ms": round(variants["amax_cast_mm"]["end_to_end_ms"]["median"], 3),
            "full_guard_ms": round(variants["full_guard"]["end_to_end_ms"]["median"], 3),
            "big_ms": round(variants["full_guard_big"]["end_to_end_ms"]["median"], 3),
            "bitwise_same": checks["bitwise_same_full_vs_amax_cast"],
            "differing_words": checks["differing_words_full_vs_amax_cast"],
            "scale": ref_scale, "peak": round(ref_peak, 3),
        }, ensure_ascii=False), flush=True)
        del x, x_big, w_t, ref_scale
        torch.npu.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY", json.dumps({"out": args.out, "device": result["device"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())