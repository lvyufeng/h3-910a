"""R235 probe: per-label sample distribution for the attribution inside attn/MLP.

R234 found that the earlier module hooks kept only the *last* call per label
(a single overwritten slot), so the "3.43 s/step unattributed" figure was derived
from single-sample differences. This probe accumulates every sample:

  * module hooks append one (start, end) event pair and one host wall per call,
  * function probes (rope / packed attention / silu) append as before,
  * collect() reports calls / median / p10 / p90 / min / max per label.

Device events only on the default stream, forwards untouched, no stream switching,
no math/dtype/block/offload/scheduling change.
"""
from __future__ import annotations

import statistics
import time

_STATE = {"installed": False, "samples": {}, "functions": {}}

MODULE_TARGETS = (
    ("block", None),
    ("attn", "attn"),
    ("mlp", "mlp"),
    ("qkv", "attn.qkv_proj"),
    ("out_proj", "attn.out_proj"),
    ("q_norm", "attn.q_norm"),
    ("k_norm", "attn.k_norm"),
    ("fc1", "mlp.fc1"),
    ("fc2", "mlp.fc2"),
    ("norm1", "norm1"),
    ("norm2", "norm2"),
    ("adaln_proj", "adaln_proj"),
)


def _resolve(block, attr):
    module = block
    if attr is None:
        return module
    for part in attr.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def _record(label, host_ms, start, end):
    entry = _STATE["samples"].setdefault(label, {"pairs": [], "host_ms": []})
    entry["pairs"].append((start, end))
    entry["host_ms"].append(host_ms)


def _span(label, fn, torch):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host0 = time.perf_counter()
    start.record()
    out = fn()
    end.record()
    _record(label, (time.perf_counter() - host0) * 1000.0, start, end)
    return out


def install_function_probes(torch, transformer_module):
    if _STATE["installed"]:
        return {"already": True}
    real_rope = getattr(transformer_module, "_apply_rope", None)
    if real_rope is not None:
        def rope_patched(x, freqs):
            return _span("rope", lambda: real_rope(x, freqs), torch)
        transformer_module._apply_rope = rope_patched
        _STATE["functions"]["_apply_rope"] = "patched"

    attn_cls = getattr(transformer_module, "MiniMaxH3Attention", None)
    if attn_cls is not None:
        real_packed = attn_cls._run_packed_attention

        def packed_patched(self, q, k, v, **kwargs):
            return _span("packed_attention_uspfakernel",
                         lambda: real_packed(self, q, k, v, **kwargs), torch)

        attn_cls._run_packed_attention = packed_patched
        _STATE["functions"]["_run_packed_attention"] = "patched"

    import torch.nn.functional as F
    real_silu = F.silu

    def silu_patched(x, *args, **kwargs):
        return _span("silu", lambda: real_silu(x, *args, **kwargs), torch)

    F.silu = silu_patched
    _STATE["functions"]["F.silu"] = "patched"
    return {"patched": _STATE["functions"]}


def install(torch, blocks):
    def wrap(label, module):
        def pre(mod, inputs):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            pre.__dict__.setdefault("stack", []).append(
                (event, time.perf_counter()))

        def post(mod, inputs, output):
            stack = pre.__dict__.setdefault("stack", [])
            if not stack:
                return None
            start, host0 = stack.pop()
            end = torch.npu.Event(enable_timing=True)
            end.record()
            _record(label, (time.perf_counter() - host0) * 1000.0, start, end)
            return None

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for block in blocks:
        for label, attr in MODULE_TARGETS:
            module = _resolve(block, attr)
            if module is None:
                raise RuntimeError(f"missing module {attr!r} for label {label!r}")
            wrap(label, module)
    _STATE["installed"] = True
    return {"blocks": len(blocks)}


def _stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None

    def pct(q):
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    return {"calls": len(values),
            "median_ms": round(statistics.median(values), 3),
            "p10_ms": round(pct(0.10), 3),
            "p90_ms": round(pct(0.90), 3),
            "min_ms": round(values[0], 3),
            "max_ms": round(values[-1], 3),
            "sum_ms": round(sum(values), 1)}


def collect(torch):
    torch.npu.synchronize()
    out = {}
    for label, entry in _STATE["samples"].items():
        device, host, bad = [], [], 0
        for first, second in entry["pairs"]:
            try:
                device.append(float(first.elapsed_time(second)))
            except BaseException:
                bad += 1
        host = list(entry["host_ms"])
        if not device:
            continue
        out[label] = {"device": _stats(device), "host": _stats(host), "event_errors": bad}
    _STATE["samples"].clear()
    return {
        "installed": True,
        "scope": "per-call samples over the measured step; default-stream device events "
                 "plus host perf_counter; forwards untouched",
        "measurement_only_no_math_change": True,
        "functions_patched": _STATE["functions"],
        "labels": out,
        "caveats": "silu is global; packed_attention wraps USP all-to-all + FA kernel",
    }