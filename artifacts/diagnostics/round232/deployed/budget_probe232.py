"""R232 probe: host wall vs device span inside the attention and MLP forwards.

R230 corrected the framing: block spans are 345.8 ms and the step is 17.9 s, so the
device is busy ~97% of the step. The 4.09 s "rest" is therefore **unattributed device
work**, not idle gaps. R228/R229/R230 already disproved weight wait, allocator churn
and audit-sync explanations.

This probe keeps the deployed forwards untouched and only adds timing:

  * standard forward hooks on the existing modules (attn, mlp, qkv_proj, out_proj,
    fc1, fc2, q_norm, k_norm),
  * a monkeypatch of the module-level ``_apply_rope`` (called twice per block),
  * a monkeypatch of ``MiniMaxH3Attention._run_packed_attention`` (wraps the USP
    all-to-all plus the FA kernel together),
  * a monkeypatch of ``torch.nn.functional.silu`` to time the FP32 silu of the MLP.

All events go on the default stream; there is no stream switching (R227 showed that
stream switching perturbs the step by 15%). The caller must verify step invariance
before trusting the numbers.
"""
from __future__ import annotations

import statistics

_STATE = {"installed": False, "pending": {}, "functions": {}, "silu": []}

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


def _span(label, fn, torch):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host0 = __import__("time").perf_counter()
    start.record()
    out = fn()
    end.record()
    host_ms = (__import__("time").perf_counter() - host0) * 1000.0
    entry = _STATE["pending"].setdefault(label, {})
    entry.setdefault("pairs", []).append((start, end))
    entry.setdefault("host_ms", []).append(host_ms)
    return out


def install_function_probes(torch, transformer_module):
    """Patch only module-level functions / one method; no stream switching."""
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
            entry = _STATE["pending"].setdefault(label, {})
            entry["start"] = event
            entry["host0"] = __import__("time").perf_counter()

        def post(mod, inputs, output):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            entry = _STATE["pending"].setdefault(label, {})
            entry["end"] = event
            entry["host_ms"] = (__import__("time").perf_counter() - entry.get("host0", 0.0)) * 1000.0
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


def collect(torch):
    torch.npu.synchronize()
    out = {}
    for label, data in _STATE["pending"].items():
        spans = []
        if "start" in data and "end" in data:
            try:
                spans.append(float(data["start"].elapsed_time(data["end"])))
            except BaseException:
                pass
        for first, second in data.get("pairs", []):
            try:
                spans.append(float(first.elapsed_time(second)))
            except BaseException:
                pass
        host = list(data.get("host_ms", [])) if isinstance(data.get("host_ms"), list) else []
        if isinstance(data.get("host_ms"), float):
            host = [data["host_ms"]]
        if spans:
            out[label] = {"calls": len(spans),
                          "median_ms": round(statistics.median(spans), 3),
                          "sum_ms": round(sum(spans), 1),
                          "host_ms": round(statistics.median(host), 3) if host else None,
                          "device_minus_host_ms": (round(statistics.median(spans) - statistics.median(host), 3)
                                                   if host else None)}
    _STATE["pending"].clear()
    return {
        "installed": True,
        "scope": "default-stream device events plus host perf_counter; forwards untouched",
        "measurement_only_no_math_change": True,
        "functions_patched": _STATE["functions"],
        "labels": out,
        "caveats": (
            "F.silu is patched globally, so its span aggregates every silu in the step "
            "(MLP, AdaLN). packed_attention_uspfakernel wraps the USP all-to-all plus the "
            "FA kernel, so it is not pure kernel time."),
    }