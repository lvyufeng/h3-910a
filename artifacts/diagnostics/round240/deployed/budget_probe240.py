"""R240 probe: where does the wrapper overhead sit, and what is the real token layout?

R239 measured `apply` at 17.07 ms (qkv) / 23.31 ms (fc1) while R237 measured the whole
module call at 41.56 / 54.43 ms, so ~24.5 / 31.1 ms per block happens outside the GEMM
call. This probe localises it by bracketing three regions in the SAME run:

  m0 --(entry gap)--> e0 --(apply/GEMM)--> e1 --(exit gap)--> m1

using module pre/post hooks (m) plus a patch of `UnquantizedLinearMethod.apply` (e).
It also records, without any device read, the real attention layout metadata:
`q/k/v` shapes, `packed_total`, `max_seqlen`, `cu_seqlens` shape/dtype and `sp_seq_lens`.

Measurement only: no math/dtype/block/offload/scheduling change, no stream switching,
no synchronize inside the forward.
"""
from __future__ import annotations

import statistics
import time

_STATE = {"installed": False, "samples": {}, "layout": []}


def _record(label, payload):
    _STATE["samples"].setdefault(label, []).append(payload)


def install(torch, transformer_module):
    if _STATE["installed"]:
        return {"already": True}
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    real_apply = UnquantizedLinearMethod.apply

    def apply_probe(self, layer, x, bias=None):
        entry = torch.npu.Event(enable_timing=True)
        exit_ = torch.npu.Event(enable_timing=True)
        host0 = time.perf_counter()
        entry.record()
        out = real_apply(self, layer, x, bias)
        exit_.record()
        host = (time.perf_counter() - host0) * 1000.0
        label = f"apply:{type(layer).__name__}:{tuple(layer.weight.shape)}"
        _record(label, {"entry": entry, "exit": exit_, "host_ms": host,
                        "x_shape": list(x.shape)})
        return out

    UnquantizedLinearMethod.apply = apply_probe

    real_packed = transformer_module.MiniMaxH3Attention._run_packed_attention

    def packed_probe(self, q, k, v, *, cu_seqlens, max_seqlen, packed_total):
        if len(_STATE["layout"]) < 4:
            _STATE["layout"].append({
                "q_shape": list(q.shape), "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "packed_total": int(packed_total) if packed_total is not None else None,
                "max_seqlen": int(max_seqlen),
                "cu_seqlens_shape": list(cu_seqlens.shape),
                "cu_seqlens_dtype": str(cu_seqlens.dtype),
            })
        return real_packed(self, q, k, v, cu_seqlens=cu_seqlens,
                           max_seqlen=max_seqlen, packed_total=packed_total)

    transformer_module.MiniMaxH3Attention._run_packed_attention = packed_probe
    _STATE["installed"] = True
    return {"patched": ["UnquantizedLinearMethod.apply",
                        "MiniMaxH3Attention._run_packed_attention"]}


def install_module_hooks(torch, blocks):
    targets = (("qkv", "attn.qkv_proj"), ("fc1", "mlp.fc1"))

    def resolve(block, attr):
        module = block
        for part in attr.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def wrap(label, module):
        def pre(mod, inputs):
            entry = torch.npu.Event(enable_timing=True)
            entry.record()
            pre.__dict__.setdefault("stack", []).append((entry, time.perf_counter()))

        def post(mod, inputs, output):
            stack = pre.__dict__.setdefault("stack", [])
            if not stack:
                return None
            entry, host0 = stack.pop()
            exit_ = torch.npu.Event(enable_timing=True)
            exit_.record()
            _record(f"module:{label}", {"entry": entry, "exit": exit_,
                                        "host_ms": (time.perf_counter() - host0) * 1000.0})
            return None

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for block in blocks:
        for label, attr in targets:
            module = resolve(block, attr)
            if module is None:
                raise RuntimeError(f"missing {attr}")
            wrap(label, module)
    return {"hooked": [label for label, _ in targets], "blocks": len(blocks)}


def _stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None

    def pct(q):
        return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]

    return {"calls": len(values), "median_ms": round(statistics.median(values), 3),
            "p10_ms": round(pct(0.10), 3), "p90_ms": round(pct(0.90), 3)}


def _spans(entries, key_a, key_b):
    out = []
    for entry in entries:
        try:
            out.append(float(entry[key_a].elapsed_time(entry[key_b])))
        except BaseException:
            continue
    return out


def collect(torch):
    torch.npu.synchronize()
    result = {"install": _STATE["installed"], "layout_samples": _STATE["layout"],
              "regions": {}, "measurement_only_no_math_change": True}
    for layer in ("qkv", "fc1"):
        modules = _STATE["samples"].get(f"module:{layer}", [])
        apply_key = None
        for key in _STATE["samples"]:
            klass = "QKVParallelLinear" if layer == "qkv" else "MergedColumnParallelLinear"
            if key.startswith(f"apply:{klass}:"):
                apply_key = key
                break
        applies = _STATE["samples"].get(apply_key, []) if apply_key else []
        pairs = min(len(modules), len(applies))
        entry_gap, apply_span, exit_gap, module_span = [], [], [], []
        for index in range(pairs):
            mod = modules[index]
            app = applies[index]
            try:
                entry_gap.append(float(mod["entry"].elapsed_time(app["entry"])))
                apply_span.append(float(app["entry"].elapsed_time(app["exit"])))
                exit_gap.append(float(app["exit"].elapsed_time(mod["exit"])))
                module_span.append(float(mod["entry"].elapsed_time(mod["exit"])))
            except BaseException:
                continue
        result["regions"][layer] = {
            "paired_calls": pairs,
            "module_span_ms": _stats(module_span),
            "entry_gap_ms": _stats(entry_gap),
            "apply_ms": _stats(apply_span),
            "exit_gap_ms": _stats(exit_gap),
            "host_module_ms": _stats([m["host_ms"] for m in modules]),
            "host_apply_ms": _stats([a["host_ms"] for a in applies]),
        }
    _STATE["samples"].clear()
    _STATE["layout"].clear()
    return result