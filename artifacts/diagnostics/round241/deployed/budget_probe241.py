"""R241 probe: measure the block-boundary weight wait and capture the real shapes.

R235 (hooks, 100 samples, +0.062 s) left 3.457 s/step unexplained inside the attn/mlp
parent spans but outside their child modules. R240 disproved the "3x GEMM / wrapper"
explanations and established a rule: never replace ``forward`` bodies to instrument.

This probe only *wraps existing methods* (it always calls the original) and adds module
pre-hooks:

  * ``LayerwiseOffloadHook.pre_forward``: records a compute-stream event at block entry
    and compares it with the copy-stream event of **this block's** weights
    (``self._prev_hook._prefetch_done``). A negative delta means the block started
    before its weights landed, i.e. exposed wait = max(0, -delta).
  * ``LayerwiseOffloadHook.offload_layer``: records whether the next block's copy had
    already completed by the time the current block finished (overlap check).
  * attention / MLP module pre-hooks: record ``x.shape`` and per-step call counts.

No stream switching, no synchronize inside the forward; spans are read after the step.
"""
from __future__ import annotations

import statistics

_STATE = {"installed": False, "blocks": [], "overlap": [], "shapes": [], "calls": 0}


def install(torch, transformer_module):
    if _STATE["installed"]:
        return {"already": True}
    import vllm_omni.diffusion.offloader.layerwise_backend as lb

    hook_cls = lb.LayerwiseOffloadHook
    real_pre = hook_cls.pre_forward
    real_offload = hook_cls.offload_layer

    def pre_wrapped(self, module, *args, **kwargs):
        entry = torch.npu.Event(enable_timing=True)
        entry.record()
        prev = getattr(self, "_prev_hook", None)
        own_evt = getattr(prev, "_prefetch_done", None) if prev is not None else None
        delta = None
        if own_evt is not None:
            try:
                delta = float(own_evt.elapsed_time(entry))
            except BaseException:
                delta = None
        result = real_pre(self, module, *args, **kwargs)
        next_evt = getattr(self, "_prefetch_done", None)
        _STATE["blocks"].append({"own_copy_delta_ms": delta,
                                 "entry": entry, "next_copy": next_evt})
        return result

    def offload_wrapped(self):
        next_evt = getattr(self, "_prefetch_done", None)
        result = real_offload(self)
        if next_evt is not None:
            end = torch.npu.Event(enable_timing=True)
            end.record()
            try:
                _STATE["overlap"].append(float(next_evt.elapsed_time(end)))
            except BaseException:
                pass
        return result

    hook_cls.pre_forward = pre_wrapped
    hook_cls.offload_layer = offload_wrapped
    _STATE["installed"] = True
    return {"patched": ["LayerwiseOffloadHook.pre_forward",
                        "LayerwiseOffloadHook.offload_layer"]}


def install_shape_hooks(torch, blocks):
    targets = (("attn", "attn"), ("mlp", "mlp"))

    def resolve(block, attr):
        module = block
        for part in attr.split("."):
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    for block in blocks:
        for label, attr in targets:
            module = resolve(block, attr)
            if module is None:
                raise RuntimeError(f"missing {attr}")

            def pre(mod, inputs, _label=label):
                if inputs and hasattr(inputs[0], "shape"):
                    _STATE["shapes"].append({"module": _label,
                                             "x_shape": list(inputs[0].shape),
                                             "x_dtype": str(inputs[0].dtype)})
                _STATE["calls"] += 1

            module.register_forward_pre_hook(pre)
    return {"hooked": [label for label, _ in targets], "blocks": len(blocks)}


def collect(torch):
    torch.npu.synchronize()
    deltas = [b["own_copy_delta_ms"] for b in _STATE["blocks"]
              if b["own_copy_delta_ms"] is not None]
    stalls = [max(0.0, -d) for d in deltas]
    by_shape = {}
    for entry in _STATE["shapes"]:
        key = f"{entry['module']}:{tuple(entry['x_shape'])}:{entry['x_dtype']}"
        by_shape[key] = by_shape.get(key, 0) + 1
    result = {
        "installed": _STATE["installed"],
        "blocks_observed": len(_STATE["blocks"]),
        "own_weight_copy": {
            "samples": len(deltas),
            "delta_median_ms": round(statistics.median(deltas), 3) if deltas else None,
            "delta_min_ms": round(min(deltas), 3) if deltas else None,
            "stall_blocks": sum(1 for s in stalls if s > 0.5),
            "stall_total_ms": round(sum(stalls), 1),
            "stall_median_ms": round(statistics.median(stalls), 3) if stalls else None,
            "stall_max_ms": round(max(stalls), 3) if stalls else None,
            "note": "delta = this block's weight copy completion -> block entry; "
                    "negative delta means the block started before the copy landed",
        },
        "next_copy_overlap": {
            "samples": len(_STATE["overlap"]),
            "median_ms": round(statistics.median(_STATE["overlap"]), 3) if _STATE["overlap"] else None,
            "positive_blocks": sum(1 for v in _STATE["overlap"] if v > 0),
            "note": "next block's copy completion -> current block end; positive means the "
                    "copy finished after the block, i.e. still in flight",
        },
        "module_call_shapes": by_shape,
        "module_pre_hook_calls": _STATE["calls"],
        "measurement_only_no_math_change": True,
    }
    _STATE["blocks"].clear()
    _STATE["overlap"].clear()
    _STATE["shapes"].clear()
    return result