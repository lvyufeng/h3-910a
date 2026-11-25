"""R239 probe: is the 3x in-model GEMM gap in the kernel call or in the wrapper?

R237: in-model qkv 41.56 ms/block vs 12.47 ms for the same shape standalone; fc1 54.43
vs 19.04. R238 ruled out eight-card contention (concurrent == single, <=0.4%). The
remaining split is:

  (a) the GEMM call itself is slower under the model's operand layout, or
  (b) the parallel-linear wrapper around it does extra work (gather/split/cast).

This probe patches ``UnquantizedLinearMethod.apply`` (the class the qkv/fc1 layers use)
and records, per call: device-event span of ``apply``, host wall, ``x`` shape/dtype/stride,
weight shape/dtype/stride, and the layer class name. Wrapper remainder is then
(module span from R237) - (apply span).

Measurement only: forwards untouched apart from the wrapper, no stream switching, no
synchronize inside the forward.
"""
from __future__ import annotations

import statistics
import time

_STATE = {"installed": False, "samples": {}}


def _record(label, payload):
    _STATE["samples"].setdefault(label, []).append(payload)


def install_linear_probe(torch):
    if _STATE["installed"]:
        return {"already": True}
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    real_apply = UnquantizedLinearMethod.apply

    def apply_probe(self, layer, x, bias=None):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        host0 = time.perf_counter()
        start.record()
        out = real_apply(self, layer, x, bias)
        end.record()
        host_ms = (time.perf_counter() - host0) * 1000.0
        weight = layer.weight
        label = f"{type(layer).__name__}:{tuple(weight.shape)}"
        _record(label, {
            "start": start, "end": end, "host_ms": host_ms,
            "x_shape": list(x.shape), "x_dtype": str(x.dtype),
            "x_contiguous": bool(x.is_contiguous()),
            "x_stride": list(x.stride()),
            "w_shape": list(weight.shape), "w_dtype": str(weight.dtype),
            "w_contiguous": bool(weight.is_contiguous()),
            "w_stride": list(weight.stride()),
            "bias": bias is not None,
        })
        return out

    UnquantizedLinearMethod.apply = apply_probe
    _STATE["installed"] = True
    return {"patched": "UnquantizedLinearMethod.apply"}


def _stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None

    def pct(q):
        return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]

    return {"calls": len(values), "median_ms": round(statistics.median(values), 3),
            "p10_ms": round(pct(0.10), 3), "p90_ms": round(pct(0.90), 3),
            "min_ms": round(values[0], 3), "max_ms": round(values[-1], 3)}


def collect(torch):
    torch.npu.synchronize()
    grouped = {}
    for label, entries in _STATE["samples"].items():
        device, host = [], []
        for entry in entries:
            try:
                device.append(float(entry["start"].elapsed_time(entry["end"])))
                host.append(entry["host_ms"])
            except BaseException:
                continue
        if not device:
            continue
        first = entries[0]
        grouped[label] = {
            "device": _stats(device), "host": _stats(host),
            "x_shape": first["x_shape"], "x_dtype": first["x_dtype"],
            "x_contiguous": first["x_contiguous"], "x_stride": first["x_stride"],
            "w_shape": first["w_shape"], "w_dtype": first["w_dtype"],
            "w_contiguous": first["w_contiguous"], "w_stride": first["w_stride"],
            "bias": first["bias"],
        }
    _STATE["samples"].clear()
    return {"installed": _STATE["installed"], "layers": grouped,
            "measurement_only_no_math_change": True}