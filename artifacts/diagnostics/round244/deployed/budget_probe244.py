"""R244 probe: account the block-boundary weight wait at its only real call site.

R243 established (source read + failed instrumentation) that the single place a weight
wait can surface is:

    post_forward -> offload_layer -> clear_block_storage(params, buffers, ready_event)
        -> current_stream().wait_event(ready_event)     # compute waits for next copy
        -> clear_tensor_storage(...)

and that instrumenting the block module itself perturbs the schedule (+11 ms/block), so
that route was rejected. This probe wraps the *function* `clear_block_storage` as it is
bound inside `layerwise_backend` (it is imported by name, so patching `tensor_utils` would
have no effect), records one compute-stream event per block, and compares it with the
event the offloader already created on the copy stream:

    delta = elapsed(ready_event -> compute_now)
    delta > 0  -> the next block's weights landed before this block ended (no exposed wait)
    delta <= 0 -> the copy was still in flight (exposed wait = -delta)

No forward, hook registry or stream is touched; the wrapper only adds one event record
and one `elapsed_time` per block.
"""
from __future__ import annotations

import statistics

_STATE = {"installed": False, "deltas": [], "errors": 0, "calls": 0, "types": None}


def install(torch):
    if _STATE["installed"]:
        return {"already": True}
    import vllm_omni.diffusion.offloader.layerwise_backend as lb

    real = lb.clear_block_storage

    def wrapped(params, buffers, ready_event=None):
        _STATE["calls"] += 1
        if ready_event is not None:
            now = torch.npu.Event(enable_timing=True)
            now.record()
            try:
                delta = float(ready_event.elapsed_time(now))
                _STATE["deltas"].append(delta)
                if _STATE["types"] is None:
                    _STATE["types"] = f"{type(ready_event).__module__}.{type(ready_event).__name__}"
            except BaseException as exc:  # pragma: no cover - defensive
                _STATE["errors"] += 1
                if _STATE["types"] is None:
                    _STATE["types"] = f"error:{type(exc).__name__}"
        return real(params, buffers, ready_event)

    lb.clear_block_storage = wrapped
    _STATE["installed"] = True
    return {"patched": "layerwise_backend.clear_block_storage"}


def collect(torch):
    torch.npu.synchronize()
    deltas = list(_STATE["deltas"])
    result = {
        "installed": _STATE["installed"],
        "clear_calls": _STATE["calls"],
        "event_samples": len(deltas),
        "event_errors": _STATE["errors"],
        "ready_event_type": _STATE["types"],
        "measurement_only_no_math_change": True,
    }
    if deltas:
        half = len(deltas) // 2
        for name, group in (("all", deltas), ("warmup_like", deltas[:half]),
                            ("measured_like", deltas[half:])):
            stalls = [max(0.0, -d) for d in group]
            result[name] = {
                "samples": len(group),
                "delta_median_ms": round(statistics.median(group), 3),
                "delta_min_ms": round(min(group), 3),
                "delta_max_ms": round(max(group), 3),
                "ready_blocks": sum(1 for d in group if d > 0),
                "late_blocks": sum(1 for d in group if d <= 0),
                "stall_total_ms": round(sum(stalls), 1),
                "stall_max_ms": round(max(stalls), 3),
                "stall_blocks_over_1ms": sum(1 for s in stalls if s > 1.0),
            }
    _STATE["deltas"].clear()
    return result