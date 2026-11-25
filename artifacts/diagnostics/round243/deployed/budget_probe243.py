"""R243 probe: expose the block-boundary weight wait, without touching any forward.

R242 read the offloader source and established the steady-state structure:

  * ``pre_forward`` issues an async copy of the NEXT block's weights onto ``copy_stream``
    and stores its completion event in ``hook._prefetch_done``;
  * ``post_forward`` -> ``offload_layer`` -> ``clear_block_storage(..., _prefetch_done)``
    makes the compute stream wait for that copy, then sets ``hook._prefetch_done = None``.

So the one place a real wait can surface is the END of a block: if the next block's copy
has not landed by then, the compute stream blocks. That blocked time is invisible in any
op span (it happens inside the wait), which is exactly the shape of the 3.457 s/step that
R235 could not attribute.

To read the event before it is cleared, this probe registers its own forward hook with
``prepend=True`` so it runs BEFORE the offloader's ``post_forward``. It records a compute
event and compares it with ``hook._prefetch_done``:

    delta = elapsed(prefetch_done -> block_end)      # on the device timeline
    positive -> copy landed before the block ended   (no exposed wait)
    negative -> copy still in flight                 (exposed wait = -delta)

Only two events per block are added and no forward/method body is replaced.
"""
from __future__ import annotations

import statistics

_STATE = {"installed": False, "blocks": []}


def _find_offload_hook(module):
    """Find the LayerwiseOffloadHook attached to a block.

    R243 attempt 1 looked in ``_forward_hooks`` and attempt 2 in the bound methods of
    ``_forward_pre_hooks``; both are wrong. Reading the source shows the hook is stored
    in a custom registry: ``registry.register_hook(LayerwiseOffloadHook._HOOK_NAME, hook)``
    on ``module._hook_registry``. This walks that registry defensively.
    """
    registry = getattr(module, "_hook_registry", None)
    if registry is None:
        return None
    seen = []
    for attr in dir(registry):
        if attr.startswith("__"):
            continue
        try:
            value = getattr(registry, attr)
        except BaseException:
            continue
        if isinstance(value, dict):
            seen.extend(value.values())
    seen.append(registry)
    for candidate in seen:
        if hasattr(candidate, "_prefetch_done") and hasattr(candidate, "prefetch_layer"):
            return candidate
    return None


def install(torch, blocks):
    if _STATE["installed"]:
        return {"already": True}
    installed = 0
    for index, block in enumerate(blocks):
        hook = _find_offload_hook(block)
        if hook is None:
            continue

        def pre(mod, inputs, _index=index):
            entry = torch.npu.Event(enable_timing=True)
            entry.record()
            pre.__dict__.setdefault("stack", []).append(entry)

        def post(mod, inputs, output, _index=index, _hook=hook):
            stack = pre.__dict__.setdefault("stack", [])
            entry = stack.pop() if stack else None
            end = torch.npu.Event(enable_timing=True)
            end.record()
            evt = getattr(_hook, "_prefetch_done", None)
            delta = None
            if evt is not None:
                try:
                    delta = float(evt.elapsed_time(end))
                except BaseException:
                    delta = None
            span = None
            if entry is not None:
                try:
                    span = float(entry.elapsed_time(end))
                except BaseException:
                    span = None
            _STATE["blocks"].append({"index": _index, "delta_ms": delta, "span_ms": span})
            return None

        block.register_forward_pre_hook(pre, prepend=True)
        block.register_forward_hook(post, prepend=True)
        installed += 1
    _STATE["installed"] = installed > 0
    return {"blocks_hooked": installed, "total_blocks": len(blocks)}


def _summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"samples": len(values),
            "median_ms": round(statistics.median(values), 3),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
            "sum_ms": round(sum(values), 1)}


def collect():
    entries = list(_STATE["blocks"])
    _STATE["blocks"].clear()
    if not entries:
        return {"installed": _STATE["installed"], "blocks": 0}
    # two forward passes happen (warmup + measured); the later half is the measured step
    half = len(entries) // 2
    groups = {"all": entries, "warmup_like": entries[:half], "measured_like": entries[half:]}
    out = {"installed": _STATE["installed"], "blocks": len(entries)}
    for name, group in groups.items():
        deltas = [e["delta_ms"] for e in group if e["delta_ms"] is not None]
        stalls = [max(0.0, -d) for d in deltas]
        spans = [e["span_ms"] for e in group if e["span_ms"] is not None]
        out[name] = {
            "delta": _summary(deltas),
            "block_span": _summary(spans),
            "ready_blocks": sum(1 for d in deltas if d > 0),
            "late_blocks": sum(1 for d in deltas if d <= 0),
            "stall_total_ms": round(sum(stalls), 1),
            "stall_max_ms": round(max(stalls), 3) if stalls else None,
            "stall_blocks_over_1ms": sum(1 for s in stalls if s > 1.0),
        }
    return out