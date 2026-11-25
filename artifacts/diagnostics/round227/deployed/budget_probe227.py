"""R227 probe: is the 3.37 s/step attn+mlp remainder exposed weight-prefetch wait?

Mechanism found in ``vllm_omni/diffusion/offloader``:

  * ``LayerwiseOffloadHook.pre_forward`` prefetches the *next* block's weights on a
    private copy stream, non-blocking.
  * ``post_forward -> offload_layer -> clear_block_storage(..., ready_event)``
    enqueues ``current_stream().wait_event(next_block_copy_event)`` on the compute
    stream before freeing the current block, so the next block cannot start until
    its weights landed.
  * ``pre_forward`` also has a *synchronous* fallback
    (``prefetch_layer(non_blocking=False)``) when the block is not materialized.

So the exposed wait shows up at block boundaries, outside the pre/post span that
R226 timed. This probe measures, without changing policy or math:

  1. per-prefetch copy duration on the copy stream (device events on that stream),
  2. how often the synchronous fallback fires and how long it blocks the host,
  3. the same block/attn/mlp spans as R226, for the same-step comparison.

Nothing is adopted; no math, dtype, block count, offload policy or scheduling change.
"""
from __future__ import annotations

import statistics
import time

_STATE = {
    "installed": False,
    "pending": {},
    "copy_ms": [],
    "sync_prefetch_ms": [],
    "sync_prefetch_count": 0,
    "blocks_seen": 0,
}

TARGETS = (
    ("block", None),
    ("attn", "attn"),
    ("mlp", "mlp"),
    ("qkv", "attn.qkv_proj"),
    ("out_proj", "attn.out_proj"),
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


def install_offloader_probe(torch, hook_cls):
    """Instrument the layerwise hook class in place (measurement only)."""
    if getattr(hook_cls, "_r227_patched", False):
        return {"already": True}

    real_prefetch = hook_cls.prefetch_layer
    real_pre_forward = hook_cls.pre_forward

    def prefetch_patched(self, non_blocking=True):
        if non_blocking:
            start = torch.npu.Event(enable_timing=True)
            with torch.npu.stream(self.copy_stream):
                start.record()
            real_prefetch(self, non_blocking=True)
            done = getattr(self, "_prefetch_done", None)
            if done is not None:
                try:
                    _STATE["copy_ms"].append(float(start.elapsed_time(done)))
                except BaseException:
                    pass
        else:
            wall = time.perf_counter()
            real_prefetch(self, non_blocking=False)
            _STATE["sync_prefetch_ms"].append((time.perf_counter() - wall) * 1000.0)
            _STATE["sync_prefetch_count"] += 1

    def pre_forward_patched(self, module, *args, **kwargs):
        _STATE["blocks_seen"] += 1
        return real_pre_forward(self, module, *args, **kwargs)

    hook_cls.prefetch_layer = prefetch_patched
    hook_cls.pre_forward = pre_forward_patched
    hook_cls._r227_patched = True
    return {"patched": True, "hook_class": hook_cls.__name__}


def install(torch, blocks):
    def wrap(label, module):
        def pre(mod, inputs):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            _STATE["pending"].setdefault(label, {})["start"] = event

        def post(mod, inputs, output):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            _STATE["pending"].setdefault(label, {})["end"] = event
            return None

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    for block in blocks:
        for label, attr in TARGETS:
            module = _resolve(block, attr)
            if module is None:
                raise RuntimeError(f"missing module {attr!r} for label {label!r}")
            wrap(label, module)
    _STATE["installed"] = True
    return {"blocks": len(blocks)}


def collect(torch):
    torch.npu.synchronize()
    spans = {}
    for label, pair in _STATE["pending"].items():
        if "start" not in pair or "end" not in pair:
            continue
        try:
            spans.setdefault(label, []).append(float(pair["start"].elapsed_time(pair["end"])))
        except BaseException:
            continue
    _STATE["pending"].clear()
    out_spans = {lbl: {"calls": len(v), "median_ms": round(statistics.median(v), 3),
                       "sum_ms": round(sum(v), 1)} for lbl, v in spans.items()}
    copies = _STATE["copy_ms"]
    syncs = _STATE["sync_prefetch_ms"]
    return {
        "installed": _STATE["installed"],
        "scope": "device events on the layerwise copy stream plus block spans; one "
                 "synchronize after the measured step; no policy change",
        "measurement_only_no_math_change": True,
        "copy_stream": {
            "prefetch_calls_timed": len(copies),
            "copy_ms_sum_per_step": round(sum(copies), 1) if copies else None,
            "copy_ms_median": round(statistics.median(copies), 3) if copies else None,
            "copy_ms_min": round(min(copies), 3) if copies else None,
            "copy_ms_max": round(max(copies), 3) if copies else None,
            "implied_GB_per_s": (round(50 * 32279142400 / 8 / (sum(copies) / 1000) / 1e9, 1)
                                 if copies else None),
            "note": "measured on the real copy stream during the real step; the "
                    "26.2% duty-cycle comparison uses 646 MB per prefetch",
        },
        "synchronous_fallback": {
            "count": _STATE["sync_prefetch_count"],
            "host_wall_ms": round(sum(syncs), 1),
            "per_call_median_ms": round(statistics.median(syncs), 3) if syncs else None,
        },
        "spans": out_spans,
        "blocks_seen": _STATE["blocks_seen"],
    }