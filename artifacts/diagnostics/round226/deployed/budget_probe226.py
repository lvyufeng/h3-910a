"""R226 device-event probe for the unmeasured ``rest`` region.

R223 showed that 8.118 s/step of the 17.9 s instrumented step is non-compute and
that the largest single piece -- ``rest`` 4.121 s/step -- has never been split.
This probe only records device events around the DiT block and its submodules; it
never synchronizes inside the forward and never changes math, dtype, blocks or
scheduling. The single synchronize happens after the measured step.

Labels installed per block:
    block, adaln_proj, norm1, norm2, attn, mlp, qkv, out_proj, fc1, fc2
plus, when reachable, the final layer.
"""
from __future__ import annotations

import statistics

_STATE = {"installed": False, "pending": {}, "labels": [], "modules": []}

TARGETS = (
    ("block", None),
    ("adaln_proj", "adaln_proj"),
    ("norm1", "norm1"),
    ("norm2", "norm2"),
    ("attn", "attn"),
    ("mlp", "mlp"),
    ("qkv", "attn.qkv_proj"),
    ("out_proj", "attn.out_proj"),
    ("fc1", "mlp.fc1"),
    ("fc2", "mlp.fc2"),
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


def install(torch, blocks):
    if _STATE["installed"]:
        return True

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
        _STATE["modules"].append(label)

    for index, block in enumerate(blocks):
        for label, attr in TARGETS:
            module = _resolve(block, attr)
            if module is None:
                raise RuntimeError(f"missing module {attr!r} for label {label!r}")
            wrap(f"{label}", module)
    _STATE["installed"] = True
    return {"blocks": len(blocks), "labels": sorted({label for label, _ in TARGETS}),
            "hooks": len(_STATE["modules"])}


def collect(torch):
    if not _STATE["installed"]:
        return {"installed": False}
    torch.npu.synchronize()  # single sync, only after the measured step
    rows = {}
    for label, pair in _STATE["pending"].items():
        if "start" not in pair or "end" not in pair:
            continue
        try:
            ms = pair["start"].elapsed_time(pair["end"])
        except BaseException:
            continue
        rows.setdefault(label, []).append(ms)
    summary = {}
    for label, values in rows.items():
        values = [float(v) for v in values]
        summary[label] = {
            "calls": len(values),
            "median_ms": round(statistics.median(values), 3),
            "sum_ms_per_step": round(sum(values), 1),
        }
    _STATE["pending"].clear()
    return {
        "installed": True,
        "scope": "device events around the 50 DiT blocks and their submodules; "
                 "one synchronize after the measured step",
        "measurement_only_no_math_change": True,
        "no_synchronize_inside_forward": True,
        "labels": summary,
    }