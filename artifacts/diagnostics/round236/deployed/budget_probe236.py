"""R236 probe: per-op device events inside MiniMaxH3MLP.forward.

R235 confirmed with 100 samples that the MLP parent span (88.25 ms/block) exceeds
fc1+fc2 (53.4 ms) by a stable 34.8 ms/block = 1.74 s/step, while the elementwise
chain itself costs only ~3.6 ms standalone. R228-R234 disproved every external
mechanism (weight wait, allocator churn, audit sync, guard sync, host dispatch,
allocation blocking).

This probe replaces ``MiniMaxH3MLP.forward`` with the **same statements in the same
order**, adding one device event between each step:

    hidden, _ = self.fc1(x)          # e0 -> e1
    gate, up = hidden.chunk(2, -1)   # e1 -> e2
    g = gate.float()                 # e2 -> e3
    s = F.silu(g)                    # e3 -> e4
    u = up.float()                   # e4 -> e5
    h = s * u                        # e5 -> e6
    out, _ = self.fc2(h)             # e6 -> e7

The deployed expression ``silu(gate.float()) * up.float()`` evaluates in exactly that
order, so the arithmetic, dtypes and allocations are unchanged. No stream switching,
no synchronize inside the forward; spans are read after the measured step.
"""
from __future__ import annotations

import statistics

_STATE = {"installed": False, "spans": {}, "calls": 0}

STEPS = (
    ("mlp_fc1", "mlp_chunk", "mlp_gate_cast", "mlp_silu", "mlp_up_cast", "mlp_mul", "mlp_fc2")
)


def _record(label, start, end):
    _STATE["spans"].setdefault(label, []).append((start, end))


def install_mlp_probe(torch, transformer_module):
    if _STATE["installed"]:
        return {"already": True}
    mlp_cls = getattr(transformer_module, "MiniMaxH3MLP", None)
    if mlp_cls is None:
        raise RuntimeError("MiniMaxH3MLP not found")

    import torch.nn.functional as F

    def forward_probe(self, x):
        ev = [torch.npu.Event(enable_timing=True) for _ in range(8)]
        ev[0].record()
        hidden, _ = self.fc1(x)
        ev[1].record()
        gate, up = hidden.chunk(2, dim=-1)
        ev[2].record()
        gate_f = gate.float()
        ev[3].record()
        silu = F.silu(gate_f)
        ev[4].record()
        up_f = up.float()
        ev[5].record()
        product = silu * up_f
        ev[6].record()
        out, _ = self.fc2(product)
        ev[7].record()
        _STATE["calls"] += 1
        labels = ("mlp_fc1__to__mlp_chunk", "mlp_chunk__to__mlp_gate_cast",
                  "mlp_gate_cast__to__mlp_silu", "mlp_silu__to__mlp_up_cast",
                  "mlp_up_cast__to__mlp_mul", "mlp_mul__to__mlp_fc2",
                  "mlp_fc2__to__mlp_end")
        for index, label in enumerate(labels):
            _record(label, ev[index], ev[index + 1])
        _record("mlp_total", ev[0], ev[7])
        return out

    mlp_cls.forward = forward_probe
    _STATE["installed"] = True
    return {"patched": "MiniMaxH3MLP.forward", "segments": list(STEPS[0])}


def collect(torch):
    torch.npu.synchronize()
    out = {}
    for label, pairs in _STATE["spans"].items():
        values = []
        errors = 0
        for first, second in pairs:
            try:
                values.append(float(first.elapsed_time(second)))
            except BaseException:
                errors += 1
        if not values:
            continue
        values.sort()
        out[label] = {
            "calls": len(values),
            "median_ms": round(statistics.median(values), 3),
            "p10_ms": round(values[max(0, int(0.10 * (len(values) - 1)))], 3),
            "p90_ms": round(values[min(len(values) - 1, int(0.90 * (len(values) - 1)))], 3),
            "max_ms": round(values[-1], 3),
            "sum_ms": round(sum(values), 1),
            "event_errors": errors,
        }
    _STATE["spans"].clear()
    return {
        "installed": _STATE["installed"],
        "mlp_forward_calls": _STATE["calls"],
        "measurement_only_no_math_change": True,
        "note": "same statements in the same order as the deployed forward; only device "
                "events added; no stream switching and no synchronize inside forward",
        "segments": out,
    }