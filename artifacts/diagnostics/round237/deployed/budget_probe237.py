"""R237 probe: per-segment device events inside MiniMaxH3Attention.forward.

R236 showed the module-hook attribution under-counts ``mlp.fc1`` by 31.3 ms (hook
23.4 ms vs 54.7 ms measured directly inside the forward), which fully explained the
MLP parent-span gap. The attention parent span still has an unexplained 25.0 ms/block,
and the same under-counting may apply to ``qkv_proj``/``out_proj``.

This probe replaces ``MiniMaxH3Attention.forward`` with the **same statements in the
same order**, adding one device event between each step:

    e0 -> qkv, _ = self.qkv_proj(x)                 -> e1
    e1 -> q/k/v split + view                        -> e2
    e2 -> q = self.q_norm(q)                        -> e3
    e3 -> k = self.k_norm(k)                        -> e4
    e4 -> q = _apply_rope(q, freqs)                 -> e5
    e5 -> k = _apply_rope(k, freqs)                 -> e6
    e6 -> out = self._run_packed_attention(...)     -> e7
    e7 -> out = out.reshape(total, heads*dim)       -> e8
    e8 -> out, _ = self.out_proj(out)               -> e9

No stream switching, no synchronize inside the forward; spans are read after the
measured step. The MLP probe from R236 is installed together so both regions stay
comparable within one run.
"""
from __future__ import annotations

import statistics

_STATE = {"installed": False, "spans": {}, "calls": 0}

ATTN_LABELS = (
    "attn_qkv",
    "attn_split_view",
    "attn_q_norm",
    "attn_k_norm",
    "attn_rope_q",
    "attn_rope_k",
    "attn_packed_uspfakernel",
    "attn_reshape",
    "attn_out_proj",
)
MLP_LABELS = (
    "mlp_fc1",
    "mlp_chunk",
    "mlp_gate_cast",
    "mlp_silu",
    "mlp_up_cast",
    "mlp_mul",
    "mlp_fc2",
)


def _record(label, start, end):
    _STATE["spans"].setdefault(label, []).append((start, end))


def install_probes(torch, transformer_module):
    if _STATE["installed"]:
        return {"already": True}
    attn_cls = getattr(transformer_module, "MiniMaxH3Attention", None)
    mlp_cls = getattr(transformer_module, "MiniMaxH3MLP", None)
    if attn_cls is None or mlp_cls is None:
        raise RuntimeError("MiniMaxH3Attention / MiniMaxH3MLP not found")

    import torch.nn.functional as F

    def attn_forward(self, x, *, rope_freqs, cu_seqlens, max_seqlen,
                     packed_total=None, sp_seq_lens=None):
        ev = [torch.npu.Event(enable_timing=True) for _ in range(10)]
        ev[0].record()
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        ev[1].record()
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)
        ev[2].record()
        q = self.q_norm(q)
        ev[3].record()
        k = self.k_norm(k)
        ev[4].record()
        if rope_freqs is not None:
            q = transformer_module._apply_rope(q, rope_freqs)
            ev[5].record()
            k = transformer_module._apply_rope(k, rope_freqs)
            ev[6].record()
        else:
            ev[5].record()
            ev[6].record()
        out = self._run_packed_attention(
            q, k, v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            packed_total=packed_total if packed_total is not None else q.shape[0],
        )
        ev[7].record()
        out = out.reshape(total, self.num_heads * self.head_dim)
        ev[8].record()
        out, _ = self.out_proj(out)
        ev[9].record()
        _STATE["calls"] += 1
        for index, label in enumerate(ATTN_LABELS):
            _record(label, ev[index], ev[index + 1])
        _record("attn_total", ev[0], ev[9])
        return out

    def mlp_forward(self, x):
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
        for index, label in enumerate(MLP_LABELS):
            _record(label, ev[index], ev[index + 1])
        _record("mlp_total", ev[0], ev[7])
        return out

    attn_cls.forward = attn_forward
    mlp_cls.forward = mlp_forward
    _STATE["installed"] = True
    return {"patched": ["MiniMaxH3Attention.forward", "MiniMaxH3MLP.forward"],
            "attn_segments": list(ATTN_LABELS), "mlp_segments": list(MLP_LABELS)}


def collect(torch):
    torch.npu.synchronize()
    out = {}
    for label, pairs in _STATE["spans"].items():
        values, errors = [], 0
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
            "event_errors": errors,
        }
    _STATE["spans"].clear()
    return {
        "installed": _STATE["installed"],
        "forward_calls": _STATE["calls"],
        "measurement_only_no_math_change": True,
        "note": "same statements in the same order; only device events added",
        "segments": out,
    }