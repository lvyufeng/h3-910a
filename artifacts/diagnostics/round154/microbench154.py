#!/usr/bin/env python3
"""R154: does the strided chunk view actually cause the MLP multiply penalty?

Measures, on one device, the exact operator sequence MiniMaxH3MLP.forward uses:

    h = fp32 [M, 2*N]                  (the fc1 output)
    gate, up = h.chunk(2, dim=-1)      (row stride 2*N, logical width N -> strided)
    out = F.silu(gate) * up

against the contiguous variant that keeps the same arithmetic:

    gate_c = gate.contiguous(); up_c = up.contiguous()
    out = F.silu(gate_c) * up_c

and also times the two copies, so the comparison is fair: the contiguous variant is
only worth it if copies + contiguous work beats the strided work.

It also compares bits between the two paths, because a different kernel may round
differently even though the formula is unchanged.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

M = 10448          # tokens per rank per block
N = 7168           # ffn_hidden per rank (fc1 merged output is 2*N)
REPS = 25
WARMUP = 5


def timed(fn, torch, reps=REPS):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    events = []
    for _ in range(reps):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    torch.npu.synchronize()
    values = [s.elapsed_time(e) for s, e in events]
    return {'median_ms': round(statistics.median(values), 3),
            'min_ms': round(min(values), 3), 'reps': reps}


def main():
    import torch, torch_npu
    torch.npu.set_device(0)
    import torch.nn.functional as F

    torch.manual_seed(0)
    h = torch.randn(M, 2 * N, device='npu', dtype=torch.float32)

    gate, up = h.chunk(2, dim=-1)
    assert not gate.is_contiguous(), 'the chunk halves are expected to be strided'
    stride_row = gate.stride(0)
    width = gate.shape[1]

    gate_c = gate.contiguous()
    up_c = up.contiguous()

    result = {
        'scope': 'R154 microbench: strided chunk view versus contiguous copies for the MLP activation',
        'device': 'npu:0', 'shape': [M, 2 * N], 'row_stride': stride_row, 'logical_width': width,
        'measurement_only': True, 'no_model_loaded': True,
    }

    result['strided'] = {
        'silu_only': timed(lambda: F.silu(gate), torch),
        'mul_only': timed(lambda: torch.mul(gate, up), torch),
        'silu_then_mul': timed(lambda: F.silu(gate) * up, torch),
    }
    result['copies'] = {
        'gate_contiguous': timed(lambda: gate.contiguous(), torch),
        'both_contiguous': timed(lambda: (gate.contiguous(), up.contiguous()), torch),
    }
    result['contiguous'] = {
        'silu_only': timed(lambda: F.silu(gate_c), torch),
        'mul_only': timed(lambda: torch.mul(gate_c, up_c), torch),
        'silu_then_mul': timed(lambda: F.silu(gate_c) * up_c, torch),
    }
    result['contiguous_including_copies'] = {
        'silu_then_mul_plus_both_copies': timed(
            lambda: F.silu(gate.contiguous()) * up.contiguous(), torch),
    }

    # bit identity of the two paths
    a = F.silu(gate) * up
    b = F.silu(gate_c) * up_c
    ac = a.detach().cpu().contiguous(); bc = b.detach().cpu().contiguous()
    diff = (ac - bc).abs()
    result['bit_identity'] = {
        'torch_equal': bool(torch.equal(ac.view(torch.uint8), bc.view(torch.uint8))),
        'differing_elements': int((ac != bc).sum()),
        'total_elements': int(ac.numel()),
        'max_abs_diff': float(diff.max()),
        'relative_RMS': float((diff.double().square().sum().sqrt()) /
                              (bc.double().square().sum().sqrt().clamp_min(1e-30))),
    }

    s = result['strided']['silu_then_mul']['median_ms']
    c = result['contiguous']['silu_then_mul']['median_ms']
    cc = result['contiguous_including_copies']['silu_then_mul_plus_both_copies']['median_ms']
    result['verdict'] = {
        'strided_ms': s, 'contiguous_ms': c, 'contiguous_plus_copies_ms': cc,
        'strided_penalty_factor_vs_contiguous': round(s / c, 2) if c else None,
        'contiguous_path_wins': cc < s,
        'net_saving_ms_per_layer': round(s - cc, 3),
        'net_saving_s_per_step': round((s - cc) * 50 / 1000, 3),
        'usable_as_a_free_optimisation': bool(cc < s and result['bit_identity']['torch_equal']),
    }

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('micro154.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
