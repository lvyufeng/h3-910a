#!/usr/bin/env python3
"""R184: how fast is one audit layer when the check becomes a single custom-op call?

Measures, on the three real audited tensor shapes, both the current audit path
(ieee_nonfinite_masks + abs==65504 + two host syncs, exactly as matrix_safety126 does it) and a
single call into the H3RangeStats custom op through ctypes. This is a per-layer measurement with
the host sync included, which is what the end-to-end step actually pays; R168 showed that device
time alone does not predict the step time.
"""
from __future__ import annotations
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, '/tmp/h3_audit184')
from npu_half_nan_compat124 import ieee_nonfinite_masks

LIB = ctypes.CDLL('/tmp/h3_lib184/libh3fast184.so')
LIB.h3_max_mag_bits.restype = ctypes.c_int
LIB.h3_max_mag_bits.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]

OUT = Path('/tmp/h3_audit184/probe184.json')
SHAPES = [('mlp_fc1', 10448, 14336, 50), ('qkv_proj', 10448, 10752, 50), ('adaln', 3, 96768, 55)]
WARMUP, REPS = 3, 12


def stream_ptr():
    # npu_stream is the raw aclrtStream handle as an int; wrapping _as_parameter_ again fails
    return ctypes.c_void_p(torch_npu.npu.current_stream().npu_stream)


def native_check(value):
    """Exactly what matrix_safety126._hook does in strict mode."""
    nan, posinf, neginf = ieee_nonfinite_masks(value)
    bad = (nan | posinf | neginf).sum(dtype=torch.int32)
    boundary = (value.abs() == 65504).sum(dtype=torch.int32)
    return int(bad), int(boundary)          # both force a host sync, as strict mode does


def fast_check(value):
    """One custom-op call; same verdict rule applied to the max magnitude bits."""
    max_mag = LIB.h3_max_mag_bits(
        ctypes.c_void_p(value.data_ptr()), value.shape[0], value.shape[1], stream_ptr())
    if max_mag < 0:
        raise RuntimeError('h3_max_mag_bits failed')
    nonfinite = max_mag >= 0x7c00
    saturated = max_mag == 0x7bff
    return nonfinite, saturated


def timeit(fn, value, reps):
    times = []
    for i in range(WARMUP + reps):
        t0 = time.perf_counter()
        fn(value)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times[WARMUP:]), min(times[WARMUP:])


def main():
    torch.npu.set_device(0)
    rows = []
    for name, r, c, count in SHAPES:
        value = torch.randn((r, c), dtype=torch.float16, device='npu')
        v_native = native_check(value)
        v_fast = fast_check(value)

        nat_med, nat_best = timeit(native_check, value, REPS)
        fast_med, fast_best = timeit(fast_check, value, REPS)
        rows.append({
            'tensor': name, 'shape': [r, c], 'mib': round(r * c * 2 / 2**20, 1),
            'calls_per_step': count,
            'native_ms': round(nat_med, 3), 'native_best_ms': round(nat_best, 3),
            'fast_ms': round(fast_med, 3), 'fast_best_ms': round(fast_best, 3),
            'speedup': round(nat_med / fast_med, 2),
            'native_verdict': list(v_native), 'fast_verdict': list(v_fast),
            'verdict_agrees': (v_native[0] > 0) == v_fast[0] and (v_native[1] > 0) == v_fast[1],
            'native_per_step_ms': round(nat_med * count, 1),
            'fast_per_step_ms': round(fast_med * count, 1),
        })
        del value
        torch.npu.empty_cache()

    native_total = sum(r['native_per_step_ms'] for r in rows)
    fast_total = sum(r['fast_per_step_ms'] for r in rows)
    out = {
        'scope': 'R184: one audit layer measured with the host sync included, native path vs a single custom-op call',
        'read_only_measurement': True, 'nothing_adopted': True, 'model_math_unchanged': True,
        'note': 'R168 showed device time alone does not predict step time, so this measures the full per-layer cost including both host syncs the strict audit performs',
        'rows': rows,
        'per_step_native_ms': round(native_total, 1),
        'per_step_fast_ms': round(fast_total, 1),
        'per_step_native_s': round(native_total / 1000.0, 3),
        'per_step_fast_s': round(fast_total / 1000.0, 3),
        'implied_saving_s_per_step': round((native_total - fast_total) / 1000.0, 3),
        'r162_measured_audit_s_per_step': 2.87,
        'all_verdicts_agree': all(r['verdict_agrees'] for r in rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('per_step_native_s', 'per_step_fast_s',
                                          'implied_saving_s_per_step', 'all_verdicts_agree')}, indent=1))
    for r in rows:
        print(f"  {r['tensor']:10s} {r['mib']:6.1f} MiB  native {r['native_ms']:7.3f} ms"
              f"  fast {r['fast_ms']:7.3f} ms  {r['speedup']:5.2f}x  agree={r['verdict_agrees']}")


if __name__ == '__main__':
    main()
