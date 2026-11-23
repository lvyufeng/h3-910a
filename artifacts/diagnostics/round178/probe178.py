#!/usr/bin/env python3
"""R178: how fast can a single pass over the audited tensors possibly be?

Stage A proposes replacing the per-call safety scan with one fused pass. Its whole value is
decided by the achievable time for one pass over the same bytes, so this measures that with
existing CANN reductions instead of waiting on a custom-op build.

For each audited shape it times one full-tensor reduction (torch.amax), which reads exactly
the bytes a hand-written statistics kernel would read. It then converts the per-call times
into a per-step total using the measured call counts (50 qkv_proj, 50 mlp.fc1, 55 adaln) and
compares that against the 2.87 s the current audit costs per step.
"""
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/h3_bench178/proxy178.json')

SHAPES = [
    ('mlp_fc1', (10448, 14336), 50),
    ('qkv_proj', (10448, 10752), 50),
    ('adaln_linear', (3, 96768), 55),
]
WARMUP, REPEATS = 3, 10


def time_call(fn, repeats):
    times = []
    for i in range(WARMUP + repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        if i >= WARMUP:
            times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


def main():
    torch.npu.set_device(0)
    rows = []
    total_ms = 0.0
    for name, shape, count in SHAPES:
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * 2                      # fp16
        x = torch.randn(shape, dtype=torch.float16, device='npu')

        # flat amax trips ReduceMax with a wrong output buffer on this CANN build (same class of
        # failure as R148); a dim-specified reduction works and still reads the whole tensor once
        dims = tuple(range(x.dim()))
        med, best = time_call(lambda: torch.amax(x, dim=dims), REPEATS)
        gbs = nbytes / (med / 1000.0) / 1e9
        per_step = med * count
        total_ms += per_step
        rows.append({'tensor': name, 'shape': list(shape), 'mib': round(nbytes / 2**20, 1),
                     'calls_per_step': count, 'median_ms': round(med, 3), 'best_ms': round(best, 3),
                     'effective_gbs': round(gbs, 1),
                     'per_step_ms': round(per_step, 1)})
        del x
        torch.npu.empty_cache()

    total_bytes = sum(1 for _ in ())            # placeholder, computed below
    per_step_bytes = sum((r['mib'] * 2**20) * r['calls_per_step'] for r in rows)
    median_of_mibs = [r['median_ms'] / r['mib'] for r in rows if r['mib'] > 10]
    out = {
        'scope': 'R178 proxy measurement: cost of one full pass over the audited tensors, using existing CANN reductions',
        'read_only_measurement': True, 'no_custom_op_needed': True,
        'model_arithmetic_unchanged': True, 'nothing_adopted': True,
        'device': 'npu:0', 'dtype': 'float16', 'op_used_as_proxy': 'torch.amax(x, dim=all) (one full-tensor read); flat amax fails on this CANN build',
        'warmup': WARMUP, 'repeats': REPEATS,
        'rows': rows,
        'per_step_total_ms': round(total_ms, 1),
        'per_step_total_s': round(total_ms / 1000.0, 3),
        'per_step_bytes_gib': round(per_step_bytes / 2**30, 2),
        'current_audit_s_per_step': 2.87,
        'current_audit_ms_per_call': 18.5,
        'implied_saving_s_per_step': round(2.87 - total_ms / 1000.0, 3),
        'verdict': None,
        'caveats': [
            'amax is a single reduction; a statistics kernel that also counts bit patterns may cost more',
            'the proxy reads the same bytes but does not reproduce the exact predicate work',
            'this bounds what stage A can save; it does not prove any kernel achieves it'],
    }
    saving = out['implied_saving_s_per_step']
    if saving > 2.0:
        verdict = 'stage A remains worth pursuing: a single pass is much cheaper than the current audit'
    elif saving > 0.8:
        verdict = 'stage A is worth about ' + str(round(saving, 2)) + ' s, less than the 2.5-2.9 s previously assumed'
    else:
        verdict = 'stage A is not worth the build cost: a single pass would save only ' + str(round(saving, 2)) + ' s'
    out['verdict'] = verdict
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('per_step_total_s', 'implied_saving_s_per_step', 'verdict')}, indent=1))
    for r in rows:
        print(' ', r['tensor'], r['mib'], 'MiB', r['median_ms'], 'ms', r['effective_gbs'], 'GB/s',
              '->', r['per_step_ms'], 'ms/step')


if __name__ == '__main__':
    main()
