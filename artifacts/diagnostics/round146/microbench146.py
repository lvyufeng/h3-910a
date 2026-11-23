#!/usr/bin/env python3
"""R146: standalone GEMM at the TRUE in-model shapes measured by R144.

Measurement only, no model weights, no math change. R142 rescaled its numbers by
FLOP ratio to guess these shapes; that rescaling is not valid for mm, so this run
measures the real shapes directly.
"""
from __future__ import annotations
import json, os, statistics
from pathlib import Path

ROOT = Path('/tmp/h3_micro146')
OUT = ROOT / 'receipts' / 'micro146_rank{rank}.json'
PEAK_FP16 = 310e12
M = 10448
# (case, K, N, out_dtype, fp32_input)
CASES = [
    ('qkv_proj_true', 5376, 10752, 'float16', False),
    ('mlp_fc1_true', 5376, 14336, 'float32', False),
    ('attn_out_proj_true', 3584, 5376, 'float32', False),
    ('mlp_fc2_true_fp32_input', 7168, 5376, 'float32', True),
]


def timed(torch, fn, warmup=3, repeats=8):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
        start.record(); fn(); end.record(); torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(samples), min(samples)


def main():
    import torch, torch_npu
    rank = int(os.environ.get('RANK', '0'))
    torch.npu.set_device(rank)
    device = torch.device(f'npu:{rank}')
    report = {'scope': __doc__, 'completed': False, 'rank': rank, 'm_tokens': M,
              'peak_assumed_FP16_TFLOPS': PEAK_FP16 / 1e12, 'no_model_weights_loaded': True}
    try:
        rows = []
        for name, k, n, out_dtype, fp32_in in CASES:
            dtype = torch.float32 if fp32_in else torch.float16
            x = (torch.randn(M, k, dtype=dtype, device=device) * 0.05)
            w = torch.randn(n, k, dtype=torch.float16, device=device) * 0.05
            od = torch.float32 if out_dtype == 'float32' else torch.float16
            if fp32_in:
                def run():
                    return torch.mm(x.to(torch.float16), w.t(), out_dtype=od)
            else:
                def run():
                    return torch.mm(x, w.t(), out_dtype=od)
            med, best = timed(torch, run)
            flops = 2 * M * k * n
            rows.append({'case': name, 'm': M, 'k': k, 'n': n,
                         'input_dtype': 'float32->float16' if fp32_in else 'float16',
                         'out_dtype': out_dtype,
                         'median_ms': round(med * 1000, 3), 'min_ms': round(best * 1000, 3),
                         'flops_TF': round(flops / 1e12, 3),
                         'TFLOPS_per_card': round(flops / med / 1e12, 1),
                         'percent_of_FP16_peak': round(100 * flops / med / PEAK_FP16, 1)})
        report['gemm'] = rows
        # in-model per-layer region times from R141 for direct comparison
        in_model_ms = {'qkv_proj_true': 41.74, 'mlp_fc1_true': 54.60,
                       'attn_out_proj_true': 19.02, 'mlp_fc2_true_fp32_input': 30.30}
        for row in rows:
            row['in_model_ms_from_R141'] = in_model_ms[row['case']]
            row['gap_x'] = round(row['in_model_ms_from_R141'] / row['median_ms'], 2)
        report['completed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:400]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Path(str(OUT).format(rank=rank)).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'rank': rank, 'completed': report['completed'],
                      'failure': str(report.get('failure'))[:160]}, flush=True))


if __name__ == '__main__':
    main()
