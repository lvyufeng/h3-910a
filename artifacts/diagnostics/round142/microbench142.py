#!/usr/bin/env python3
"""R142: decide whether in-model GEMM/FA slowness is staging-wait or kernel efficiency.

Measurement only, standalone: no model weights are loaded, no model math is changed,
nothing is installed into the model. Shapes are taken from the measured R141 roofline
(M = tokens per rank per USP shard, K/N from the public architecture config).
"""
from __future__ import annotations
import json, os, statistics, time
from pathlib import Path

ROOT = Path('/tmp/h3_micro142')
OUT = ROOT / 'receipts' / 'microbench142_rank{rank}.json'

N_TOKENS = 41792
USP = 4
TOKENS_LOCAL = N_TOKENS // USP
D, HEADS, HD, FFN, TP = 5376, 56, 128, 14336, 2
QKV_OUT = HEADS * HD                      # 7168, column parallel
FFN_OUT = FFN                             # merged gate/up is 2*FFN, handled below
PEAK_FP16 = 310e12


def timed(torch, fn, warmup=3, repeats=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record(); fn(); end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return {'median_s': statistics.median(samples), 'min_s': min(samples), 'samples_s': [round(s, 5) for s in samples]}


def gemm_case(torch, device, name, m, k, n, out_dtype, with_prefetch=False, prefetch_bytes=None):
    x = torch.randn(m, k, dtype=torch.float16, device=device) * 0.05
    w = torch.randn(n, k, dtype=torch.float16, device=device) * 0.05
    host_master = torch.empty(n, k, dtype=torch.float16, pin_memory=True) if with_prefetch else None
    def run():
        if with_prefetch:
            w.copy_(host_master, non_blocking=True)
        return torch.mm(x, w.t(), out_dtype=out_dtype)
    row = timed(torch, run)
    flops = 2 * m * k * n
    tf = flops / row['median_s'] / 1e12
    return {'case': name, 'm': m, 'k': k, 'n': n, 'out_dtype': str(out_dtype).replace('torch.', ''),
            'weight_prefetch_per_call_MiB': (round(n * k * 2 / 2**20, 1) if with_prefetch else None),
            'median_s': round(row['median_s'], 6), 'achieved_TFLOPS_per_card': round(tf, 1),
            'percent_of_FP16_peak': round(100 * flops / row['median_s'] / PEAK_FP16, 1),
            'flops_TF': round(flops / 1e12, 2)}


def guard_split(torch, device, name, m, k, n):
    """Reproduce the native range-guard prologue/epilogue and time each stage."""
    import math
    x = torch.randn(m, k, dtype=torch.float16, device=device) * 0.05
    w = torch.randn(n, k, dtype=torch.float16, device=device) * 0.05
    def prologue():
        # Exactly the native prologue shape: FP32 upcast, whole-tensor abs max held
        # on host, power-of-two scale, divide and narrow back to FP16.
        product = x.float()
        peak = float(product.abs().amax())
        exponent = max(0, math.ceil(math.log2(peak / 32752.0))) if peak > 32752.0 else 0
        scale = float(2 ** exponent)
        return (product / scale).to(torch.float16), scale
    def mm_only():
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    pro = timed(torch, prologue)
    mm = timed(torch, mm_only)
    half_input, scale = prologue()
    def epilogue():
        result = torch.mm(half_input, w.t(), out_dtype=torch.float32) * scale
        return result + torch.zeros(n, dtype=torch.float32, device=device)
    epi = timed(torch, epilogue)
    return {'case': name, 'm': m, 'k': k, 'n': n, 'scale_used': scale,
            'prologue_s': round(pro['median_s'], 6), 'mm_only_s': round(mm['median_s'], 6),
            'mm_plus_epilogue_s': round(epi['median_s'], 6),
            'prologue_share_of_full_percent': round(100 * pro['median_s'] / epi['median_s'], 1) if epi['median_s'] else None}


def fa_case(torch, device):
    """Time the real 910A FA backend on the measured shape; no weights involved."""
    from vllm_omni.diffusion.attention.backends import fa910a
    import inspect, hashlib
    src = Path(inspect.getfile(fa910a))
    b, h, n, d = 1, 7, 43776, 128
    q = torch.randn(b, h, n, d, dtype=torch.float16, device=device) * 0.05
    k = torch.randn(b, h, n, d, dtype=torch.float16, device=device) * 0.05
    v = torch.randn(b, h, n, d, dtype=torch.float16, device=device) * 0.05
    fn = None
    for attr in ('flash_attn_varlen_func', 'flash_attn_func', 'attn_forward'):
        if hasattr(fa910a, attr):
            fn = getattr(fa910a, attr); break
    if fn is None:
        return {'case': 'fa910a', 'skipped': True, 'reason': 'no callable entry point found',
                'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                'public_names': [n for n in dir(fa910a) if not n.startswith('_')][:20]}
    row = timed(torch, lambda: fn(q, k, v, causal=False), warmup=2, repeats=5)
    flops = 4 * b * h * n * n * d          # QK^T + P@V
    tf = flops / row['median_s'] / 1e12
    return {'case': 'fa910a_real_shape', 'b': b, 'h': h, 'n': n, 'd': d, 'causal': False,
            'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
            'median_s': round(row['median_s'], 6), 'flops_TF': round(flops / 1e12, 2),
            'achieved_TFLOPS_per_card': round(tf, 1), 'percent_of_FP16_peak': round(100 * flops / row['median_s'] / PEAK_FP16, 1),
            'measured_in_model_kernel_s_R126': 7.454}


def main():
    import torch, torch_npu
    rank = int(os.environ.get('RANK', '0'))
    torch.npu.set_device(rank)
    device = torch.device(f'npu:{rank}')
    global OUT
    OUT = Path(str(OUT).format(rank=rank))
    report = {'scope': __doc__, 'completed': False, 'rank': rank, 'tokens_local': TOKENS_LOCAL,
              'peak_assumed_FP16_TFLOPS': PEAK_FP16 / 1e12, 'no_model_weights_loaded': True}
    try:
        cases = [
            gemm_case(torch, device, 'qkv_proj_resident', TOKENS_LOCAL, D, QKV_OUT, torch.float16),
            gemm_case(torch, device, 'qkv_proj_with_weight_prefetch', TOKENS_LOCAL, D, QKV_OUT, torch.float16, True),
            gemm_case(torch, device, 'mlp_fc1_resident', TOKENS_LOCAL, D, 2 * FFN_OUT, torch.float16),
            gemm_case(torch, device, 'mlp_fc1_with_weight_prefetch', TOKENS_LOCAL, D, 2 * FFN_OUT, torch.float16, True),
            gemm_case(torch, device, 'attn_out_proj_resident_fp32out', TOKENS_LOCAL, QKV_OUT, D, torch.float32),
            gemm_case(torch, device, 'mlp_fc2_resident_fp32out', TOKENS_LOCAL, FFN_OUT, D, torch.float32),
        ]
        report['gemm'] = cases
        guards = [guard_split(torch, device, 'attn_out_proj_guard', TOKENS_LOCAL, QKV_OUT, D),
                  guard_split(torch, device, 'mlp_fc2_guard', TOKENS_LOCAL, FFN_OUT, D)]
        report['guard_split'] = guards
        report['gemm'] = cases
        report['fa'] = fa_case(torch, device)
        report['completed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'rank': rank, 'completed': report['completed'],
                      'failure': str(report.get('failure'))[:200]}, flush=True))


if __name__ == '__main__':
    main()
