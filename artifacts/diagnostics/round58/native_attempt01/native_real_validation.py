#!/usr/bin/env python3
"""Same actual inputs: native BF16-GM fused kernels vs complete eager references."""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from safetensors import safe_open
from fused_ops58 import eager_norm
from native_bf16_vector58 import native_launch

ROOT = Path('/tmp/h3_round58_vec01')
CAP = Path('/tmp/h3_round56_capture01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
OUT = ROOT / 'native_real_validation.json'
if OUT.exists():
    raise RuntimeError('Refuse native real evidence overwrite')
torch.npu.set_device(0)
torch.set_num_threads(16)
r = {'norm_cases': [], 'activation_cases': [], 'boundary_cases': [],
     'scope': 'Native BF16 GM codec plus FP32 math on actual/boundary inputs; no external casts, isolated operators not whole quality/lossless proof'}


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    xf, yf = x.float(), y.float()
    e = xf.double() - yf.double()
    return {'raw_bits_equal': x.dtype == y.dtype and torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'finite': bool(torch.isfinite(xf).all() and torch.isfinite(yf).all()),
            'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / yf.double().square().mean().sqrt().clamp_min(1e-100)),
            'word_mismatches': int((x.view(torch.int16) != y.view(torch.int16)).sum()),
            'signed_zero_mismatches': int(((xf == 0) & (yf == 0) & torch.signbit(xf).ne(torch.signbit(yf))).sum()),
            'elements': x.numel()}


def timed(fn):
    fn(); torch.npu.synchronize(); times = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(10):
            y = fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t) * 1000 / 10)
    return y, {'groups_ms': times, 'median_ms': statistics.median(times)}


def nrm(x, g, residual=None):
    y, updated = native_launch('addrms' if residual is not None else 'rms', x.contiguous(), g.contiguous(),
                                residual.contiguous() if residual is not None else None)
    return (y, updated) if residual is not None else y


try:
    with torch.inference_mode():
        index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
        for name in ('layer49_norm1', 'layer49_norm2'):
            d = torch.load(CAP / (name + '.pt'), weights_only=True)
            xc, rc = d['args'][0], d['args'][1] if len(d['args']) > 1 else None
            key = 'model.language_model.layers.49.' + ('input_layernorm' if name.endswith('norm1') else 'post_attention_layernorm') + '.weight'
            with safe_open(str(MODEL / index[key]), framework='pt', device='cpu') as f:
                gc = f.get_tensor(key).bfloat16()
            x, g, residual = xc.to('npu'), gc.to('npu'), rc.to('npu') if rc is not None else None
            ref = eager_norm(xc, gc, rc)
            t = time.perf_counter(); nrm(x, g, residual); torch.npu.synchronize(); cold = time.perf_counter() - t
            y, nt = timed(lambda: nrm(x, g, residual)); eager, et = timed(lambda: eager_norm(x, g, residual))
            row = {'name': name, 'shape': list(xc.shape), 'cold_compile_and_call_s': cold,
                   'cpu_reference': metric(y[0] if rc is not None else y, ref[0] if rc is not None else ref),
                   'npu_eager_reference': metric(y[0] if rc is not None else y, eager[0] if rc is not None else eager),
                   'native_wrapper': nt, 'eager_wrapper': et}
            if rc is not None:
                row['updated_residual'] = metric(y[1], ref[1]); assert row['updated_residual']['raw_bits_equal']
            assert row['cpu_reference']['finite'] and row['cpu_reference']['relative_rms'] < 5e-4
            r['norm_cases'].append(row); print(json.dumps(row), flush=True)
            OUT.write_text(json.dumps(r, indent=2) + '\n')
        packed = torch.load(CAP / 'layer49_gate_up.pt', weights_only=True)['output']
        half = packed.shape[-1] // 2
        pn = packed.to('npu')
        ref = torch.nn.functional.silu(packed[..., :half]) * packed[..., half:]
        t = time.perf_counter(); native_launch('swiglu', pn); torch.npu.synchronize(); cold = time.perf_counter() - t
        y, nt = timed(lambda: native_launch('swiglu', pn)[0])
        eager, et = timed(lambda: torch.nn.functional.silu(pn[..., :half]) * pn[..., half:])
        row = {'name': 'actual_layer49_packed_swiglu', 'packed_shape': list(packed.shape), 'cold_compile_and_call_s': cold,
               'cpu_reference': metric(y, ref), 'npu_eager_reference': metric(y, eager), 'native_wrapper': nt, 'eager_wrapper': et}
        assert row['cpu_reference']['finite'] and row['cpu_reference']['relative_rms'] < 5e-4
        r['activation_cases'].append(row); print(json.dumps(row), flush=True)
        # This is real post-RoPE Q reused as a same-shape RMSNorm input, NOT a
        # capture of actual pre-QNorm values or a whole-layer comparison.
        qc = torch.load(Path('/tmp/h3_round57_text01/attention_layer_49.pt'), weights_only=True)['q'].transpose(1, 2).contiguous()
        key = 'model.language_model.layers.49.self_attn.q_norm.weight'
        with safe_open(str(MODEL / index[key]), framework='pt', device='cpu') as f:
            gc = f.get_tensor(key).bfloat16()
        q, g = qc.to('npu'), gc.to('npu')
        y, nt = timed(lambda: nrm(q, g)); eager, et = timed(lambda: eager_norm(q, g))
        ref = eager_norm(qc, gc)
        row = {'name': 'post_rope_query_reused_as_same_shape_norm_input', 'shape': list(qc.shape),
               'cpu_reference': metric(y, ref), 'npu_eager_reference': metric(y, eager), 'native_wrapper': nt, 'eager_wrapper': et}
        assert row['cpu_reference']['finite'] and row['cpu_reference']['relative_rms'] < 5e-4
        r['norm_cases'].append(row); print(json.dumps(row), flush=True)
        torch.manual_seed(580)
        xc = torch.randn(16, 128).bfloat16(); rc = torch.randn_like(xc) * .01; gc = torch.randn(128).bfloat16()
        xc[0].zero_(); rc[0].zero_()
        xc[1].fill_(-0.); rc[1].fill_(-0.)
        xc[2].fill_(131072); xc[3, 17] = 20736
        xc[4].fill_(1e20); xc[5].fill_(-1e20)
        xc[6].fill_(2.0**-100); xc[7].fill_(2.0**-133)
        for add in (False, True):
            ref = eager_norm(xc, gc, rc if add else None)
            got = nrm(xc.to('npu'), gc.to('npu'), rc.to('npu') if add else None); torch.npu.synchronize()
            row = {'name': 'bf16_range_signed_zero_overflow_add' if add else 'bf16_range_signed_zero_overflow_plain',
                   'output': metric(got[0] if add else got, ref[0] if add else ref)}
            if add:
                row['residual'] = metric(got[1], ref[1])
            r['boundary_cases'].append(row)
        gate = torch.randn(16, 128).bfloat16(); up = torch.randn_like(gate)
        gate[0, :8] = torch.tensor([-100., -80., -0., 0., .5, 20., 80., 100.]).bfloat16()
        packed = torch.cat((gate, up), dim=-1)
        ref = torch.nn.functional.silu(gate) * up
        y, _ = native_launch('swiglu', packed.to('npu')); torch.npu.synchronize()
        r['boundary_cases'].append({'name': 'bf16_swiglu_signed_zero_outliers', 'output': metric(y, ref)})
        r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)
    print('Evidence:', OUT, flush=True)
