#!/usr/bin/env python3
"""Real and boundary-input validation/timing of isolated vector fusion wrappers."""
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from safetensors import safe_open
from fused_ops58 import eager_norm, fused_norm, fused_swiglu, norm, swiglu
from fused_vector58 import launch

ROOT = Path('/tmp/h3_round58_vec01')
CAP = Path('/tmp/h3_round56_capture01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
OUT = ROOT / 'vector_real_validation03.json'
if OUT.exists():
    raise RuntimeError('Refuse actual vector evidence overwrite')
torch.npu.set_device(0)
torch.set_num_threads(16)
r = {'norm_cases': [], 'activation_cases': [], 'scope': 'Same actual/boundary inputs, complete wrapper including dtype casts; isolated operators, not full model/lossless video'}


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    xf, yf = x.float(), y.float()
    finite = bool(torch.isfinite(xf).all() and torch.isfinite(yf).all())
    e = xf.double() - yf.double()
    bits_equal = x.dtype == y.dtype and torch.equal(x.view(torch.uint8), y.view(torch.uint8))
    word_diff = (x.view(torch.int16) != y.view(torch.int16)) if x.dtype == y.dtype and x.dtype in (torch.bfloat16, torch.float16) else (xf != yf)
    zero_diff = (xf == 0) & (yf == 0) & torch.signbit(xf).ne(torch.signbit(yf))
    return {'raw_bits_equal': bits_equal, 'finite': finite, 'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / yf.double().square().mean().sqrt().clamp_min(1e-30)),
            'word_mismatches': int(word_diff.sum()), 'signed_zero_mismatches': int(zero_diff.sum()), 'elements': x.numel()}


def timed(fn):
    fn(); torch.npu.synchronize()
    groups = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(5):
            value = fn()
        torch.npu.synchronize()
        groups.append((time.perf_counter() - t) * 1000 / 5)
    return value, {'groups_ms': groups, 'median_ms': statistics.median(groups)}


try:
    with torch.inference_mode():
        os.environ['VLLM_OMNI_910A_FUSED_VEC'] = '0'
        index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
        for name in ('layer49_norm1', 'layer49_norm2'):
            data = torch.load(CAP / (name + '.pt'), weights_only=True)
            xc = data['args'][0]
            rc = data['args'][1] if len(data['args']) > 1 else None
            key = 'model.language_model.layers.49.' + ('input_layernorm' if name.endswith('norm1') else 'post_attention_layernorm') + '.weight'
            with safe_open(str(MODEL / index[key]), framework='pt', device='cpu') as f:
                gamma = f.get_tensor(key).bfloat16()
            xn, gn = xc.to('npu'), gamma.to('npu')
            rn = rc.to('npu') if rc is not None else None
            reference = eager_norm(xc, gamma, rc)
            call = lambda: fused_norm(xn, gn, rn)
            t = time.perf_counter(); initial = call(); torch.npu.synchronize()
            cold = time.perf_counter() - t
            y, ft = timed(call); baseline, et = timed(lambda: eager_norm(xn, gn, rn))
            yo = y[0] if rc is not None else y
            bo = baseline[0] if rc is not None else baseline
            ref = reference[0] if rc is not None else reference
            row = {'name': name, 'shape': list(xc.shape), 'cold_build_and_call_s': cold,
                   'cpu_reference': metric(yo, ref), 'npu_eager_reference': metric(yo, bo),
                   'fused_wrapper': ft, 'eager_wrapper': et}
            if rc is not None:
                row['updated_residual'] = metric(y[1], reference[1])
                assert row['updated_residual']['raw_bits_equal']
            disabled = norm(xn, gn, rn)
            disabled_out = disabled[0] if rc is not None else disabled
            row['default_off_exact_eager'] = metric(disabled_out, bo)['raw_bits_equal']
            assert row['default_off_exact_eager'] and row['cpu_reference']['finite']
            assert row['cpu_reference']['relative_rms'] < 5e-4
            r['norm_cases'].append(row)
            OUT.write_text(json.dumps(r, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        torch.manual_seed(58)
        # BF16 range beyond FP16, zeros and residual rounding boundaries.
        xc = torch.randn(9, 5120).bfloat16()
        xc[0].zero_(); xc[1].fill_(1.); xc[2, 731] = 20736; xc[3, 17] = 131072
        xc[4].fill_(2.0**-100); xc[5, :8] = torch.tensor([-0., 0., -.5, .5, -1., 1., -256., 256.]).bfloat16()
        gc = torch.randn(5120).bfloat16(); rc = torch.full_like(xc, 0.003936767578125)
        for add in (False, True):
            ref = eager_norm(xc, gc, rc if add else None)
            got = fused_norm(xc.to('npu'), gc.to('npu'), rc.to('npu') if add else None)
            torch.npu.synchronize()
            row = {'name': 'bf16_boundaries_add' if add else 'bf16_boundaries_plain',
                   'output': metric(got[0] if add else got, ref[0] if add else ref)}
            if add:
                row['residual'] = metric(got[1], ref[1]); assert row['residual']['raw_bits_equal']
            assert row['output']['finite'] and row['output']['relative_rms'] < 5e-4
            r['norm_cases'].append(row)
        # Shared DiT-style contract: FP32 normalization/gamma before output cast.
        for dtype in (torch.float16, torch.float32):
            xc = torch.randn(17, 3072).to(dtype); gc = torch.randn(3072).to(dtype)
            for add in (False, True):
                rc = torch.randn_like(xc) * .1 if add else None
                ref = eager_norm(xc, gc, rc, qwen=False)
                got = fused_norm(xc.to('npu'), gc.to('npu'), rc.to('npu') if add else None, qwen=False)
                torch.npu.synchronize()
                row = {'name': str(dtype) + ('_add' if add else '_plain'),
                       'output': metric(got[0] if add else got, ref[0] if add else ref)}
                if add:
                    row['residual'] = metric(got[1], ref[1]); assert row['residual']['raw_bits_equal']
                assert row['output']['finite'] and row['output']['relative_rms'] < 5e-4
                r['norm_cases'].append(row)
        packed = torch.load(CAP / 'layer49_gate_up.pt', weights_only=True)['output']
        half = packed.shape[-1] // 2
        gatec, upc = packed[..., :half], packed[..., half:]
        packedn = packed.to('npu'); gate, up = packedn[..., :half], packedn[..., half:]
        ref = torch.nn.functional.silu(gatec) * upc
        t = time.perf_counter(); fused_swiglu(gate, up); torch.npu.synchronize(); cold = time.perf_counter() - t
        y, ft = timed(lambda: fused_swiglu(gate, up)); base, et = timed(lambda: torch.nn.functional.silu(gate) * up)
        row = {'name': 'actual_layer49_packed_swiglu', 'packed_shape': list(packed.shape), 'cold_build_and_call_s': cold,
               'cpu_reference': metric(y, ref), 'npu_eager_reference': metric(y, base), 'fused_wrapper': ft, 'eager_wrapper': et,
               'default_off_exact_eager': metric(swiglu(gate, up), base)['raw_bits_equal']}
        assert row['cpu_reference']['finite'] and row['cpu_reference']['relative_rms'] < 5e-4 and row['default_off_exact_eager']
        r['activation_cases'].append(row); print(json.dumps(row), flush=True)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            gatec = torch.randn(13, 3200).to(dtype); upc = torch.randn_like(gatec)
            gatec[0, :8] = torch.tensor([-100., -80., -0., 0., .5, 20., 80., 100.]).to(dtype)
            gatec[1, 731] = 20736
            ref = torch.nn.functional.silu(gatec) * upc
            y = fused_swiglu(gatec.to('npu'), upc.to('npu')); torch.npu.synchronize()
            row = {'name': str(dtype) + '_swiglu_boundaries', 'output': metric(y, ref)}
            assert row['output']['finite'] and row['output']['relative_rms'] < 5e-4
            r['activation_cases'].append(row)
        r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)
    print('Evidence:', OUT, flush=True)
