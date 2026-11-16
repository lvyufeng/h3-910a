#!/usr/bin/env python3
"""Finite BF16 SwiGLU: distinguish CPU overflow behavior from actual NPU reference."""
import json
from pathlib import Path
import numpy as np
import torch
import torch_npu
from fused_vector58 import launch
from native_bf16_vector58 import native_launch

ROOT = Path('/tmp/h3_round58_vec01')
OUT = ROOT / 'exhaustive_npu_reference03.json'
if OUT.exists():
    raise RuntimeError('Refuse NPU-reference exhaustive evidence overwrite')
torch.npu.set_device(0)
r = {'scope': 'All finite BF16 gate words times eight up factors, explicitly actual NPU reference; no nonfinite-input or full-model claim', 'cases': []}


def compare(a, b):
    a, b = a.cpu().contiguous(), b.cpu().contiguous()
    af, bf = a.float(), b.float()
    different = a.view(torch.int16).ne(b.view(torch.int16))
    equal_numeric = af.eq(bf) | (torch.isnan(af) & torch.isnan(bf))
    zeros = af.eq(0) & bf.eq(0)
    return {'raw_bits_exact': not bool(different.any()), 'word_mismatches': int(different.sum()),
            'numeric_mismatches': int((~equal_numeric).sum()),
            'signed_zero_mismatches': int((different & zeros).sum()),
            'infinity_outputs': int(torch.isinf(bf).sum()), 'nan_outputs': int(torch.isnan(bf).sum())}


try:
    with torch.inference_mode():
        words = np.arange(65536, dtype=np.uint32)
        finite = words[(words & 0x7f80) != 0x7f80]
        bits = np.concatenate([(finite << 16) | low for low in (0x7fff, 0x8000, 0x8001)]).astype(np.uint32)
        x = torch.from_numpy(bits.view(np.float32).copy()).reshape(-1, 128)
        ref = x.bfloat16().float()
        y, _ = launch('round', x.to('npu'), torch.ones(128, device='npu')); torch.npu.synchronize()
        exact = torch.equal(y.cpu().contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8))
        r['cases'].append({'name': 'finite_bf16_midpoints', 'inputs': x.numel(), 'raw_bits_exact': exact})
        assert exact
        gates = torch.from_numpy(finite.astype(np.uint16).view(np.int16).copy()).view(torch.bfloat16)
        factors = torch.tensor([1., -1., 0., -0., 2.0**-133, 1e20, -1e20, torch.finfo(torch.bfloat16).max], dtype=torch.bfloat16)
        gate = gates.repeat(len(factors)).reshape(-1, 128)
        up = factors.repeat_interleave(len(gates)).reshape(-1, 128)
        packed = torch.cat((gate, up), -1); pn = packed.to('npu')
        cpu_ref = torch.nn.functional.silu(gate) * up
        gpu_ref = torch.nn.functional.silu(pn[..., :128]) * pn[..., 128:]
        y, _ = native_launch('swiglu', pn); torch.npu.synchronize()
        row = {'name': 'finite_bf16_gate_up_sweep', 'gate_words': len(gates), 'up_factors': len(factors), 'outputs': y.numel(),
               'input_format': torch_npu.get_npu_format(pn), 'vs_cpu': compare(y, cpu_ref), 'vs_actual_npu': compare(y, gpu_ref)}
        cpu_s = torch.nn.functional.silu(gates)
        npu_s = torch.nn.functional.silu(gates.to('npu')).cpu()
        row['cpu_silu_vs_npu_silu'] = compare(cpu_s, npu_s)
        mask = y.cpu().float().ne(gpu_ref.cpu().float()).reshape(-1)
        row['npu_numeric_examples'] = []
        for index in torch.nonzero(mask).reshape(-1)[:32].tolist():
            row['npu_numeric_examples'].append({'gate': float(gate.reshape(-1)[index]), 'up': float(up.reshape(-1)[index]),
                                                 'native': float(y.cpu().reshape(-1)[index]), 'npu_ref': float(gpu_ref.cpu().reshape(-1)[index])})
        r['cases'].append(row)
        r['completed'] = True
        r['npu_raw_bits_exact'] = row['vs_actual_npu']['raw_bits_exact']
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)
