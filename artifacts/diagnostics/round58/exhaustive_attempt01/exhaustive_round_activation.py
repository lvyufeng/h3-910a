#!/usr/bin/env python3
"""Finite-BF16 activation sweep and exact software RNE boundary sweep."""
import json
from pathlib import Path
import numpy as np
import torch
import torch_npu
from fused_vector58 import launch
from native_bf16_vector58 import native_launch

ROOT = Path('/tmp/h3_round58_vec01')
OUT = ROOT / 'exhaustive_round_activation.json'
if OUT.exists():
    raise RuntimeError('Refuse exhaustive vector proof overwrite')
torch.npu.set_device(0)
r = {'scope': 'All finite BF16 gate words with representative up factors and all finite BF16 midpoint rounding boundaries; not every pair/full model', 'cases': []}
try:
    with torch.inference_mode():
        words = np.arange(65536, dtype=np.uint32)
        finite = words[(words & 0x7f80) != 0x7f80]
        # Exact BF16 halfway boundaries, just below and just above, both signs.
        bits = np.concatenate([(finite << 16) | low for low in (0x7fff, 0x8000, 0x8001)]).astype(np.uint32)
        x = torch.from_numpy(bits.view(np.float32).copy()).reshape(-1, 128)
        expected = x.bfloat16().float()
        y, _ = launch('round', x.to('npu'), torch.ones(128, device='npu'))
        torch.npu.synchronize()
        got = y.cpu().contiguous()
        item = {'name': 'finite_bf16_midpoint_sweep', 'input_count': x.numel(), 'raw_bits_exact': torch.equal(got.view(torch.uint8), expected.view(torch.uint8)),
                'finite_input_word_count': int(finite.size), 'output_infinity_count': int(torch.isinf(expected).sum())}
        assert item['raw_bits_exact']; r['cases'].append(item); print(json.dumps(item), flush=True)
        gates = torch.from_numpy(finite.astype(np.uint16).view(np.int16).copy()).view(torch.bfloat16)
        factors = torch.tensor([1., -1., 0., -0., 2.0**-133, 1e20, -1e20, torch.finfo(torch.bfloat16).max], dtype=torch.bfloat16)
        gate = gates.repeat(len(factors)).reshape(-1, 128)
        up = factors.repeat_interleave(len(gates)).reshape(-1, 128).contiguous()
        ref = torch.nn.functional.silu(gate) * up
        packed = torch.cat((gate, up), -1)
        y, _ = native_launch('swiglu', packed.to('npu')); torch.npu.synchronize()
        got = y.cpu().contiguous()
        item = {'name': 'all_finite_gate_words_representative_up', 'gate_word_count': len(gates), 'up_factor_count': len(factors),
                'output_count': got.numel(), 'raw_bits_exact': torch.equal(got.view(torch.uint8), ref.contiguous().view(torch.uint8)),
                'word_mismatches': int(got.view(torch.int16).ne(ref.contiguous().view(torch.int16)).sum()),
                'infinity_outputs_valid': int(torch.isinf(ref.float()).sum()), 'input_nonfinite_count': int((~torch.isfinite(packed.float())).sum())}
        r['cases'].append(item); print(json.dumps(item), flush=True)
        # Do not conflate an executed sweep with bitwise acceptance.
        r['completed'] = True
        r['all_cases_raw_bits_exact'] = all(x['raw_bits_exact'] for x in r['cases'])
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)
