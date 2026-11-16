#!/usr/bin/env python3
"""Exhaustive raw BF16 codec checks, including every exponent/NaN bit pattern."""
import json
from pathlib import Path
import numpy as np
import torch
import torch_npu
from native_bf16_vector58 import native_launch

ROOT = Path('/tmp/h3_round58_vec01')
OUT = ROOT / 'native_codec_validation.json'
if OUT.exists():
    raise RuntimeError('Refuse native codec evidence overwrite')
torch.npu.set_device(0)
r = {'cases': [], 'scope': 'Native uint16 codec preserves every BF16 bit pattern; arithmetic nonfinite behavior is a separate contract'}
try:
    with torch.inference_mode():
        for rows, width in ((512, 128), (26, 3200), (16, 5120)):
            bits = (np.arange(rows * width, dtype=np.uint32) % 65536).astype(np.uint16).reshape(rows, width)
            cpu = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
            expected = torch.from_numpy((bits.astype(np.uint32) << 16).view(np.float32).copy())
            unpacked, repacked = native_launch('codec', cpu.to('npu'))
            torch.npu.synchronize()
            a, b = unpacked.cpu().contiguous(), repacked.cpu().contiguous()
            item = {'rows': rows, 'width': width, 'padded_width': ((width + 255) // 256) * 256,
                    'unique_input_words': int(np.unique(bits).size),
                    'unpacked_bits_exact': torch.equal(a.view(torch.uint8), expected.view(torch.uint8)),
                    'repacked_bits_exact': torch.equal(b.view(torch.uint8), cpu.view(torch.uint8))}
            assert item['unpacked_bits_exact'] and item['repacked_bits_exact']
            r['cases'].append(item); print(json.dumps(item), flush=True)
        r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)
