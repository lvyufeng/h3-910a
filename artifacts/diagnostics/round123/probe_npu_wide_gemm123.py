#!/usr/bin/env python3
"""910A capability-only check for GPU-matching FP16-input FP32-output GEMM.

No CPU inference baseline, no BF16 execution, no weight download, no shared-runtime
change. Compare only a small exact known product on the owned 910A. A native Half
output cast is intentionally shown separately; it is not an FP32 accumulator.
"""
import json
from pathlib import Path
import time
import torch
import torch_npu

ROOT = Path('/tmp/h3_align12301')


def info(x):
    y = x.detach().float().cpu()
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'device': str(x.device),
            'finite': bool(torch.isfinite(y).all()), 'peak': float(y.abs().max()),
            'all_equal_exact_128000': bool((y == 128000.).all())}


def main():
    path = ROOT / 'NPU_F16_input_F32_output_capability123.json'
    assert not path.exists()
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(4)
    x = torch.full((16, 128), 100., dtype=torch.float16, device='npu:0')
    w = torch.full((32, 128), 10., dtype=torch.float16, device='npu:0')
    r = {'scope': __doc__, 'completed': False, 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
         'expected_exact_output': 128000., 'input_dtypes': [str(x.dtype), str(w.dtype)],
         'native_functions': {}, 'cases': {}}
    for name in ['npu_mm', 'npu_matmul', 'npu_bmmV2', 'npu_grouped_matmul', 'npu_linear', 'npu_transpose_batchmatmul']:
        try:
            r['native_functions'][name] = {str(k): str(v) for k, v in getattr(torch.ops.npu, name)._schemas.items()}
        except (AttributeError, RuntimeError):
            r['native_functions'][name] = None
    cases = [
        ('torch_mm_native_Half_output', lambda: torch.mm(x, w.t())),
        ('torch_mm_out_dtype_FP32', lambda: torch.mm(x, w.t(), out_dtype=torch.float32)),
        ('torch_mm_FP32_output_buffer', lambda: torch.mm(x, w.t(), out=torch.empty((16, 32), dtype=torch.float32, device='npu:0'))),
        ('npu_grouped_matmul_FP32_output', lambda: torch_npu.npu_grouped_matmul([x], [w.t().contiguous()], split_item=0, group_type=-1, output_dtype=torch.float32)[0]),
    ]
    for name, fn in cases:
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                y = fn()
            torch.npu.synchronize()
            r['cases'][name] = {'supported': True, 'output': info(y), 'elapsed_s': time.perf_counter() - start}
        except Exception as exc:
            r['cases'][name] = {'supported': False, 'error_type': type(exc).__name__, 'message': str(exc)[-2500:]}
        path.write_text(json.dumps(r, indent=2) + '\n')
    r['completed'] = True
    r['native_GPU_matching_wide_GEMM_supported'] = any(row.get('supported') and row['output']['dtype'] == 'torch.float32' and row['output']['all_equal_exact_128000'] for row in r['cases'].values())
    path.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)


if __name__ == '__main__':
    main()
