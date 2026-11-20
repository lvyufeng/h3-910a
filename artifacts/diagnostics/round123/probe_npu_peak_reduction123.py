#!/usr/bin/env python3
"""Check explicit-axis peak reductions on 910A for the exact refiner projection geometry."""
import json
from pathlib import Path
import torch
import torch_npu

ROOT = Path('/tmp/h3_align12301')


def main():
    p = ROOT / 'NPU_explicit_axis_peak_probe123.json'
    assert not p.exists()
    torch.npu.set_device(0)
    x = torch.full((2043, 3584), 0.25, dtype=torch.float32, device='npu:0')
    x[123, 456] = -92795.5
    r = {'scope': __doc__, 'completed': False, 'cases': {}, 'CPU_inference': False,
         'expected_exact_absmax': 92795.5, 'geometry': list(x.shape)}
    for name, function in [
        ('explicit_all_axes', lambda v: v.abs().amax(dim=(0, 1))),
        ('flatten_then_dim0', lambda v: v.abs().flatten().amax(dim=0)),
        ('two_axis_max', lambda v: v.abs().max(dim=1).values.max(dim=0).values),
    ]:
        try:
            value = float(function(x))
            torch.npu.synchronize()
            r['cases'][name] = {'passed': value == 92795.5, 'value': value}
        except Exception as exc:
            r['cases'][name] = {'passed': False, 'error_type': type(exc).__name__, 'message': str(exc)}
        p.write_text(json.dumps(r, indent=2) + '\n')
    r['completed'] = True
    assert any(c['passed'] for c in r['cases'].values()), r
    p.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)


if __name__ == '__main__':
    main()
