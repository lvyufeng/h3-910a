#!/usr/bin/env python3
"""R165: how reliably does this device report NaN/Inf under different reduction forms?

R164's cheap check reported zero non-finite values on a two-element tensor that provably
contained NaN, while the native bit-pattern check found it. Before a fail-safe design can
lean on isnan/isinf, this establishes which reduction forms are trustworthy, and whether
the failure depends on shape, on where the NaN sits, or on the reduction itself.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def main():
    import torch, torch_npu
    torch.npu.set_device(0)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from npu_half_nan_compat124 import ieee_nonfinite_masks
    f16 = torch.float16
    nan = float('nan'); inf = float('inf')

    def native(value):
        nan_mask, posinf, neginf = ieee_nonfinite_masks(value)
        return int((nan_mask | posinf | neginf).sum(dtype=torch.int32))

    shapes = [(1, 2), (1, 17), (1, 64), (1, 1024), (2048, 3584)]
    rows = []
    for shape in shapes:
        total = 1
        for dim in shape:
            total *= dim
        for pos in (0, total // 2, total - 1):
            for kind, fill in (('nan', nan), ('posinf', inf), ('neginf', -inf)):
                values = torch.zeros(total, dtype=f16)
                values[pos] = fill
                tensor = values.reshape(*shape).npu()
                record = {'shape': list(shape), 'pos': pos, 'kind': kind, 'native': native(tensor)}
                record['isnan_sum'] = int(torch.isnan(tensor).sum(dtype=torch.int32))
                record['isinf_sum'] = int(torch.isinf(tensor).sum(dtype=torch.int32))
                record['isfinite_all'] = bool(torch.isfinite(tensor).all())
                record['isnan_any'] = bool(torch.isnan(tensor).any())
                record['isinf_any'] = bool(torch.isinf(tensor).any())
                record['max'] = float(tensor.max()); record['min'] = float(tensor.min())
                # a form that cannot be poisoned by NaN: compare against itself
                record['ne_self_count'] = int((tensor != tensor).sum(dtype=torch.int32))
                rows.append(record)

    # also: saturated values, to see whether max/min find them when nothing is non-finite
    sat_rows = []
    for shape in shapes:
        total = 1
        for dim in shape:
            total *= dim
        values = torch.zeros(total, dtype=f16)
        values[total // 2] = 65504.0
        tensor = values.reshape(*shape).npu()
        sat_rows.append({'shape': list(shape), 'max': float(tensor.max()), 'min': float(tensor.min()),
                         'max_is_65504': bool(tensor.max() == 65504),
                         'abs_eq_count': int((tensor.abs() == 65504).sum(dtype=torch.int32))})

    failures = [r for r in rows if r['isnan_sum'] + r['isinf_sum'] != r['native']]
    ne_failures = [r for r in rows if r['ne_self_count'] != r['native']]
    out = {'scope': 'R165 reliability of NaN/Inf reporting on this device',
           'verification_only': True, 'model_arithmetic_unchanged': True, 'nothing_adopted': True,
           'cases': len(rows),
           'isnan_plus_isinf_disagrees_with_native': len(failures),
           'self_inequality_disagrees_with_native': len(ne_failures),
           'failure_examples': failures[:6],
           'self_inequality_examples': ne_failures[:4],
           'saturation_rows': sat_rows,
           'verdict': ('isnan().sum() + isinf().sum() reproduces the native non-finite count in every case'
                       if not failures else
                       'isnan/isinf disagree with the native bit-pattern count, so a fail-safe design cannot rely on them'),
           'self_inequality_note': ('x != x is true exactly for NaN and needs no NaN-aware reduction, so if the plain reductions '
             'are unreliable this is a candidate replacement')}
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('nan_behaviour165.json')
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('cases', 'isnan_plus_isinf_disagrees_with_native',
                                         'self_inequality_disagrees_with_native', 'verdict')}, indent=1))
    for row in failures[:6]:
        print('  fail:', row)


if __name__ == '__main__':
    main()
