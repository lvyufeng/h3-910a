#!/usr/bin/env python3
"""R166: a bit-pattern safety check that is exactly count-equivalent and cheaper.

R165 established that FP16 float predicates are unreliable on this device, which is why
the shipped check inspects bit patterns. That approach is required, but the shipped
implementation widens the bit view to int32 (doubling traffic) and materialises three
full-size masks. This version stays on int16 and derives both counts from the magnitude
field:

    magnitude = bits & 0x7fff
    non-finite count  = (magnitude >= 0x7c00).sum()   # covers NaN and Inf together
    saturation count  = (magnitude == 0x7bff).sum()   # 65504 is 0x7BFF in FP16

Both should reproduce the native counts exactly, including multiplicity, because the two
predicates are the same bit tests the native code performs.

Verification here is adversarial plus randomised: crafted edge cases and fuzz over the
shapes the model actually audits, with bad values injected at random positions.
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

    def native_counts(value):
        nan_mask, posinf, neginf = ieee_nonfinite_masks(value)
        bad = int((nan_mask | posinf | neginf).sum(dtype=torch.int32))
        boundary = int((value.abs() == 65504).sum(dtype=torch.int32))
        return bad, boundary

    def bit_counts(value):
        dense = torch_npu.npu_format_cast(value, 2)
        if not dense.is_contiguous():
            dense = dense.contiguous()
        bits = dense.view(torch.int16)
        magnitude = bits & 0x7fff
        bad = int((magnitude >= 0x7c00).sum(dtype=torch.int32))
        saturated = int((magnitude == 0x7bff).sum(dtype=torch.int32))
        return bad, saturated

    rows, verdict_mismatches, count_mismatches = [], 0, 0
    def check(name, tensor):
        nonlocal verdict_mismatches, count_mismatches
        nb, ns = native_counts(tensor)
        fb, fs = bit_counts(tensor)
        verdict_same = (nb != 0) == (fb != 0) and (ns != 0) == (fs != 0)
        counts_same = (nb == fb) and (ns == fs)
        verdict_mismatches += 0 if verdict_same else 1
        count_mismatches += 0 if counts_same else 1
        rows.append({'case': name, 'native': [nb, ns], 'bit': [fb, fs],
                     'verdict_same': verdict_same, 'counts_same': counts_same})

    # --- crafted edge cases (the R164 suite, which broke the float-predicate version)
    nan = float('nan'); inf = float('inf')
    f16 = torch.float16
    def t(values):
        return torch.tensor(values, dtype=f16, device='npu').reshape(1, -1)
    crafted = [
        ('all_finite_normal', [1.0, -2.5, 100.0]),
        ('single_nan', [1.0, nan, 3.0]),
        ('single_posinf', [1.0, inf, 3.0]),
        ('single_neginf', [1.0, -inf, 3.0]),
        ('single_pos_saturation', [1.0, 65504.0]),
        ('single_neg_saturation', [1.0, -65504.0]),
        ('two_pos_saturation', [65504.0, 65504.0]),
        ('two_neg_saturation', [-65504.0, -65504.0]),
        ('both_signs', [65504.0, -65504.0]),
        ('nan_and_saturation', [nan, 65504.0]),
        ('inf_and_saturation', [inf, -65504.0]),
        ('many_bad', [nan, inf, -inf, 65504.0, -65504.0, 1.0]),
        ('just_below_max', [65503.0, -65503.0]),
        ('subnormals', [6.1e-5, -6.1e-5, 1e-7]),
        ('zeros', [0.0, -0.0]),
        ('max_finite_many', [65504.0] * 8),
    ]
    for name, values in crafted:
        check('crafted:' + name, t(values))

    # --- randomised fuzz over the shapes the model audits
    generator = torch.Generator(device='cpu').manual_seed(20261123)
    fuzz_shapes = [(1, 2), (1, 17), (1, 64), (1, 1024), (55, 5376), (2048, 3584), (10448, 14336)]
    for shape in fuzz_shapes:
        for trial in range(3):
            total = 1
            for dim in shape:
                total *= dim
            values = torch.randn(total, generator=generator, dtype=torch.float32).clamp_(-1000, 1000)
            ncn = total
            for _ in range(min(5, ncn)):
                pos = int(torch.randint(0, ncn, (1,), generator=generator).item())
                kind = int(torch.randint(0, 5, (1,), generator=generator).item())
                values[pos] = [nan, inf, -inf, 65504.0, -65504.0][kind]
            tensor = values.to(f16).reshape(*shape).npu()
            check(f'fuzz:{shape}:trial{trial}', tensor)

    out = {'scope': 'R166 exact-count equivalence of a bit-pattern safety check on int16',
           'verification_only': True, 'model_arithmetic_unchanged': True, 'nothing_adopted': True,
           'device': 'npu:0', 'cases': len(rows),
           'verdict_mismatches': verdict_mismatches, 'count_mismatches': count_mismatches,
           'exactly_count_equivalent': count_mismatches == 0,
           'predicates_used': ['magnitude >= 0x7c00 for NaN and Inf together',
                               'magnitude == 0x7bff for the 65504 saturation bit pattern'],
           'no_float_predicates': True,
           'differences_from_the_shipped_implementation': [
               'stays on the int16 view instead of widening to int32',
               'one comparison covers NaN and Inf instead of three materialised masks',
               'saturation is a bit comparison instead of an abs() plus equality over a new temporary'],
           'rows': rows}
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('bitcheck166.json')
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('cases', 'verdict_mismatches', 'count_mismatches',
                                         'exactly_count_equivalent')}, indent=1))
    for row in rows:
        if not row['counts_same']:
            print('  diff:', row)


if __name__ == '__main__':
    main()
