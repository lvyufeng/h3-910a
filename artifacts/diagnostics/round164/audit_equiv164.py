#!/usr/bin/env python3
"""R164: adversarial unit test of native vs cheap safety-check implementations.

R163 compared the two implementations on 465 real audited tensors and found identical
counts, but every one of those tensors had zero bad elements, so the comparison only
demonstrated "both report zero". This test drives the edge cases directly:

  NaN, +Inf, -Inf, exactly +65504, exactly -65504, several of each, mixed cases,
  the neighbourhood just below the maximum finite magnitude, and subnormals.

It also pins down one nuance R163 could not show: the native saturation check counts
occurrences, whereas the cheap check tests presence, so with duplicate saturated values
the counts differ while the verdict does not. The audit only ever consumes the verdict
(zero versus non-zero), which is why presence is sufficient, but the distinction is
recorded here rather than glossed over.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def cases(torch, device):
    f16 = torch.float16
    nan = float('nan'); inf = float('inf')
    def t(values):
        # the native helper only accepts NPU tensors, so build every case on the device
        return torch.tensor(values, dtype=f16, device=device).reshape(1, -1)
    return [
        ('all_finite_normal', t([1.0, -2.5, 100.0, -0.5])),
        ('single_nan', t([1.0, nan, 3.0])),
        ('single_posinf', t([1.0, inf, 3.0])),
        ('single_neginf', t([1.0, -inf, 3.0])),
        ('single_pos_saturation', t([1.0, 65504.0, 3.0])),
        ('single_neg_saturation', t([1.0, -65504.0, 3.0])),
        ('two_pos_saturation_multiplicity', t([65504.0, 65504.0, 1.0])),
        ('two_neg_saturation_multiplicity', t([-65504.0, -65504.0, 1.0])),
        ('both_signs_saturated', t([65504.0, -65504.0])),
        ('nan_and_saturation', t([nan, 65504.0])),
        ('inf_and_saturation', t([inf, -65504.0])),
        ('just_below_max_finite', t([65503.0, -65503.0])),
        ('largest_finite_below_65504', t([65504.0 - 32.0])),
        ('subnormals', t([6.1e-5, -6.1e-5, 1e-7])),
        ('zeros', t([0.0, -0.0])),
        ('many_bad', t([nan, inf, -inf, 65504.0, -65504.0, 1.0])),
    ]


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

    def fast_counts(value):
        bad = int(torch.isnan(value).sum(dtype=torch.int32)) + int(torch.isinf(value).sum(dtype=torch.int32))
        top = value.max(); bottom = value.min()
        presence = int(bool(top == 65504)) + int(bool(bottom == -65504))
        return bad, presence

    rows = []
    verdict_mismatches = 0
    count_mismatches = 0
    for name, tensor in cases(torch, 'npu'):
        nb, nboundary = native_counts(tensor)
        fb, fboundary = fast_counts(tensor)
        verdict_same = (nb != 0) == (fb != 0) and (nboundary != 0) == (fboundary != 0)
        counts_same = (nb == fb) and (nboundary == fboundary)
        if not verdict_same:
            verdict_mismatches += 1
        if not counts_same:
            count_mismatches += 1
        rows.append({'case': name, 'native': [nb, nboundary], 'fast': [fb, fboundary],
                     'verdict_same': verdict_same, 'counts_same': counts_same})

    out = {'scope': 'R164 adversarial unit test of the native versus cheap safety checks',
           'verification_only': True, 'model_arithmetic_unchanged': True,
           'no_check_removed': True, 'nothing_adopted': True,
           'device': 'npu:0', 'cases': len(rows),
           'verdict_mismatches': verdict_mismatches,
           'count_mismatches': count_mismatches,
           'verdicts_always_agree': verdict_mismatches == 0,
           'rows': rows,
           'multiplicity_nuance': ('the native saturation check counts occurrences while the cheap check tests presence, so '
             'duplicate saturated values make the counts differ even though the verdict does not. The audit only consumes the '
             'verdict in both modes, so presence is sufficient, but R163s claim of identical counts is only valid where the '
             'counts are zero or one.'),
           'why_the_verdicts_are_equivalent': ('65504 is the largest finite FP16 magnitude, so at least one element with abs 65504 '
             'exists exactly when max(x) is 65504 or min(x) is -65504; non-finite elements are counted directly by isnan and isinf.'),
           'consequence_for_adoption': ('if the cheap check is adopted, its contract should be stated as an identical pass/fail '
             'verdict, not as identical counts, unless the count itself is needed.')}
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('audit_equiv164.json')
    out_path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('cases', 'verdict_mismatches', 'count_mismatches', 'verdicts_always_agree')}, indent=1))
    for row in rows:
        if not row['verdict_same'] or not row['counts_same']:
            print('  diff:', row)


if __name__ == '__main__':
    main()
