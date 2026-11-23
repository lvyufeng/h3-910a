#!/usr/bin/env python3
"""R180: golden reference for the H3RangeStats acceptance test.

Computes the expected verdict for float16 data purely from bit patterns, so the on-device
kernel can be compared against an independent implementation. No float predicates are used:
R165 measured that this device cannot be trusted to report NaN through isnan/isfinite/x!=x/max.

Deliberately dependency-free (struct only), so it runs both on the Ascend host and on the
client, and cannot disagree because of a numpy version.

Run standalone to self-check the adversarial suite:
    python3 reference180.py
"""
from __future__ import annotations
import struct

FP16_MAX_MAG = 0x7BFF      # bit pattern of 65504.0, the largest finite magnitude
MAG_MASK = 0x7FFF
INF_NAN_BASE = 0x7C00      # magnitude equal to this is Inf, greater is NaN


def bits_of(values) -> list[int]:
    """Raw 16-bit patterns of the given values, encoded as float16."""
    return [int.from_bytes(struct.pack('<e', v), 'little') for v in values]


def max_magnitude_bits(values) -> int:
    mags = [b & MAG_MASK for b in bits_of(values)]
    return max(mags) if mags else 0


def verdicts_from_max_mag(max_mag: int) -> dict:
    return {
        'max_magnitude_bits': max_mag,
        'nonfinite_present': bool(max_mag >= INF_NAN_BASE),
        'saturated_present': bool(max_mag == FP16_MAX_MAG),
    }


def reference_verdict(values) -> dict:
    return verdicts_from_max_mag(max_magnitude_bits(values))


NAN = float('nan')
INF = float('inf')

# (name, values, expected nonfinite present, expected saturated present)
CASES = [
    ('all_finite_normal',        [1.0, -2.5, 100.0],            False, False),
    ('single_nan',               [1.0, NAN, 3.0],               True,  False),
    ('single_posinf',            [1.0, INF, 3.0],               True,  False),
    ('single_neginf',            [1.0, -INF, 3.0],              True,  False),
    ('single_pos_saturation',    [1.0, 65504.0],                False, True),
    ('single_neg_saturation',    [1.0, -65504.0],               False, True),
    ('two_pos_saturation',       [65504.0, 65504.0],            False, True),
    ('two_neg_saturation',       [-65504.0, -65504.0],          False, True),
    ('both_signs',               [65504.0, -65504.0],           False, True),
    # the two cases that broke the float-predicate implementation in R164
    ('nan_and_saturation',       [NAN, 65504.0],                True,  False),
    ('inf_and_saturation',       [INF, -65504.0],               True,  False),
    ('many_bad',                 [NAN, INF, -INF, 65504.0, -65504.0, 1.0], True, False),
    # 65504 is the largest finite magnitude and the spacing there is 32, so the next
    # representable value below it is 65472; 65503 rounds UP to 65504 and is therefore itself
    # a saturation case. Both are covered so the boundary cannot be mistaken.
    ('just_below_max',           [65472.0, -65472.0],           False, False),
    ('rounds_up_to_max',         [65503.0, -65503.0],           False, True),
    ('subnormals',               [6.1e-5, -6.1e-5, 1e-7],       False, False),
    ('zeros',                    [0.0, -0.0],                   False, False),
    ('max_finite_many',          [65504.0] * 8,                 False, True),
]


def self_check() -> int:
    failures = []
    for name, values, want_nf, want_sat in CASES:
        got = reference_verdict(values)
        if got['nonfinite_present'] != want_nf or got['saturated_present'] != want_sat:
            failures.append((name, got, (want_nf, want_sat)))
    print(f'cases={len(CASES)} failures={len(failures)}')
    for f in failures:
        print('  MISMATCH', f)
    # a couple of bit-pattern spot checks that pin the constants down
    assert max_magnitude_bits([65504.0]) == 0x7BFF, 'FP16 max magnitude must be 0x7bff'
    assert max_magnitude_bits([float('inf')]) == 0x7C00, 'Inf must be 0x7c00'
    assert max_magnitude_bits([float('nan')]) > 0x7C00, 'NaN magnitude must exceed 0x7c00'
    assert max_magnitude_bits([1.0]) == 0x3C00, '1.0 must be 0x3c00'
    assert max_magnitude_bits([65472.0]) == 0x7BFE, '65504-32 must be 0x7bfe'
    assert max_magnitude_bits([65503.0]) == 0x7BFF, '65503 rounds up to 65504 in float16'
    print('bit-pattern spot checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(self_check())
