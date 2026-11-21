#!/usr/bin/env python3
"""Check exact range-selector boundaries; scalar arithmetic only, no model inference."""
import bisect
import json
import math
import random
import struct
from pathlib import Path


def float32(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bits32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def main():
    thresholds = [bits32(32752.0*(2.0**i)) for i in range(114)]
    words = {0, 1, 0x007fffff, 0x00800000, 0x7f7fffff}
    for b in thresholds:
        words.update((b-1, b, b+1))
    rng = random.Random(126)
    words.update(rng.randrange(0x7f800000) for _ in range(200000))
    failures = []
    for bits in sorted(words):
        peak = float32(bits)
        reference = max(0, math.ceil(math.log2(peak/32752.0))) if peak>32752.0 else 0
        selected = bisect.bisect_left(thresholds, bits)
        if reference != selected:
            failures.append({'bits':bits,'peak':peak,'reference':reference,'selected':selected})
    result={'scope':__doc__,'completed':True,'positive_finite_words_checked':len(words),
            'all_threshold_below_equal_above_cases_included':True,
            'failures':failures,'verification_passed':not failures,
            'NPU_execution_tested':False,'CPU_model_inference':False}
    print(json.dumps(result,indent=2))
    destination=Path(__file__).with_name('range_scalar_math126.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if not failures else 1)

if __name__=='__main__':main()
