# R215 — tile probes preserved, not adopted

This directory captures the tile experiment that preceded R216/R217. It is **not** a whole-step optimization result or a paired numerical equivalence proof.

| Actual compiled tile, from run logging | Wrapper warm median (ms) | Qualification |
|---|---:|---|
| 128/304 baseline | ≈154.53 | Later fixed-input R216/R217 reproduce this scale |
| 64/768 | 137.89 | Unseeded original synthetic probe; no paired exact-output comparison |
| 64/912 (requested Bc cap1024) | 137.75 | Unseeded original synthetic probe; no paired exact-output comparison |

[64/768 raw receipt](br64_bc768_raw215.json) and [64/912 raw receipt](br64_bc912_raw215.json) retain original fields. Their reported `br=64,bc=64`, `tiles` and `causal_TFLOP` fields are **wrong** because the earlier probe calculated tiles from input41792 rather than compiled43776, and inferred a causal mode that the actual wrapper does not request. Do not use these fields for FLOPs, MFU or occupancy. R217 records actual instances and proves dense attention. The timing alone suggests ≈1.12×, not 1.5×/2×, and is not adopted.

Failures from the prior execution transcript:

- 64/1152: L0C request294,912 B exceeds262,144 B.
- 32/1824: V L1 request466,944 B exceeds440,320 B.
- 32/1216: next-K L1 request311,296 B exceeds245,760 B.
- Wrong `/opt/conda/envs/py310/bin/python` path and an uncommented R215 label caused early launch/syntax failures; fixed before the timing above.

Source snapshots keep pinned SHA values; no model weights or activations were downloaded. The [current probe](sweep215.py) fixes seed20261124, raw-output SHA and padded geometry calculation. It has **not** made the old receipts seeded or numerically verified after the fact. [R217 measured follow-up](../round217/RESULTS.md) is the current local experiment outcome; GitHub remains the live status source.
