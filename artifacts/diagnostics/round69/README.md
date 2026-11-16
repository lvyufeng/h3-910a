# Round69 — native-layout exact BF16 activation expansion

## Result: resident text-only2s gate reached, not the complete-service goal

This same-run exact experimental resident50-layer text-only TP8 stack improves
from **2.10667s to1.82918s (1.152×)** by replacing BF16 activation casts with an
integer-codec TBE expansion. Combined mode's per-call rank maxima are
**1.82918/1.82935/1.82592s**, all below2s. The full-service text encode goal is
**not marked complete**: processor, vision, loading/offload and pipeline residency
are excluded, and the stack still uses11.1084GiB/rank FP32 shadow weights.

Five-mode single-factor A/B medians (three per-call rank maxima each):

- Native casts:2.10667s.
- Shared Q/K/V scope cast only:2.00212s; not all calls below2s.
- Other linear activation casts only:1.92095s; all warm calls below2s.
- Both:1.82918s; all warm calls below2s.
- Native restored:2.09923s (-0.353% median drift).

All **160 complete text outputs** (five modes×four forwards×eight ranks), each
10,460,160 values, match raw bytes. The three candidate first forwards also pass
**4,800 actual cast checks** (full bits, native shape/stride/dtype/physical format,
input readonly) and **7,200 same-input projection checks**, including100 FP32
row partials/rank/candidate before all-reduce. All actual inputs and parameter
bytes remain unchanged. This is relative to the unchanged stable experimental
Round60 full tensor, **not originalBF16/HF-oracle or official generation quality**.

[Verified summary](<summary.json>) and [rank0 raw proof](<ranks/cast_rank0.json>)
retain the full mode/rank/repeat matrix. Allocated/reserved HBM snapshots remain
20,208,014,848 /21,105,737,728 bytes/rank across all modes. These are not peak HBM
or proof that the full H3 pipeline can co-reside with these caches. No serving
source or checkpoint is modified or deployed.

## What changed—and what did not

[Kernel](<activation_cast69.py>) uses the previously tested integer BF16 codec to
insert zero low halfwords and the original BF16 high word into each FP32 value.
No floating-point multiply, rounding, NaN canonicalization, clamp, signed-zero
fix, weight packing or GEMM exists in the kernel.4096-element tiles and128-aligned
remaining tails are padded to256 elements only in UB; GM reads/stores use real
counts, including the3200-width activation's last tile.

The API accepts only current-device32-byte-aligned contiguous dense ND/NCHW
BF16 activations. It does not silently make noncontiguous RoPE/causal tensors
contiguous or reinterpret their physical order. Output logical shape/stride is
preserved; each actual native cast's exact physical format is checked during
same-input acceptance. Noncontiguous, unaligned and unsupported inputs reject.

[Harness](<text_cast_tp8.py>) changes only two conversion points in a process-local
SharedActivation subclass:

1. `scope(value)` converts the identical Q/K/V input once before its three GEMMs.
2. `get(value)` fallback converts gate/up, O and down activations; the same-object
   shared Q/K/V hits are **not** converted again.

Signature/lifetime checks and `finally` release remain intact. Each complete
forward has50 scope casts,150 fallback casts and150 shared hits, so combined mode
performs200 actual conversions, not350. No cross-forward activation cache, global
`Tensor.float` monkeypatch or weight-cast change is used. FP32 shadow weights,
three independent Q/K/V GEMM shapes,1024-output padded K/V, FP32 all-reduce→BF16
output, merged gate/up GEMM, native causal attention, exact-prefix native mean/
rsqrt and exact-tail Norm, native-round TBE RoPE/SwiGLU all remain fixed.

Causal q/k/v conversions are **not changed**. They consume transposed views and
need a separate layout-aware proof before they can use a contiguous-only codec.
Changing them here would turn a pure linear-operand cast experiment into another
attention-layout/dispatch experiment.

## Native conversion and same-input gates

[Standalone validation](<validation01.json>) compares all **65,536 BF16 input
words** to both exact integer FP32 bit expansion and the actual native NPU cast,
including all sign/subnormal/Inf/NaN payload patterns; both full-bit comparisons
and native/candidate layout match. This validates the cast itself, **not** generic
subsequent exceptional Norm/add/GEMM behavior. Previous NaN-add and Norm-tail
payload counterexamples remain unwaived.

Three actual linear operand shapes—`[1,2043,5120]`, `[1,2043,1024]`,
`[1,2043,3200]`—also pass full-bit/read-only/native-layout checks and four repeated
calls. Their synchronized standalone microtimes (e.g.~2.0ms native vs~0.53ms
candidate for10.46M elements) motivate but do **not replace** the full text A/B.
An offset continuous view is actually constructed and rejected by the alignment
guard; `.contiguous()` is not assumed to repair a misaligned already-contiguous
view.

Candidate first forwards independently convert and compare every actual cast;
compare full projection outputs and FP32 row partials; then compare the complete
50-layer output. Reference double-compute, CPU copy/hash, kernel first-use/cache
construction are all in the **excluded first call**, never clean warm timing.

Clean timing retains start barrier+device drain outside, complete model plus
ending device drain inside, stop before CPU copy/hash. Report median of same-call
rank maxima, not maximum of rank medians. Current kernel stream/device and GM
alignment are explicitly checked. The [aggregator](<summarize.py>) rejects failed/
incomplete ranks, changed source/input/parameter bytes, missing cast/projection
layer/shape/element coverage, wrong mode or physical layout, changed final raw
bytes, altered shadow/KV footprint, unsafe input acceptance or scope leakage.

## Verification and next acceptance gate

```bash
python3 -m unittest discover -s artifacts/diagnostics/round69 -p 'test_*.py' -v
python3 artifacts/diagnostics/round69/summarize.py
```

Real NPU replay needs a fresh owned lease and fresh evidence/claim ID; existing
raw code and proof files are immutable. [Final source verification](<final_sources.json>)
matches23 unchanged production/model/dedicated-module fingerprints and14 exact
current/prior diagnostic source fingerprints. [Live-probed cleanup](<cleanup_receipt.json>)
records eight idle NPUs, zero accelerator processes and the owned lease released.
[Regression receipt](<test_receipt.json>) retains9 new cast/access/layout/layer-
coverage tests and99 preceding prefix/Norm/profile/cache/service/kernel/block
tests, all108 passing; these portable checks validate contracts and retained
actual-NPU proof, not simulated TBE arithmetic or complete-service timing.
The resident-under2s observation completes only this controlled microstack
threshold. Before production adoption, integrate
reference-compatible lifecycle/load/offload guards, quantify processor/vision/
residency time, preserve the complete-service memory budget and measure actual
HTTP encoder A/B. Whole-A/V identity, strictly sub15s complete DiT steps,49-step
quality and official task/precision/warm-protocol matched comparison remain open.
The overall objective stays active.
