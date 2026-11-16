# Round68 — native-reduction-preserving Norm prefix fusion and new-stack profile

## Measured result

On the same real-input cached resident50-layer text-only TP8 stack, with Round67
Norm tail fixed:

- Native decode/square/residual prefix: **2.61719s**.
- Self-developed BF16-decode→FP32-square: **2.41063s**,1.086×.
- Also fuse the prescribed BF16 residual update before square: **2.08740s**,1.254×.
- Restore native prefix with the same exact tail/caches: **2.60554s**.

These are medians of three per-call rank maxima. The combined prefix candidate
is2.08118/2.08891/2.08740s before profiling and2.08410s median after profiling.
It **does not meet2s**, still depends on the existing11.1084GiB/rank FP32 weight
shadows, and is **not the complete production encode stage**. Processor, vision,
loading/offload and all other H3 pipeline components are excluded.

All **272 complete outputs** (34 complete forwards×8 ranks), each10,460,160
values, match the same-run and earlier stable Round60 text tensor raw bits.
Both candidates pass all200 actual Norms/rank: **3,200 prefix checks**, each
covering the complete FP32 square, native mean variance, native inverse-RMS and
BF16 Norm output. **800 residual updates** are additionally checked. Candidate
square shape, stride, dtype and NPU physical format match the native reduction
input before calling the unchanged mean. Actual input/parameter bytes stay
unchanged across the entire run.

[Verified summary](<summary.json>), [rank0 raw proof](<ranks/prefix_rank0.json>) and
[compact same-rank profile](<breakdown.json>) retain timings and scope. This is
exact preservation of the measured **finite experimental workload**, not original
BF16/HF/official49-step quality or arbitrary exceptional arithmetic equivalence.

## Kernel and preserved boundaries

[Prefix kernel](<norm_prefix68.py>) implements two modes:

1. BF16 x integer-decode to FP32 → square in FP32 → store **all32 square bits**.
2. BF16 residual and x decode to FP32 → residual+x → explicit BF16 RNE boundary
   → store updated BF16 → square the rounded update in FP32.

The candidate does not calculate reduction, mean, epsilon or rsqrt, and never
rounds square to BF16. Native `squares.mean(-1, keepdim=True)`, epsilon add and
`torch.rsqrt` remain the original operators. The already-verified Round67
pointwise tail then consumes native inverse-RMS and preserves both BF16 output
round boundaries. No Newton step, signed-zero fix or overflow-marker shortcut is
introduced. All non-Norm mathematics and collective order stay fixed.

GM reads/writes use real widths128/5120; the128 codec UB buffers pad to256 with
zeroed tails but do not write beyond real-width rows. FP32 square stores use
`width//8`32-byte blocks; BF16 updated stores use`width//16`. Inputs are validated
as current-device,32-byte-aligned dense ND/NCHW storage. No-residual mode never
writes its original x/updated alias; independent read-only checks below certify
that rather than assuming equality between two aliases proves no mutation.

## Native bounded and exceptional probes

[Initial native validation](<validation01.json>) passes square and residual-update
fixtures17×128,9×5120,2043×128,2043×5120, plus repeats. Both candidates produce
native-identical raw values and logical shape/stride in these bounded fixtures.

The square-only sweep covers **all65536 BF16 input words** and matches the actual
native NPU FP32-square output raw bytes. Its `different_values=254` comes from
numeric `NaN != NaN`, **not254 raw-bit differences**. The explicitly bounded finite
subset `abs(x)≤2**60` has47,874 words and finite, raw-identical square outputs.

The retained fields named `native_overflow_values` and `candidate_overflow_values`
count `isinf(output)`, both2 in this probe; here those are the input±Inf cases,
**not a count of finite overflow events**. Matching native NPU square, including
its behavior for huge finite inputs, is not proof of CPU/IEEE overflow equivalence.
No finite-overflow quality promise is inferred from those field names.

[Independent readonly/extreme control](<validation02.json>) snapshots x/residual
before launch, checks both stay unchanged, and checks actual output alignment
and ND/NCHW formats. Finite bounded pairs, both zero signs, cancellation,
subnormal boundaries and maximal finite additions are retained. Extreme
NaN+finite residual pairs expose a real **NaN-payload difference**: the TBE add
preserves payload where native canonicalizes. Their updated and square raw
identity flags are false and remain unaccepted. No payload correction or
`nan_to_num` hides that counterexample, and no generic exceptional-add equivalence
is claimed. This diagnostic candidate is not a production default.

## Whole-stack and new-path timing gates

[Harness](<text_prefix_tp8.py>) compares native-prefix, square-only, add-square and
native-restored with the same Round67 tail and exact FP32 shadow/shared-activation
stack. Each of the first three modes has one excluded first call,3 clean-before,
3 profiled and3 clean-after forwards. Restored has first+3 clean calls. Candidate
first calls double-compute native reference square/variance/inverse/output/residual
and capture proof inputs; clean/profile calls do not double-compute. Every candidate
calls the original native mean/rsqrt anew, not precomputed statistics.

Only first candidate square-mode rank0 residual Norms at layers0,24,49 capture
the original pre-add x and residual, native BF16 update, variance and inverse.
Those now **are original two-input AddRMSNorm fixtures**, unlike the Round67
post-add tail-only fixture. They remain in the remote diagnostic run root with
exclusive filenames and verified content hashes.

Event recording reuses the byte-pinned Round66 preallocated recorder with no
per-component synchronize. Top intervals include whole Norm, local projections,
row calls, causal attention, RoPE, SwiGLU, shared activation entry and DeepStack.
Nested F.linear/all-reduce intervals are **never added again**. Wrapped host-call
wall includes the event record overhead and overlaps event intervals; it is not
pure CPU time. HCCL may use internal streams, so communication event brackets
measure current-stream-visible submission/dependencies, not pure network time.

For the combined prefix, the median rank-max profile call is2.08981s(call2,rank1):
Norm top intervals total **0.10209s**, local projections0.46621s, row-inclusive
intervals0.67556s, causal0.35671s and shared activation entry0.08401s. The top
sum is1.75987s; **0.32994s stays unassigned**. Nested collective intervals0.26281s
are already in row time. Those are same-call/same-rank intervals, not per-category
independent maxima or pure/removable kernel budgets. Baseline new-tail Norm totals
0.61842s in its representative profile; the revised path shifts the next useful
focus from Norm to projection/conversion/collective-visible overhead.

Profile wall median exceeds clean-before by0.318% native,0.131% square and0.116%
add-square. Clean-after drift is-0.582%/+0.0357%/-0.158%, respectively. The signed
wall-minus-top difference is not assigned to CPU/network to force closure.

## Reproduction and remaining gates

```bash
python3 -m unittest discover -s artifacts/diagnostics/round68 -p 'test_*.py' -v
python3 artifacts/diagnostics/round68/summarize.py
```

Real execution requires a fresh owned lease and fresh proof IDs; existing claims
and raw sources must not be overwritten. No production source or checkpoint is
modified or installed. [Final verification](<final_verification.json>) matches23
production/model/dedicated-module fingerprints and15 exact prior/current
experimental source fingerprints; it also verifies all three original pre-add
fixture sets are finite with positive inverse-RMS values. The
[live-probed cleanup](<cleanup_receipt.json>) records eight idle NPUs, zero
accelerator processes and the owned lease released. The
[regression receipt](<test_receipt.json>) records12 new prefix/native-reduction/
actual-proof tests and87 prior Norm/profile/cache/service/kernel/block tests,
all99 passing. These portable tests check contracts and retained real-NPU proof;
they do not emulate TBE arithmetic or replace actual8-card measurement.

Full text service≤2s, native-reference quality beyond this fixed experimental
stack, full A/V acceptance, strict complete DiT<15s and the matched official
A2/A3 comparison remain open.

Next experiment should keep the newly small Norm path fixed and target the
measured projection/cast/row interval, while retaining GEMM shapes and collective
order. Avoid chasing another Norm reduction approximation for the remaining
~0.09s margin: there is little observed Norm interval left, and the service
encode time will include more than this resident model. Byte-preserving TBE
BF16→FP32 activation conversion (or layout+conversion into the exact same linear
operand format) can be a controlled next candidate; any change in native cast
handling of signed/subnormal/NaN words or GEMM dispatch needs new same-input
projection/full-tensor checks, not an assumed speedup or2s guarantee.
