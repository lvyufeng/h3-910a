# Round67 — native-statistics-preserving BF16 Norm-tail fusion

## Measured result and acceptance boundary

The complete resident experimental50-layer text forward improves from
**3.61755s to2.63961s (1.370×)** with the prescribed native statistics unchanged.
Three candidate per-call rank maxima are2.63606/2.63961/2.64224s. Original Norm
restoration returns3.61775s (+0.00562% median drift). This removes~0.978s but
**does not reach2s**, and is not the production complete text-encode stage.

All **96 complete text outputs** (three modes×four forwards×eight ranks), each
10,460,160 values, match raw bytes. The first candidate also passes **1,600 actual
Norm output checks plus400 separate native residual checks**, covering every
Norm in every layer/rank. All actual input and parameter bytes remain unchanged.
The baseline continues to match the earlier stable Round60 complete text tensor,
not an invented or silently changed reference.

[Verified summary](<summary.json>) and [rank0 raw proof](<ranks/text_norm_rank0.json>)
retain the full mode/rank/repeat matrix. The whole run keeps the exact Round65
shadow/shared-activation stack fixed; it does not claim an additive cross-run
speedup relative to the original BF16/HF or production encoder.

Allocated HBM snapshots remain20,208,014,848 bytes/rank in all modes. Reserved
snapshots rise only2,097,152 bytes/rank with kernel first use; these are not peak
HBM measurements. The existing **11.1084GiB/rank FP32 shadow overhead** is still
present and must not disappear from the complete-service memory budget.

Exceptional-input equivalence is **not** accepted: the prototype's252 differing
NaN-payload words remain visible below. The accepted result is exact output
preservation on the measured finite actual workload, not a blanket claim that
all BF16/FP32 exceptional arithmetic is interchangeable.

## Scope and candidate

This experiment changes **only the pointwise tail** of the current stable
experimental resident text-only TP8 stack. It is not production encoder serving,
originalBF16/HF-oracle equivalence or an official49-step generation result.
FP32 shadow weights/shared activation, padded K/V geometry, FP32 causal math,
SwiGLU/RoPE, communication order and checkpoint tensors stay fixed.

The [guarded TBE kernel](<norm_tail_guard67.py>) implements:

```text
BF16 x → exact FP32 decode → multiply by native FP32 inverse-RMS
→ normalized BF16 RNE boundary → exact FP32 decode
→ multiply by decoded BF16 gamma → final BF16 RNE boundary
```

The low16 bits are explicitly cleared by each RNE step, so the middle BF16
round-trip is represented exactly in FP32 without an extra GM write/read.
There is no reduction, rsqrt, Newton iteration, overflow-marker branch or
signed-zero correction inside this kernel. The native residual BF16 update,
`xf.square().mean(...)`, epsilon add and `torch.rsqrt` remain the original
operators, recalculated **on every complete candidate forward**.

The [integration harness](<text_norm_tp8.py>) uses native Norm → tail fusion →
native Norm restored on the **same already-cached experimental stack**. Only
the candidate's first complete forward does same-input native-reference double
computations and fixture captures; first calls, compilation and all CPU proof
work are excluded from the three clean warm calls per mode.

## Standalone numerical and memory-safety gates

[Original prototype validation](<validation01.json>) checks native-statistics
fixtures17×128,9×5120,2043×128 and2043×5120, including both zero signs and normal/
subnormal boundary values. All outputs match native raw BF16 words, including
three repeated calls each. Those reported sub-millisecond/millisecond numbers
are **precomputed-inverse tail microtimes**, not complete Norm or text latency.

Its all65536 BF16-word, unit-inverse/unit-gamma probe matches all **65280 finite
input words**, but **252 NaN payload words differ** from the native canonicalized
result. This failed exceptional-input identity is deliberately retained. Neither
the codec proof nor this finite-workload experiment establishes arbitrary NaN/
Inf/overflow equivalence. No `nan_to_num`, clamp or hidden exception waiver is
introduced. These modules remain diagnostic-only and are not installed as a
production default.

[Boundary validation source](<validate_boundaries67.py>) and its
[successful raw receipt](<validation02.json>) test the guarded version:

- Native FP32 inverse just below, exactly at and just above the normalized
  BF16 midpoint, with both signs.
- All128×128 positive BF16 mantissa-product pairs and their negative counterparts
  for the final gamma multiply:281 exact midpoints,8008 below-midpoint nonexact
  products and7639 above-midpoint products in the positive fixture.
- Contiguous-but-unaligned BF16 x/gamma views are explicitly rejected; contiguous
  does **not** imply32-byte alignment.
- Wrong inverse broadcast shape is rejected even when its element count matches.

The [original kernel prototype](<norm_tail67.py>) and guarded version share the
same build/arithmetic AST. The guarded version adds exact per-row inverse shape
and32-byte final GM-address checks under a **new source fingerprint**, leaving
the original executed source/receipt intact.

For safe GM reads, inverse storage is padded to an8-FP32/32-byte boundary;
row2042 reads block2040…2047 from an allocated2048-value buffer. Q-head rows16344
are already divisible by8. Width128 uses256-element codec UB buffers with zeroed
tails, but reads/writes only the real128 BF16 words. Width5120 has no codec tail.
All launches use the current device/stream and fresh outputs, never overwrite
x, gamma, residual or native inverse. Unsupported alignment/layout is rejected.

## Full-stack gate

The first candidate forward compares **all200 actual pre-Norm calls per rank**:
hidden input, residual-add hidden, actual Q-head and actual K-head inputs in each
of50 layers. Residual outputs are checked separately in all50 residual Norms.
This does not substitute a post-RoPE tensor of the same shape for an actual
pre-QNorm activation.

Selected rank0 layers0,24,49 save all four actual Norm-family x/gamma/native
inverse/native output/candidate output fixtures remotely. For hidden-residual,
this saved `x` is **after** native residual addition and BF16 rounding; it can
replay the tail but does not contain the two pre-add inputs needed to replay a
complete AddRMSNorm. The separate400 actual residual checks above still compare
the original native add/update. Each file is exclusive
and cannot be silently replaced. Captures and same-input comparisons occur only
in the excluded first call. Every first and warm complete output must compare
all10,460,160 values raw-bit-identically to the same-run baseline, which itself
must reproduce the earlier stable Round60 complete tensor. All actual parameter,
embedding/positions/mask and DeepStack bytes are hashed before and after.

Clean timing preserves barrier+start drain outside, model plus ending device
drain inside, and CPU download/hash outside the interval. Report median of
per-call **rank maxima**, never maximum of rank medians. The added Norm tail
still includes native stats/residual work and its inverse padding/allocation;
no precomputed-statistics shortcut is used in complete text timing.

[Aggregator](<summarize.py>) rejects failed/incomplete/mode-mismatched jobs,
missing Norm layers/ranks, changed source fingerprints, wrong shape/alignment,
changed residual/input/parameter/final-output bytes, altered fixed shadow/KV
cache footprints or missing standalone native boundary proofs. Pending numbers
are not promoted to an achieved target.

[Final verification](<final_verification.json>) matches23 unchanged production/
model/dedicated-module sources and14 executed diagnostic/dependency sources. It
also fingerprints and rechecks all12 selected remote tail fixtures: all tensors
finite, inverse values positive, native/candidate outputs raw-byte equal. The
initial CPU-only verification attempt stopped during Torch's automatic NPU
backend import (`libhccl.so` absent without the CANN environment), **before any
evidence claim/write**; rerunning unchanged verification code with
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` succeeds without using NPU resources. This was
a verification-environment error, not a model-output failure.

The [live-probed cleanup receipt](<cleanup_receipt.json>) records all eight NPUs
at0% utilization, no accelerator processes and the owned lease released. No
production rollback is needed because no production file was changed. The
[regression receipt](<test_receipt.json>) records13 new Norm-tail/access/native-AST/
actual-proof tests and74 preceding profiling/cache/service/kernel/block tests,
all87 passing. Those portable checks validate contracts and the retained NPU
proofs; they do not simulate NPU arithmetic or replace the measured A/B.

## Reproduction

```bash
python3 -m unittest discover -s artifacts/diagnostics/round67 -p 'test_*.py' -v
python3 artifacts/diagnostics/round67/summarize.py
```

Real NPU replay requires a fresh owned lease and fresh evidence claim/output IDs.
No source or checkpoint in the serving stack is deployed or changed. Full text
stage≤2s, original native quality, audio repeatability, strict DiT<15s and the
matched official comparison remain separate open acceptance gates.

The next measured gate should profile the **new2.64s stack**, rather than subtract
this run's gain from a previous run's Norm category and call the remainder an
exact attribution. A bounded follow-up can test BF16 decode→FP32 square fusion
while retaining the same native mean/reduction and rsqrt, initially on the saved
actual tail inputs and then on all actual Norms. Residual pre-add fusion requires
new original pre-add/residual fixtures; the saved post-add hidden-residual `x`
alone cannot prove that transformation. None of those follow-ups is an achieved
speedup, a license to change reduction order, or a2s promise.
