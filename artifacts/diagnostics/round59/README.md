# Round59 — remove narrow-projection repeat variation; verify lossless SwiGLU integration

## Established result and remaining objective

The previous repeated-forward variation is reproduced and localized **before
communication**, to the128-channel K/V **FP32 projection path** in the isolated
experimental text stack. Actual inputs and all3200 rank/parameter tensors are
bit-identical before/after the audit. Padding just those measured projections
to1024 outputs then cropping makes the complete50-layer TP8 experiment
**repeatable on all8 ranks** in the tested first/second/third forwards.

That supplies a stable fusion A/B: self-developed native BF16 TBE SwiGLU leaves
rank0's **complete final text tensor raw-bit identical** to the padded eager
reference, and warmed text-only rank-max changes from about5.00s to4.17s. Norm+
SwiGLU repeats in2.724s but differs **6.318% RMS** from that stable reference;
therefore it is **not adopted as lossless**. CPU FP32-linear HF-oracle differences
still remain, no49-step video was run, and complete encoder2s/decoder20s and the
official task-matched comparison are not achieved. **Production is unchanged.**

[Compact raw-derived summary](<summary.json>) records these scopes; do not treat
padding/repeatability alone as a generation-quality or BF16-HF implementation pass.

## Immutable-state repeat audit

[Audit harness](<repeat_audit_tp8.py>) uses the same actual round54 processor/text/
visual/DeepStack captures, BF16 checkpoint storage/I-O, original50-layer loaders,
actual Omni TP8 initialization, **uninstalled experimental FP32 linears/causal
attention**, HF-math option and native apply-RoPE. No vector fusion is enabled.
The unused vocabulary is seeded for loader completeness then removed.

All8 ranks complete8 forwards:3 clean,3 hook-observed and2 with fresh input
allocations. Full input/position/mask/3DeepStack bytes are hashed before/after
**every** forward. Every400 parameter tensors per rank is fully hashed before
and after the sequence, not merely sampled or checked by a version counter.
All3200 parameter hashes and every input hash remain unchanged. Original BF16
resident parameter storage is5.677GiB per rank, not a full pipeline memory budget.

[Rank0 audit](<repeat_audit/repeat_audit_rank0.json>) and
[rank7 audit](<repeat_audit/repeat_audit_rank7.json>) retain hashes/context/results;
all8 complete receipts are versioned. Final hashes agree **across ranks at each
forward**, but differ between forwards. Clean0/1/2 rank-max times20.545/4.980/4.990s,
with next outputs6.650/11.537% RMS different from the first. Synchronizing hooks
and fresh allocations do not cure it; their outputs differ6.980–22.910% vs the
first clean call. This is **not just a one-time cold/warm result transition**.

Actual contexts still show autocast off, default deviceCPU, matmul HF32false.
Hooks always returnNone and do not replace output; numerical comparison is full
rank0, four sampled rows on other ranks, while **full tensor hashes cover all
elements on every rank**. Timing is diagnostic because CPU copies/hashes sync.
Selected tensor evidence is capped below512MiB per file and kept local/remote.

## Earliest divergence is in local K/V, not the final layer or a first HCCL sum

All observed repeated and fresh-input forwards first differ at **layer0**.
Rank0 layer0 output RMS differences vs the hook baseline are0.0230–0.0259%, then
amplify downstream. Before that:

- Initial Norm output, Q projection, Q norm and cos/sin tables are byte-identical.
- K/V local projection outputs change on the **identical Norm input**.
- The K norm, attention, post Norm and MLP then carry that difference forward.
- In hook1 K/V output RMS differences are0.000464/0.000747%, despite the same
  input hash; the first layer output already changes.

The K/V projection calls happen **before their first attention row all-reduce**.
No collectives or attention are present in the direct projection replay below.
It is therefore incorrect to blame the observed earliest variation on HCCL,
input mutation, dropout, a broken final block or the new SwiGLU kernel. This
narrows a specific experimental path, not all FP32 math or the unchanged
production BF16 encoder.

## Direct same-input projection replays

[Projection reproducer](<projection_repeat.py>) reconstructs actual rank0 layer0
Norm output from captured input/gamma and asserts its SHA equals the full audit.
It then holds that input and original rank-local Q/K/V weights constant. CPU
FP32 F.linear is the numerical oracle;20 warmed repeats are hashed **before and
after BF16 output rounding**. Input/weights are unchanged, no attention/HCCL/fusion.

[Raw projection receipt](<projection_repeat.json>):

| Measured case | Q FP32/BF16 unique outputs | K unique outputs | V unique outputs |
|---|---:|---:|---:|
|original per-call FP32 casts|1/1|20/20|20/20|
|resident contiguous FP32 operands|1/1|20/20|20/20|
|explicit `torch.mm`|1/1|20/20|20/20|
|input rows2043→2048|1/1|20/20|20/20|
|`deterministic_algorithms(True)`|1/1|20/20|20/20|
|output channels→1024 + crop|1/1|**1/1**|**1/1**|

Padding K/V outputs only to256/512 is insufficient: FP32 results still vary and
BF16 results have8–13 distinct hashes. Merely increasing residency, using MM,
row alignment or the high-level deterministic flag is **not a verified fix**.
The flag is restored in the isolated probe; no global production settings change.

For K the unpadded resident last/first FP32 maxdifference is5.96e-8; V2.98e-8.
These tiny FP32 low bits cross BF16 rounding boundaries. The padded valid outputs
remain within~8.6e-7/7.6e-7 relative RMS of CPU FP32. **Padding changes accumulation
order**, so this is not a claim of bitwise equality to every random unpadded
result or unchanged BF16 HF CPU linears.

The causal mechanism inside CANN (for example an atomic/split-K schedule) is **not
proven by a kernel trace**; the measured narrow-dispatch variation and effective
dimension intervention are what is established. No physical910A throughput ceiling
or A2 reclassification is inferred.

## Default-off measured-width candidate

[Padding helper](<padded_fp32_projection59.py>) accepts only the measured BF16
1×2043×5120 input/128×5120 weight inference shape on NPU. It pads FP32 weight rows
to1024 with zeros, preserves all valid channels, crops/casts back to BF16, and
caches by original storage/shape/stride/version/device. Holding the original view
prevents data pointer reuse. Training, unsupported sizes and other precision/
quantized paths are not silently changed.

[Candidate source](<candidate_padded_text_encoder.py>) adds only a default-off
`VLLM_OMNI_910A_TEXT_PAD_KV` branch to the experimental local FP32-linear helper.
Parameter constructors/loaders, shared communication, native production norm and
all unrelated executable AST are unchanged. The100 padded K/V matrices consume
**2097152000 additional FP32 bytes (~1.953GiB) per rank**. This must not be confused
with testing co-residency alongside the full DiT/VAE model; no pipeline memory/
offload optimization is claimed. The candidates/helpers remain in diagnostic paths.

## Stable complete-text fusion A/B

[Full-stack harness](<text_padded_vectors_tp8.py>) uses original50-layer loaders and
same visual captures. FP32 linears/causal and HF math stay fixed; the measured
padding and optional native vector kernels are the changed paths.200 norm modules
are wrapped only inside this isolated model. Each case has first/second/third
complete text forwards; first SwiGLU-enabled forwards include50 independent
same-input eager comparisons and CPU copies, which are disabled for warm timing.

[Rank0 full receipt](<padded_vectors/text_padded_vectors_rank0.json>) and
[rank7 full receipt](<padded_vectors/text_padded_vectors_rank7.json>) retain all
local checks and comparisons; all8 ranks complete.

| Experimental resident text case | Second rank-max | Third rank-max | Repeats exact on all8? | Rank0 final vs stable padded eager |
|---|---:|---:|---|---:|
|unpadded baseline|5.025s|5.006s|**no**|unstable reference|
|padded eager|5.007s|4.999s|**yes**|reference|
|padded + self-developed SwiGLU|4.223s|**4.167s**|**yes**|**raw-bit exact**|
|padded + native Norm/SwiGLU|2.725s|**2.724s**|**yes**|**6.318% RMS**|

SwiGLU integration now has a **stable same-run full-output identity check**,
not merely isolated scalar agreement. All800 layer/rank comparisons across the
SwiGLU-enabled cases remain raw-bit exact and finite (**5.23008billion scalar
outputs**). Full-output comparison to the stable reference is explicitly recorded
on rank0; every rank's own repeats are checked over the complete final tensor.

Norm fusion has a stable but nonzero difference; prior tiny same-input Norm
operator errors do **not** establish whole-model losslessness. Do not adopt it
as a lossless stack just because it is faster. Final CPU FP32-linear HF-oracle
RMS is7.220% for padded eager and identical SwiGLU,7.454% for both vector fusions.
The original unpadded first output was14.914%, but the reference is unstable,
so that is not a controlled quality-improvement ratio or an unchanged BF16-HF pass.

These timings exclude processor/vision/offload/DiT/VAE and do not achieve
complete encoder2s. No new49-step quality/performance run or matched warmed
A2/A3 comparison is claimed. Production still does not contain the experimental
FP32 GEMM, padding helper or native vector kernels.

## Verification, cleanup and next step

[Structural regressions](<test_repeat_contracts.py>) check default-off measured
shape, version/storage cache, unchanged non-helper AST/loaders, padded valid
mathematical channels, output-preserving hooks, full input/parameter hashes,
rejected deterministic/layout alternatives and exclusion of diagnostic checks
from repeated timing. Every raw test job completes and is collected.

[Final source verification](<final_source_verification.json>) confirms production
encoderSHA7c22d98c/noncausalFA937671ef/shared norm/communication/rotary helper
are unchanged and pins all executed round59 candidates. It also fingerprints
the reused round58 native-vector dependency. Current machine is the **910A**
requested by the user; compile SoC remainsAscend910A. Torch2.10.0/torch_npu
2.10.0.post7/transformers5.14.1 use the existing CANN9.2.0 environment.

One live fleet probe transiently times out while SSH source verification succeeds;
a second live probe confirms all8 NPUs0% and no processes. Isolated distributed
groups are destroyed, the lease is released, and no source cleanup/deployment
rollback is needed because production was never changed. GitHub publication
remains pending network reconciliation; no blind uncertain push replay is used.

The objective remains active. Next work can now evaluate Norm/RoPE rounding and
remaining text/vision reference errors against a **stable** text path, then proceed
to RoPE/AdaLN fusion, whole encoder/decoder targets and genuine task-matched video
lossless/performance validation. Repeatability is a prerequisite, not completion.
