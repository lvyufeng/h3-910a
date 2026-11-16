# Round60 — self-developed BF16 RoPE; keep native/HF rounding contracts separate

## Status

A self-developed TBE RoPE kernel now reads/writes native BF16 GM using the
previously proven integer codec. It batches all heads of each token in UB and
preserves caller BNSD layout through dense BSND storage. It does not invoke a
native BF16 vector-cast instruction or use external FP32 casts.

**Native-rounding mode** matches the current NPU RoPE on all measured actual
Q/K tensors,800 layer/rank/side checks and rank0's complete final text tensor.
**HF-rounding mode** separately matches the two-product BF16 reference on the
same inputs,800 checks and complete rank0 tensor. Both paths are repeat-stable
on all8 ranks. Native full experimental text changes roughly4.22→4.10s, not a
full encoder2s result; the larger single-call speedups are not an E2E ratio.

HF rounding is **not a whole-model accuracy fix**: final CPU FP32-linear HF-oracle
RMS worsens7.220→16.477%, despite raw-bit operator/reference agreement. No new
49-step video or official matched quality/performance comparison is claimed.
**Production remains unchanged**, and the entire goal stays active.

[Compact summary](<summary.json>) derives from completed raw receipts. The prior
stable-projection/SwiGLU evidence was verified and committed as`aa94244` at the
start of this round; network publication remains unconfirmed.

## Stable actual pre-RoPE captures

[Capture/experiment harness](<text_rope_tp8.py>) keeps the round59 stable measured
K/V output-padding, experimental FP32 text linears/causal attention, HF-math norm
option and native TBE SwiGLU fixed. No Norm fusion is enabled. Original50-layer
loaders, BF16 checkpoint storage/I-O and shared actual round54 visual/DeepStack
inputs remain in use; the unused vocabulary is seeded/removed as before.

Rank0 captures actual pre-rotation Q/K, cos/sin and native outputs at layers0/24/49.
[Metadata0](<capture_metadata/rope_layer00.json>),
[metadata24](<capture_metadata/rope_layer24.json>) and
[metadata49](<capture_metadata/rope_layer49.json>) verify inputs are unchanged by
the native call. Each complete tensor payload is explicitly below512MiB; large
blobs remain local/remote, not silently omitted numerical data in the comparison.

Actual Q shapes1×8×2043×128 and K1×1×2043×128 are retained. Q has the real
noncontiguous BNSD transpose stride, and tables are1×2043×128 BF16. The isolated
operator replay reconstructs this actual BSND-backed layout, not just an arbitrary
contiguous BNSD copy. It reproduces captured native outputs raw-bit exactly.

[Original executed capture source](<source/capture_text_rope_tp8.py>) and
[rank0 capture receipt](<text_capture/text_rope_capture_rank0.json>) are frozen.
That initial harness still copied selected pre-inputs on repeated calls even after
files existed, so its warm timing is diagnostic, not the final clean A/B. The
full experiment adds an early file-existence guard and disables all reference
checks on its second/third forwards. No capture code is installed in production.

## Two distinct reference formulas, not interchangeable lossless claims

HF's BF16 expression is:

```python
first  = (x.float() * cos.float()).bfloat16()
second = (rotate_half(x.float()) * sin.float()).bfloat16()
output = (first.float() + second.float()).bfloat16()
```

The installed native fused call instead agrees on these actual samples with:

```python
output = (x.float() * cos.float()
          + rotate_half(x.float()) * sin.float()).bfloat16()
```

On all six real Q/K cases, captured native output is **raw-bit identical** to the
single-final-round CPU formula, but differs from CPU HF two-product rounding.
Q RMS differences are0.123–0.167%, K0.0345–0.0996%; tens/hundreds of thousands of
BF16 words change. These are rounding-contract differences, not an assumed AMP,
trig-frequency or FP16-overflow cause. The tables are kept fixed.

[Initial operator receipt](<rope_validation.json>) and
[final source-matched replay](<rope_validation02.json>) preserve both references.
HF eager and HF TBE return raw-bit equal CPU HF outputs; native TBE returns
raw-bit equal actual native outputs. No reference is relabelled to hide a failure.

## Native BF16-GM TBE implementation

[RoPE kernel builder/wrapper](<rope_vector60.py>) reuses round58's exhaustively
verified BF16 codec and software RNE. For each token:

1. Load all local heads as one UB tile, without reordering logical channels.
2. Decode table values exactly, broadcast the128 coefficients to each head.
3. Rotate halves **within each head**, negating only the first half.
4. Multiply; optionally round each product for the HF contract.
5. Sum, round final BF16, pack into native GM and return the original caller shape.

Contracts have isolated hash/cache identities including token count, head count,
rounding choice and live AI-core budget. Shared compile locks/private verified
build directories avoid cross-rank collisions. Kernels compile for **Ascend910A**
and use the live30-core budget, not an unverified32-core default.

Inputs are BF16 on the same NPU with batch1, heads1/8, D128, at most4096 tokens,
matching table shapes and dense ND/NCHW storage. Unsupported shapes/training
are rejected, not silently quantized. Fresh output tensors and the existing
five-GM-pointer launch ABI/current stream are used; an auxiliary immutable input
copy fills that ABI. There is no per-call global sync in the kernel wrapper.

This is not a generic/autograd/cross-stream/arbitrary-nonfinite-service
certification; the local cache/dependency paths and limited existing launch-table
capacity are explicit prototype constraints. All required inputs in the real
checks are finite. **No production rotary helper was rewritten.**

## Final complete-wrapper same-input timings

[Final validation](<rope_validation02.json>) uses three groups of10 warmed calls.
It includes dense-layout handling, cache lookup, output allocation and return view;
not an isolated device-only event. Both actual/native and exact HF references are
checked on every element.

| Layer/side | Native NPU | HF eager | TBE native | TBE HF |
|---|---:|---:|---:|---:|
|0 Q|2.047ms|7.223ms|0.201ms|0.200ms|
|0 K|1.010ms|2.320ms|0.203ms|0.198ms|
|24 Q|2.035ms|7.161ms|0.203ms|0.207ms|
|24 K|0.982ms|2.287ms|0.194ms|0.203ms|
|49 Q|2.034ms|7.138ms|0.198ms|0.198ms|
|49 K|0.974ms|2.270ms|0.191ms|0.190ms|

Native Q is about10× faster and K about5× in this same-input wrapper measurement;
HF TBE is about35×/11× vs its slower eager reference. Those are **operator** ratios,
not claims about the original39s encoder, full generation or hardware ceilings.

Short/tail cases N1/7/129 with heads1/8 all return raw-bit equal appropriate
native/HF references. The codec's65536-word proof is a **storage conversion**
proof from round58, not proof of every possible RoPE arithmetic input.

## Complete resident TP8 verification

[Full harness](<text_rope_tp8.py>) runs all50 layers with the measured stable
padding and native SwiGLU fixed. Only apply-RoPE changes among native, HF eager,
TBE native and TBE HF. Every8-rank case finishes, all tensors finite, and every
rank's full first/second/third outputs are raw-bit equal to its own prior forward.

During each TBE mode's first forward, every layer/side is independently compared
against its correct same-input reference: native NPU for native mode, protected
HF eager for HF mode. All **1600 layer/rank/side checks** are raw-bit equal across
the two contracts, covering **1882828800 scalar outputs**. These extra calls and
CPU copies are disabled for the repeated timing; first calls include checks/
compile/cache overhead and are not comparable clean speed samples.

| Fixed experimental text stack | Second rank-max | Third rank-max | Rank0 full final output identity | RMS vs CPU FP32-linear HF oracle |
|---|---:|---:|---|---:|
|native apply-RoPE|4.218s|4.235s|reference|7.220%|
|HF eager apply-RoPE|4.465s|4.530s|different contract|16.477%|
|TBE native|4.096s|4.109s|**bit-exact native**|7.220%|
|TBE HF|4.097s|4.101s|**bit-exact HF eager**|16.477%|

[Direct final-tensor proof](<full_output_contract_comparison.json>) reads saved
rank0 outputs and verifies each TBE mode against its own reference, not only
matching summary errors. [Rank0 raw receipt](<text_rope/text_rope_rank0.json>) and
[rank7 raw receipt](<text_rope/text_rope_rank7.json>) plus all other rank receipts
retain every local check and full-repeat comparison.

The HF formula's separately correct rounding **worsens the existing whole-oracle
comparison**. This strong stable counterexample prevents calling a single-operator
contract fix a complete encoder precision fix, unique error cause or video quality
improvement. The remaining reference uses CPU FP32-linear/BF16-I-O and shared
actual visual inputs, not an unchanged BF16 CPU-HF/full multimodal teacher.

Timings exclude processor/vision/offload/DiT/VAE and still exceed complete-text2s.
No new49-step video or official warmed task-matched benchmark is run. The norm
fusion's prior6.318% full-text difference also remains unaccepted.

## Final state and continuation

[Structural contracts](<test_rope_contracts.py>) check per-head half rotation,
explicit product rounds, no external cast, native layout/coeff broadcast, rounding/
shape/core cache separation and bounded capture/clean repeat timing.
[Final source verification](<final_source_verification.json>) pins executed
prototypes and unchanged encoder/noncausal FA/shared norm/communication/rotary
sources. It also pins round58 codec/RNE dependencies. Existing Torch2.10.0/
torch_npu2.10.0.post7/transformers5.14.1 source CANN9.2.0; compile SoC is910A,
not reclassified from the fleet's device label.

All relevant jobs are collected, isolated distributed groups stopped, all8 cards
live-probed idle and lease released. New paths remain diagnostic; no replacement
server, package install or production precision change is made. Local commits
still require GitHub remote reconciliation/publication after the prior SSL failures.

The goal remains active for AdaLN fusion, stable reference precision, full
encoder/decoder targets and validated task-matched official lossless generation.
Native operator identity is concrete progress, not completion of the whole objective.
