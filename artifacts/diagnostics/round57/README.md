# Round57 — causal text attention is the measured bottleneck; TBE prototype verified

## Status and boundary

No production source is changed this round. All new causal/eager/GEMM combinations
are **experimental**, not full-model/video quality or lossless passes. The existing
production exact-prefix noncausal DiT wrapper and default-off text math option
remain intact. Target text2s, decoder20s and task-matched official comparison
remain unmet. No ComfyUI path is used.

[Attention probe](<text_attention_probe.py>) uses the real round54 multimodal text
inputs, actual50-layer TP8 loaders/effective NPU patches, and the round56 **uninstalled
FP32-linear prototype**. Thus its full stack is not the unchanged production encoder.
BF16 parameters/outputs are retained. All rank records are saved; rank0 captures
only five attention calls. Operator synchronization changes overlap, so the timing
is diagnostic rather than a clean end-to-end deployment A/B.

## Actual text attention dominates the current experimental forward

At8 ranks the measured50-layer text forward is **76.1639s rank maximum**. Causal
attention sum is **70.409–71.843s** across ranks, about92.4–94.3% of the local wall.
This shows a concrete kernel path bottleneck, not a physical 910A compute ceiling.

Rank0 actual query shapes are **[1,8,2043,128]**, KV **[1,1,2043,128]**, BF16.
Selected layers0,24,40,48,49 each take **1.435–1.439s** in the effective native GQA
call. Last-layer module timings are only about5.7ms input norm,6.7ms output
projection,12.1ms post norm,10.6ms gate/up and9.0ms down. Nonquantized Q/K/V use
separate local F.linear calls and bypass the qkv-module forward hook, so no qkv
module timing is invented.

[Rank0 receipt](<attention_probe/attention_probe_rank0.json>) and
[rank7 receipt](<attention_probe/attention_probe_rank7.json>) bracket the diagnostic
range; all8 rank records exist. Q/K/V/output captures are detached contiguous CPU
copies bounded below512MiB per file. They are not directly the original NPU tensor
strides; that affects the standalone SDPA reproducer below.

## Complete real causal FP32 reference

[Same-input reference](<attention_reference.py>) checks all query rows, all8 local
query heads and every causally valid key on each of the five captured layers.
There is no sparse approximation or future-token inclusion. CPU full-key FP32
matmul/softmax/PV is the numerical oracle. Outputs remain BF16 where compared;
conversion/explicit KV repeats/masks are included in wrapper timings.

| Captured text layer | Native captured RMS vs CPU FP32 | Explicit NPU FP32 math RMS | FP32 call median |
|---|---:|---:|---:|
| 0 |0.3785%|0.1611%|5.984ms|
| 24 |0.8386%|0.1642%|5.862ms|
| 40 |1.1726%|0.1656%|5.879ms|
| 48 |1.0164%|0.1657%|5.952ms|
| 49 |0.8462%|0.1654%|5.847ms|

Remaining error includes BF16 output rounding and FP32 ordering; there is **no
bitwise lossless equivalence claim**. The CPU reference divides by sqrt(D), while
NPU math multiplies by reciprocal sqrt(D), another explicit ordering difference.
These isolated improvements do not automatically imply full encoder equality.

In this contiguous-copy standalone replay, nativeBF16, FP32-SDPA and FP16-SDPA GQA
calls all fail `can not cast format when output is input`. Those failures are
preserved in [raw reference evidence](<attention_reference.json>); do not claim
native attention generally cannot run, since the full runtime probe did run it.
The explicit FP32/FP16 math variants remain finite and execute. The additional
[layout probe](<sdpa_layout_probe.json>) also fails for reconstructed transpose views
and explicit repeated KV. These retries do not establish a stride-only cause.
An optional [format-policy probe](<sdpa_format_policy_probe_failure.json>) fails
before any change because this torch_npu version exposes no `allow_internal_format`
attribute on its config object. No flags/packages/production policy are changed;
actual standalone-vs-runtime differences remain unisolated.

## Self-developed causal TBE mask/prototype

A separate [v4 causal TBE wrapper](<candidate_causal_fa910a.py>) adapts the existing
Ascend910A dense TIK kernel, using30-core repartitioned task geometry, square
Br/Bc tiles, compile-time upper-triangular invalid-column masks and exact
zero probabilities after exp. Actual2043-token queries are padded to2048 and
all added keys are masked using the true prefix. Diagonal keys remain valid.
No extra dense mask argument is required by the launch ABI.

Causal mode has an isolated `_causal_v4` cache/metadata identity and cannot alias
production noncausal artifacts. [Structural tests](<test_causal_contracts.py>)
check all128 row masks, default-off bounded eager path, cache version, and hooks
that never replace module outputs. Three **real NPU uniform-prefix tests** at
(N,valid)=(128,127),(256,243),(512,511), with large invalid-tail values, match the
causal prefix mean **exactly**. These cover diagonal boundaries and true tail
masking, not just random input arithmetic.

[Kernel validation](<causal_kernel_validation.json>) also checks actual layer49:
finite FP16 TBE output RMS error0.0817% vs CPU FP32. This cold call includes
**145.81s compilation** and is not a steady kernel time. The existing TIK
score/probability workspace is still FP16; the prototype is not BF16-lossless.

## Warm complete-wrapper real QKV A/B

[Complete wrapper A/B](<causal_kernel_real_ab.json>) includes BF16→FP16 casts, explicit
GQA KV expansion, tail clear/pad, NZ format copies, Q scale, crop and BF16 return.
Three groups of five warmed calls, fixed order per layer:

| Actual layer | FP32 eager median | Causal TBE median | BF16 TBE output RMS vs CPU FP32 |
|---|---:|---:|---:|
|0|5.732ms|3.674ms|0.1677%|
|24|5.766ms|3.673ms|0.1972%|
|40|5.720ms|3.642ms|0.2100%|
|48|5.716ms|3.654ms|0.1971%|
|49|5.725ms|3.627ms|0.1865%|

This is a1.56–1.58× advantage over the FP32 eager operator in this limited
same-input test, not a clean whole-model lossless speedup or a fixed hardware
ceiling. Float16 TBE output before BF16 conversion and BF16 returned output are
separate numerical comparisons; do not conflate0.0817% and0.1865%.

[Reproducer](<causal_kernel_real_ab.py>) and
[experimental integration wrapper](<h3_text_causal_tbe57.py>) are retained only under
round57 diagnostic paths. **Production noncausal FA is unmodified.**

## Complete resident50-layer text stack comparison — speed is real, quality is not solved

Both [FP32 eager](<text_causal_tp8.py>) and [TBE stack](<text_causal_tbe_tp8.py>) keep
BF16 checkpoint storage/I-O but use experimental FP32 text linears and FP32 row
partials from round56. Captured visual/DeepStack features are shared. Each option
has a first and repeated forward; fixed order/compiler caches prevent interpreting
cold differences as an isolated kernel A/B or E2E speed ratio.

| Full experimental resident text stack | First rank-max call | Repeated rank-max call | Final RMS vs CPU FP32-HF oracle |
|---|---:|---:|---:|
|FP32 causal, old norm/RoPE|18.996s|**4.533s**|15.616%|
|FP32 causal, HF-math option|8.201s|**5.134s**|11.257%|
|TBE causal, old norm/RoPE|18.827s|**4.409s**|11.495%|
|TBE causal, HF-math option|8.058s|**4.954s**|**42.179%**|

All8 ranks finish, every final tensor is finite, but the last combination is
**numerically much worse** despite separately accurate captured QKV operators.
The table is a counterexample to promoting isolated error/time gains as
whole-model lossless validation. Nothing is deployed or recommended as a new
production default. It also does not include vision/preprocessing and does not
achieve complete text encoding2s.

[FP32 rank0](<text_causal_rank0.json>) and [TBE rank0](<text_causal_tbe_rank0.json>)
receipts include final-error comparisons to both native captures and CPU oracle;
all other rank timing receipts are available in this directory.

## Verified v5 prefix traversal — isolated kernel only

The initial causal v4 computes wholly future tiles then masks them away.
The [v5 traversal](<candidate_causal_fast_fa910a.py>) sets future-block count0 and
uses **prefix-work-aware** static core partitioning, with isolated `_causal_v5`
cache/metadata. Its [real NPU validation](<causal_fast_validation.json>) again
returns exact uniform-prefix means on all three boundary cases. Actual FP16
layer49 output retains the same0.0817% reference RMS. Cold compile+call113.82s
is not steady runtime; warmed packed-NZ/core calls are1.000/0.582/0.575ms.

A separate [full wrapper A/B](<causal_fast_real_ab.json>) includes all casts/repeats/
pad/clear/NZ copies/crop/BF16 return. Across the five real layers the v5 wrapper
medians are **3.332–3.393ms**, FP32 eager medians5.760–5.796ms. Errors remain the
same0.1677–0.2100% BF16-output RMS as v4. These fixed-order isolated observations
do not prove the full-text v4 table above applies unchanged to v5 or establish
lossless equivalence. **No v5 complete text-stack integration test is claimed.**

The two causal TBE versions are real measured prototype code, not a deployed
production replacement. Their FP16 score/probability workspace and BF16 final-stack
counterexample need attention before adoption. All new helpers/candidates are
kept in the round57 diagnostic path; the original production encoder/noncausal
FA and shared norm/communication remain byte-identical.

## Final state and remaining work

[Final source verification](<final_source_verification.json>) confirms production
encoderSHA `7c22d98c...` and original noncausal wrapperSHA `937671ef...`; no new
causal helper is installed into production. All relevant jobs are collected,
owned distributed groups stopped and cards live-probed idle; lease released at
round completion. Capture/timing hooks were only attached to isolated text-only
models, not installed in the production source.

Further precision/reference agreement, per-model fused RMSNorm/AddRMSNorm,
SwiGLU/RoPE/AdaLN kernels, residual communication/launch costs, full encoder2s,
decoder20s and official task-matched quality/performance remain unresolved.
A read-only GitHub remote-head check again times out; prior commit9967e31 remains
locally authoritative, remote publication unconfirmed. No blind write replay or
alternative transport is attempted. The goal stays active for these measurable
remaining tasks, not blocked by a speculative physical limit.
