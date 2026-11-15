# Round 56 — isolate model-specific text Norm/RoPE errors before fusion

## What is and is not established

The objective is still active. This round validates **operator contracts**, locates
the previously observed large late-text error **before** the final layer, and
rejects an accuracy-worsening GEMM combination. It does not claim a full encoder,
49-step video-quality/lossless pass, encoder2s, decoder20s, or new fused-kernel
speedup. No ComfyUI path is involved.

All references use the same actual round54 processor/text/visual captures unless
stated otherwise. CPU teachers keep original BF16 parameters and layer outputs
but use labelled **FP32 linear computation**. They are not the unchanged BF16 CPU
HF implementation. Vision features are shared captured native inputs, not an
independent whole multimodal validation. H3's selected layer50 is compared before
HF's final model norm.

Compact receipts/code are versioned; large tensor binaries and complete logs stay
local/remote. Read [the prior scope](<../round55/README.md>) before treating this
round's isolated improvements as full-model correctness.

## Four complete CPU norm/RoPE ablations

[Reproducer](<text_norm_rope_ab.py>) runs all50 retained layers and independently
changes the HF-vs-native norm formula and HF-vs-captured-native RoPE tables.
The shared inputs are identical; linears compute in FP32 then return BF16 in all
four cases. Final CPU input/output tensors are saved (~tens of MB, bounded below
512MiB) for teacher-forced comparison. Late-layer metrics24…49 are preserved.

| CPU oracle settings | RMS vs actual old native final output | Visual-token RMS | Maxabs difference |
|---|---:|---:|---:|
| HF norm + HF tables | **16.558%** | 17.999% | 6464 |
| Native norm + HF tables | **4.086%** | 4.412% | 1408 |
| HF norm + captured native tables | **8.560%** | 9.261% | 2560 |
| Native norm + captured native tables | **12.384%** | 13.440% | 4416 |

These non-additive results demonstrate sensitivity to small changes; they do not
prove either difference alone is the complete cause, nor does closer agreement
to the old native output mean better reference accuracy. Each teacher forward is
~315s in concurrent CPU runs, **not deployment text performance**.

Receipts: [HF](<cpu_ab_hf.json>), [native norm](<cpu_ab_norm-native.json>),
[native RoPE](<cpu_ab_rope-native.json>) and [both](<cpu_ab_both-native.json>).

## Actual RoPE error is the K=1 NPU matrix multiply

[Context probe](<autocast_probe.json>) reproduces actual captured RoPE tables
**bit-exactly even without autocast**. FP16 autocast introduces an additional
difference; NPU BF16 autocast is unsupported in this stack and the request to
use it is disabled with a warning. Therefore the captured error cannot simply
be blamed on an assumed global autocast context.

[Stage isolation](<rope_stages.json>) checks actual positions/configuration:

- Inverse frequencies are **bit-exact** to CPU HF.
- Batched **K=1 matrix multiply** frequency maxabs error is **0.020401**; making
  both inputs contiguous does not fix it.
- The mathematically equivalent **FP32 elementwise outer product** makes frequencies
  **bit-exact**. Explicit modality-mask `where` preserves the HF interleaving.
- BF16 cos/sin of those corrected frequencies are **raw-bit equal** to HF.
  Passing exact CPU frequencies to NPU trig also gives exact tables, so trig is
  not this sample's error source.

The candidate uses direct outer products and protects FP32 computation from
inherited autocast. [Actual/multi-batch NPU tests](<text_math_validation.json>) check
both unautocast and inherited FP16-autocast contexts, obtaining raw-byte equality
for all tested tables. This is a precise operation diagnosis, not a universal
hardware limit or proof for all possible contexts/configurations.

## Focused real late-input capture and repeatability

A **one-step native FL2VA diagnostic** reuses the same images/prompt/seed, default
BF16 encoder storage, and disables optional vision compute to reproduce round54.
It does not optimize or qualify video quality and is not a49-step benchmark.
[Raw request](<e2e_capture/result.json>) completes in545.39s with1 denoise step;
capture synchronization/copy overhead makes its stage costs diagnostic only.

The default-off [helper](<h3_text_audit56.py>) saves text input, RoPE, actual layer49
input/output, two norms, attention, packed gate/up and reduced down output. All
10 tensor captures have detached write-once output and an explicit512MiB cap.
[CPU hook tests](<capture_hook_tests.json>) check no default-off hooks/writes,
identical first/second module outputs, detachment, byte caps and no Boolean
output replacement. Six structural capture/math tests additionally restrict scope.
The helper is temporary and unreferenced after restoration.

[Actual context summary](<capture_summary.json>) shows **autocast disabled**,
default deviceCPU, BF16 tensors, norm/activation fallback flags1. Initial text
embeddings, positions and all3 DeepStack tensors are **raw-bit equal** to round54;
final native output is also **raw-bit equal**. This directly validates same-input
repeatability rather than inferring context from source comments. Metadata receipts
are in [the capture directory](<capture_metadata/text_input.json>).

## Large difference exists before the final layer

[Teacher-forced last layer](<last_layer_reference.json>) uses its **actual native
input** and original weights. Comparison with the complete HF oracle shows:

- Actual last-layer input already differs by **16.702% RMS**, maxabs6464.
- Complete last layer on that identical actual input, native norm/captured RoPE,
  CPU FP32-linear oracle vs actual native output: only **0.0686% RMS**, maxabs1.
- Using HF norm/captured RoPE: **0.0842%**; HF norm/HF RoPE: **0.0840%** vs old native.
- A CPU **BF16-linear** final-layer repeat agrees with the FP32 oracle within
  **0.0280% RMS** and matches native within0.0687%. Its88.85s CPU runtime is not
  an NPU benchmark. The late difference is not explained merely by changing this
  last-layer CPU linear precision.

Thus the prior16.56% discrepancy is **not localized to a broken final block**.
The output carries an upstream difference. Isolated attention on identical actual
norm input differs by1.14% (native q/k norm), while rank0 packed gate/up differs
by0.00565%. The reduced MLP down output differs0.2468% from an unsharded FP32 oracle;
FP32 summing BF16 CPU partials differs only0.0513% from actual reduced down. These
are different comparison points, not additive causes or proof that TP sum is absent.

[Last-layer reference source](<last_layer_reference.py>) preserves all these scopes.

## Model-specific RMSNorm/AddRMSNorm rounding, not a shared norm rewrite

The shared fallback multiplies FP32 normalized values/gamma before output cast;
HF Qwen first rounds normalized values to BF16 and then multiplies by BF16 gamma.
The tested NPU path preserves this two-rounding formula, using FP32 point math
between explicit BF16 rounding boundaries. Residual addition is also rounded to
BF16 before normalization. Other devices/non-BF16 behavior and shared DiT norm
are unchanged.

On actual final-layer norm inputs, [NPU validation](<text_math_validation.json>):

| Same actual norm input | Shared fallback RMS vs HF | Correct rounding RMS vs HF |
|---|---:|---:|
| layer49 input norm | **0.27825%** | **0.000948%** |
| layer49 residual/post-attention norm | **0.28138%** | **0.001530%** |

The updated BF16 residual is raw-bit equal to HF. Adversarial zero/constant/outlier
rows plus near-rounding-boundary residual are raw-bit equal for both residual
and output in the tested sample. Default-off matches the original NPU result
exactly. This verifies actual dispatch to the model-specific override, not only
a mocked CPU formula. It is **not** a fully fused AscendC/TBE implementation yet;
these contracts define the rounding a later custom kernel must retain.

## Embedding-only FP32 promotion: row reductions were already FP32

The actual coordinator promotes BF16 to FP16 for HCCL. However **text attention
and MLP row projections already convert partial outputs to FP32 before calling it**.
They are unchanged by the model-specific embedding fix. An earlier broad suspicion
about all text reductions is narrowed by reading the actual call sites; do not
claim this conversion is a proven cause of the large final outliers.

Eight real HCCL ranks validate the embedding-only helper:
[rank0](<embedding_rank0.json>) through [rank7](<embedding_rank7.json>) all pass.
Actual old embedding error is only1.98e-7 relative RMS (max2.98e-8); FP32 promotion
returns raw-bit exact values. Tiny2.33e-10 BF16 values lost by FP16 and finite131072
values overflowing FP16 are retained exactly. These adversarial examples prove
range preservation, **not actual embedding overflow** in this request. Disabled
helper is byte-identical to the existing path. Shared coordinator code is unchanged.

## Complete text-only TP8 verification; target remains far from2s

[Text-only harness](<text_only_tp8.py>) uses original50-layer weight loaders, actual
Omni parallel initialization and effective NPU fused-RoPE/native-GQA/packed-SwiGLU
patches. All BF16 parameters fit resident; captured input/visual/DeepStack tensors
bypass vision and unused vocabulary storage. It is not a full encoder stage/E2E
measurement, and order/cache effects are not a speed attribution.

The [successful third attempt](<text_only_attempt03/text_only_attempt03_rank0.json>)
completes all8 ranks. Disabled candidate's final output is **raw-bit identical**
to the old real native capture. Baseline first forward196.364s (rank maximum),
HF-math option's subsequent forward183.574s; the latter still differs **13.558% RMS**
from the CPU FP32-linear HF oracle. Finiteness/runtime success is not a quality pass.
This round does not claim the option's speedup from these fixed-order first calls.

The first harness attempt lacked vLLM TP metadata initialization; the second
completed the forward but its **BF16 NPU IsFinite diagnostic is unsupported**.
Both failures are preserved in [attempt1](<text_only_attempt01/text_only_rank0.json>)
and [attempt2](<text_only_attempt02/text_only_attempt02_rank0.json>) with executed
sources. The third uses actual Omni groups and CPU FP32 finite checks, not a new
server/package installation or policy workaround.

## Rejected FP32 text-GEMM prototype

A separate [experimental candidate](<candidate_text_linear_encoder.py>) computes
nonquantized text GEMMs in FP32, preserves BF16 local outputs and keeps row
partials FP32 **until after HCCL**. Existing packed activation/attention is retained.
Parameter constructors/loaders and quantized paths are checked by four extra
structural tests. No global device precision settings or shared communication
code is changed.

[Raw8-rank results](<text_fp32/text_fp32_rank0.json>):

| Complete resident50-layer text path | First rank-max forward | Second rank-max forward | RMS vs CPU FP32 HF oracle |
|---|---:|---:|---:|
| FP32 GEMMs only, old norm/RoPE | 116.888s | **76.341s** | **11.595%** |
| FP32 GEMMs + HF norm/RoPE | 79.437s | **76.942s** | **31.876%** |

The combination is **worse numerically** despite separately verified operators.
Neither path reaches2s. Cold/warm and different executable/cache order limit direct
speed ratios; no clean reversed-order whole-model A/B or quality pass is inferred.
This prototype is **NOT DEPLOYED**. Keep its counterexample visible and investigate
earlier text attention/accumulation before recommending or enabling it.

## Final installed state

Only the tested **default-off** `VLLM_OMNI_910A_TEXT_HF_MATH=1` option is installed:
model-specific two-rounding norm, elementwise RoPE, embedding-only promotion;
existing optional vision FP32 linear also receives inherited-autocast protection.
The rejected text-GEMM flag does not exist in production. Source hash is
`7c22d98c28f4e93550fd3848bfc174121250259fdf50bcd2f7920f826acf311d`.
[Deployment](<text_math_deployment.json>) preserves the exact pre-capture and
instrumented backups, removes capture imports, and leaves the helper unreferenced.
[Portable diff](<../../design/patches/encoder_text_hf_math_optin.patch>) reproduces
this installed source exactly; shared norm/coordinator and prefix FA stay unchanged.
[Final source/runtime verification](<final_source_verification.json>) records their
hashes and torch2.10.0+cpu/torch_npu2.10.0.post7/transformers5.14.1. Existing
CANN9.2.0 is sourced for all remote Torch work; no packages or replacement servers
are installed.

All required jobs are collected, owned server/process groups stopped and all8 cards
live-probed idle; lease released at round completion. Next work remains actual
text-attention Q/K/V/timing capture, locating the earliest upstream divergence,
lossless-safe custom causal FA/norm/activation fusion, and encoder/decoder targets.
There is no task-matched official quality/performance comparison yet and the goal
is deliberately still active.
