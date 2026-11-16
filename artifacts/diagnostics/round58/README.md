# Round58 — native BF16-GM TBE fusion: RMSNorm, AddRMSNorm and SwiGLU

## Status and limits

Self-developed vector kernels now perform **real 910A fusion**, including direct
BF16 GM input/output using an integer codec. They do not call a missing native
BF16 cast, native AddRMSNorm or native SwiGLU API. **Production is unchanged**;
all new paths remain isolated diagnostic prototypes, not an adopted full-model
lossless pass. No ComfyUI or package installation is involved.

Same-input native wrapper speedups are **18.5× RMSNorm,24.4× AddRMSNorm,34.0×
SwiGLU**. SwiGLU matches the **actual NPU reference** bitwise on an exhaustive
finite-gate sweep and800 checked real layer/rank calls. Norm still changes a few
BF16 words. A complete experimental text stack repeats in **2.709s**, but even its
**unchanged baseline is not repeat-stable**, so no clean whole-model quality or
encoder2s conclusion is drawn. Decoder20s and official task-matched video
quality/performance remain unmet; the goal stays active.

[Compact summary](<summary.json>) derives only from the versioned raw receipts.
[Prototype policy](<prototype_policy.json>) explicitly preserves these limits.

## Arithmetic contracts before optimization

The Qwen contract rounds the BF16 residual sum **before normalization**, then
rounds normalized values to BF16 **before multiplying gamma**, then rounds output.
The shared DiT-style norm instead multiplies normalized FP32/gamma before the
final output cast. Both contracts are represented separately; the shared norm
implementation is not rewritten.

SwiGLU on BF16 rounds `silu(gate)` to BF16 before multiplying BF16 up. The kernels
retain that boundary, rather than silently computing one different FP32 fused
expression. Stable sigmoid uses `exp(-abs(gate))`, and two Newton corrections
refine the reciprocal. FP16/FP32 boundary variants are also tested, but are not
claimed to be globally bit-identical.

[FP32-boundary builder](<fused_vector58.py>) uses a FP32 reduction and explicit
software BF16 round-to-nearest-even. 910A supports int32 add/conversion and uint16
AND, but not the queried native BF16 vector cast/shift primitives. RNE therefore
extracts bit16 with halfword AND, converts exact0/65536 to0/1, adds0x7fff+LSB to
raw bits, and clears the low halfword. [195840 finite midpoint tests](<exhaustive_npu_reference03.json>)
are bit-exact, including both signs, just-below/tie/just-above boundaries and4
valid BF16 infinity outputs from finite FP32 inputs.

## Native BF16 codec removes the external conversions

The first [floating wrapper](<fused_ops58.py>) still used external Torch BF16 casts;
its actual norm/activation medians were5.327/10.012/12.839ms against
9.943/16.101/15.616ms eager. See [the completed first real comparison](<attempt04/vector_real_validation04.json>).
That is only partial fusion and not the final performance result.

[Native BF16 builder](<native_bf16_vector58.py>) instead declares BF16 storage as
uint16 GM words and unpacks each word into a FP32 high halfword with a zero low
halfword using **three matrix transposes**. Three inverse transposes pack explicitly
rounded results back into BF16 GM. This preserves the ordinary channel order,
not a permuted approximation, and eliminates external dtype casts. Packed
SwiGLU reads gate/up directly from the two halves of the original GEMM output.

[Real codec proof](<native_attempt01/native_codec_validation.json>) covers **every
65536 BF16 bit pattern** on each tested width128/3200/5120, including signed
zero, infinity and all NaN payloads. Both unpacked FP32 bits and repacked BF16
bits are exact. This codec proof does **not** certify arbitrary nonfinite norm/
activation arithmetic. Partial widths128 and3200 are padded to256 and3328 inside
UB; the **true width**, not padded width, controls the mean and GM stores.

The launcher reuses the existing five-GM-pointer C ABI, enqueues on the current
Torch stream, and uses fresh output tensors. No per-call synchronization is added
inside the wrappers. Binary identity includes source hashes, shape, dtype/contract,
epsilon and the **live30-core budget**; a shared file lock protects cross-rank
compilation/publication. BF16 inputs must have dense ND/NCHW storage, compatible
weight/residual shapes and equal packed halves. This is not a generalized concurrent
service/autograd/quantized-kernel certification; the existing launcher has a limited
registration table and the diagnostic cache directory is intentionally local.

## Final real same-input wrapper timings

[Final source-matched receipt](<native_real_validation03.json>) uses three groups
of10 warmed calls per variant on the real captured rank0 late-layer inputs.
Timings include registration/cache lookup and output allocation, but not a new
compile; the final binaries were already cached. Earlier uncached builds took
about2.6–2.8s in [the saturation-corrected run](<native_attempt02/native_real_validation02.json>).

| Input/operator | Eager median | Native BF16 fused median | Ratio | CPU-reference output RMS |
|---|---:|---:|---:|---:|
|actual layer49 RMSNorm,1×2043×5120|9.958ms|**0.538ms**|18.5×|0.001291%|
|actual layer49 AddRMSNorm, same size|16.151ms|**0.662ms**|24.4×|0.002044%|
|actual packed SwiGLU,1×2043×6400|15.415ms|**0.454ms**|34.0×|**bit-exact**|
|post-RoPE Q reused as same-shape RMS input,1×2043×8×128|3.097ms|0.850ms|3.65×|0.000995%|

The last row uses real post-RoPE Q **as a test input**, not actual pre-QNorm
captures, and is not a whole-attention timing. Native final SwiGLU matches both
CPU and NPU references on all **6537600 actual output elements**. Actual updated
AddRMSNorm residual is bit-exact. Norm output has95/156 changed words vs CPU and
74/82 vs NPU out of10460160; those are tiny measured differences, **not bitwise
lossless norm completion**. Fixed-order microbenchmarks are not a video speed ratio.

## Finite-overflow counterexample: diagnose, do not infer a hardware ceiling

The first native norm reproduced ordinary samples but failed huge finite BF16
outliers near1e20. [Preserved counterexample](<native_attempt01/native_real_validation.json>)
shows maximum error196, not a hidden successful test.

[Stage isolation](<native_attempt01/overflow_stages.json>) proves the relevant TIK
FP32 vector square/reduce-add **saturate overflow to maximum finite FP32** in this
sample, while the CPU teacher produces infinity. For BF16≈1e19, individual squares
are finite but their128-element sum already saturates; for≈1e20, square and sum
both saturate. Dividing that marker by128 creates a false finite inverse.
The raw reciprocal-square-root estimate is also approximate (998 vs CPU1000 on
epsilon1e-6), hence the Newton corrections are necessary.

The final norm treats this observed saturated-sum marker as overflow and returns
a zero inverse, matching the CPU reference on the tested positive/negative
outliers. [Final boundary cases](<native_real_validation03.json>) then match raw bits,
including signed zeros, BF16 min-subnormal,2^-100,131072,20736,1e19,1e20 and2^64.
**Exactly legitimately-maximal finite sums and arbitrary nonfinite arithmetic are
not certified.** This is a scoped compatibility guard, not a general CPU/NPU
IEEE equivalence claim or a statement that910A FP16 paper performance is inadequate.

## Choose the real NPU SwiGLU reference, not CPU overflow behavior

The earlier [CPU sweep](<exhaustive_attempt01/exhaustive_round_activation.json>) found
85 mismatches in522240 outputs, despite the actual moderate-range tensor being
bit-exact. [Counterexample summary](<exhaustive_reference_summary.json>) identifies
17 gate values−89…−97 and large up factors. CPU BF16 SiLU returns zero in that
implementation's exp-overflow regime, whereas the native NPU retains the tiny
term. CPU/NPU SiLU also differ in many zero signs. Do **not** erase the correct
NPU tiny values merely to match that CPU comparison.

After preserving the **native NPU multiply's zero behavior**, the final
[exhaustive actual-NPU comparison](<exhaustive_npu_reference03.json>) is **raw-bit
exact on every522240 output**, using65280 finite BF16 gate words×8 up factors,
including min-subnormal and maximum finite BF16 up.33375 valid infinity outputs
are retained, with no nonfinite inputs. CPU comparison still has85 numerical
and127488 signed-zero differences. The scalar CPU-vs-NPU SiLU comparison has17
numerical and15537 zero-sign differences; the reference distinction stays visible.
This is strong finite BF16/operator evidence, not every4-billion input pair,
arbitrary nonfinite input, or complete-video equivalence.

## Complete resident text-only TP8 — verify local identity, expose repeat instability

[Controlled full-text harness](<text_vectors_tp8.py>) retains original50-layer loaders,
BF16 checkpoint storage/I-O, captured round54 visual/DeepStack/text inputs, actual
Omni groups and native apply-RoPE. **Experimental FP32 linears/causal attention and
HF math remain fixed** across cases; only vector fusion changes. The unused
vocabulary is seeded for loader completeness then removed. This is not the
unchanged production encoder, independent vision, processor/offload cost or E2E.

All8 ranks finish.200 norm modules are wrapped only inside this isolated model.
On the first SwiGLU-enabled forward, every layer independently compares native
SwiGLU to same-input NPU eager and checks finite packed inputs. **All800 layer/rank
calls are bit-exact** across SwiGLU-only and norm+SwiGLU cases: **5230080000 checked
scalar outputs**. These extra eager calls/copies are disabled before repeated
performance forwards. First calls contain compilation/check overhead and must not
be used as clean speed/quality attribution.

| Same resident experimental text stack | First rank-max (may contain checks) | Repeated rank-max | Rank0 first output RMS vs CPU FP32-linear HF oracle |
|---|---:|---:|---:|
|baseline|20.130s|**4.971s**|7.904%|
|native SwiGLU only|12.098s|**4.179s**|10.259%|
|native norm only|12.134s|**3.490s**|17.721%|
|both native vector fusions|8.239s|**2.709s**|21.485%|

Crucially, even the **unchanged baseline's first vs repeated output differs6.718%
RMS**; other cases differ6.580–13.097%. These are explicit newly measured
repeatability failures, not proof that the new SwiGLU introduces error (all
same-input SwiGLU checks pass), nor a settled cause attribution to HCCL, GEMM,
input mutation or a last layer. Source inspection alone does not establish input
immutability. There is **no whole-model lossless acceptance or causal quality
conclusion** from this fixed-order/cold-vs-warm table. The2.709s run still misses2s
and excludes vision/preprocessing; no production deployment is made.

[Rank0 raw receipt](<text_vectors/text_vectors_rank0.json>) and
[rank7 raw receipt](<text_vectors/text_vectors_rank7.json>) retain timing, first/
repeat differences, same-run first-baseline comparisons and every local check.
All8 rank receipts are versioned. Next measurement must freeze/verify real input
and parameter state, capture the earliest repeated-forward divergence and isolate
reduction/stream/kernel behavior before adopting the experimental full stack.

## Errors retained and final state

- [Attempt1](<attempt01/kernel_smoke.json>) verifies software RNE but compilation
  stops on unsupported Tensor×Python-float scalar expression. Replaced by explicit
  vector scalar operations, not a package install.
- [Attempt2](<attempt02/kernel_smoke02.json>) completes small norm/activation tests;
  zero signs are examined separately rather than treating zero RMS as bit equality.
- [Attempt3](<attempt03/vector_real_validation03.json>) validates actual norms but
  fails FP16 codegen: the UB variable name `half` shadows the compiler type.
  Rename to `fp16_round_ub`; [attempt4](<attempt04/vector_real_validation04.json>) then
  completes BF16/FP16/FP32 norm/activation cases. These floating-boundary prototypes
  are distinct from final native BF16 GM fusion.
- [Structural tests](<test_vector_contracts.py>) cover channel ordering/inverse codec,
  all195840 midpoint decisions, width/padding, source/cache/lock identity, native
  zero policy, storage/residual constraints, saturation limits and exclusion of
  reference checks from warm timing.

[Final source verification](<final_source_verification.json>) confirms encoder,
noncausal FA, shared norm and communication are unchanged; native helpers and the
FP32-linear prototype are **not installed in production**. Hashes also pin the
exact final native/floating builders, wrappers and complete-stack script. Runtime
package metadata is torch2.10.0/torch_npu2.10.0.post7/transformers5.14.1 on aarch64;
GPU tasks source the existing CANN9.2.0 environment and compile for **Ascend910A**.
The fleet's device-label string is not reinterpreted as A2 hardware.

Relevant jobs are collected, isolated distributed groups stopped, all8 cards
live-probed idle and lease released at round completion. Local commits remain
pending remote publication while GitHub's read-only branch check times out; no
blind replay of an uncertain push is used. Goal remains active for stable whole
precision, RoPE/AdaLN fusion, encoder/decoder targets and a correctly matched
warmed official lossless benchmark.
