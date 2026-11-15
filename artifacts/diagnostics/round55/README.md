# Round 55 — isolate vision interpolation and 910A BF16 linear fallback

## Scope and sources

Use the same real processor tensors/original weights as round54: two 1×48×84 image
grids, 8064 patch rows, BF16 checkpoint storage and activation outputs. Production
attention retains the verified exact-prefix v3 fix. No ComfyUI path is used.

[Baseline source](<source/encoder.py>) is verified SHA256
`ac48b620a1246228342af777a037dffe5ff6a6f0d6370d0317b10eb72069d0a1`.
The isolated CPU/NPU tests do not change running production code. Compact receipts
and source are versioned; tensor blobs, full logs, frames and MP4 media remain local
or remote evidence rather than Git payloads. The final
[candidate](<candidate_encoder.py>) is a **default-off** opt-in; its installed source
SHA is `f04123afdeda7cceb9f98e84f3873644fd64d0c37e2e24d4490959a14b90ab32`.
All vision linear classes retain nn.Linear parameter names/loaders. Text layers,
attention formula, norms, patch Conv3d and DiT math are unchanged by this candidate.
Nine structural contracts additionally verify the limited change scope.

## Corrected CPU vision port is bitwise equal to HF on the real sample

The prior CPU native-vs-HF whole-vision error of 9.52% was not an unavoidable
port-level numerical budget. Replacing only early-BF16 positional interpolation
weights/products/four-add sequence with FP32 interpolation followed by BF16 cast
makes the **entire CPU tower merged output and all three DeepStack features
bit-exact to installed HF**. Five selected blocks **0,8,16,24,26**, teacher-forced
with the same actual CPU input/weights/positions, are also bit-exact.

This is full-tensor equality on the real two-image sample, not a universal model
correctness proof, and CPU wall time is not NPU performance. The complete CPU
reference including five extra HF block checks is **598.20 s**.

[CPU receipt](<cpu_reference.json>) and [reproducer](<vision_cpu_reference.py>) preserve
matched inputs/outputs as bounded remote block captures (~60.4 MB each). These
compare the actual BF16 CPU implementations, not an FP32 teacher substitute.

## Teacher-forced first NPU block identifies the slower/inaccurate path

Every isolated sub-operator consumes **the matching HF input**, so accumulated
upstream errors do not confound its comparison. At block0:

- LayerNorm RMS differences: **0.00288% / 0.00216%**.
- GELU difference: **0.000534%**.
- QKV / MLP fc1 / fc2 linear differences: **0.2223% / 0.1941% / 0.2617%**.
- Same QKV, NPU FP32 attention math vs CPU FP32 math: **0.0261%**.
- CPU explicit FP32 attention math vs CPU SDPA: **0.0691%**.
- Complete NPU block on identical HF block input: **0.3972%**.

The actual controlled sub-operator replay is **block0 only**. The five CPU checks
must not be represented as five NPU teacher-forced block comparisons.
[NPU receipt](<npu_matched_reference_0.json>) and
[matched-input reproducer](<vision_npu_matched_reference.py>) state this distinction.
The explicit CPU math divides scores by sqrt(D), while the NPU helper multiplies
by its reciprocal; their floating-point ordering can also differ. These numerical
math-path comparisons are not pure hardware-attribution measurements.

## Actual linear compute A/B: FP32 is faster and closer to HF than native BF16

All inputs/weights remain derived from the same BF16 tensors. Casts are included
in each timed variant, no transformed-weight cache is used, and device-global
precision flags are unchanged (`npu.matmul.allow_hf32=false`, `npu.conv.allow_hf32=true`).

| Actual block0 operator | Native BF16 median | FP16 compute→BF16 | FP32 compute→BF16 | Native/FP32 RMS vs HF |
|---|---:|---:|---:|---:|
| QKV: 8064×1152, weights3456×1152 | 402.71 ms | 14.83 ms | **10.64 ms** | 0.2223%→**0.00233%** |
| fc1: weights4304×1152 | 499.56 ms | 17.61 ms | **12.50 ms** | 0.1941%→**0.00186%** |
| fc2: weights1152×4304 | 416.71 ms | 18.89 ms | **9.28 ms** | 0.2617%→**0.00297%** |

Moving only the BF16 bias addition into FP32 does **not** fix the slow/error path.
Explicit FP32 linear with/without fused bias has effectively the same improved
error. This evidence identifies an actionable **compute-path** issue, not a
universal 910A compute ceiling or a generic claim that FP32 always beats FP16.

[Raw linear receipts](<linear_precision_probe.json>) and
[reproducer](<linear_precision_probe.py>) include CPU BF16 and CPU FP32-rounded
reference comparisons. Small native CPU BF16-vs-FP32-rounded differences are
retained instead of treating the two CPU implementations as bitwise identical.

## Full real NPU tower A/B

The candidate uses FP32 compute for the six vision linear construction sites plus
FP32 interpolation, then returns to BF16. All **351 parameter tensors** and their
names/storage are equal; actual outputs remain BF16. Tests run the full tower in
interleaved and reversed order, two warmed calls each, with explicit local drains.

| Complete two-image tower | Baseline | Candidate |
|---|---:|---:|
| First observed cold call | 62.004 s | 17.364 s |
| Warm calls | 46.401 / 46.419 s | 4.814 / 4.850 s |
| Warm mean | **46.410 s** | **4.832 s** |
| Full merged RMS vs HF | **12.330%** | **5.661%** |
| DeepStack RMS vs HF | 2.456 / 4.921 / 7.732% | **1.136 / 2.211 / 3.769%** |

This is a controlled **9.605×** full vision-tower speedup and a measurable numerical
improvement. Residual whole-vision error is still substantial: it is **not lossless**,
not the whole text encoder reaching2s, and not a whole-model/video quality pass.
Only two warm observations per variant exist, not a broad variance/stability study.

[Full A/B](<full_tower_ab.json>) and [reproducer](<full_tower_ab.py>) use the
[frozen unconditional candidate](<candidate_encoder_unconditional_fp32.py>) for
reproducibility. The [exact executed source](<full_tower_ab_executed.py>) is retained
separately; the later reproducer changes only its candidate source filename.

An exploratory patch-projection variant in the raw A/B omitted HF's BF16 pixel
cast and is **not a like-for-like math comparison**. Its worse9.456% whole-tower
error is retained, not blamed on FP32 itself. A separate
[correct-input-cast repeat](<projection_cast_validation.json>) reduces patch RMS to
0.000122% but whole tower only5.584%, vs5.661% without it. The first call23.145s is
not a warmed A/B. **No projection change is deployed.**

## Default-off compatibility and quality request

The deployed variant is gated by `VLLM_OMNI_910A_VISION_FP32=1`, only for NPU BF16
vision compute. With the switch absent, the NPU linear/interpolation results are
bit-exact to the original path. CPU FP16/BF16/FP32 results and parameter keys remain
unchanged in eight formula/non-NPU regression cases. Opt-in full tower is **4.839s**
warm, the same5.661% RMS as the frozen measured candidate. No global HF32 policy is
modified; no FP32 weight caches are created.

[Opt-in validation](<optin_validation.json>),
[formula/CPU regressions](<linear_regressions_optin.json>),
[structural tests](<test_vision_audit_contracts.py>) and
[deployment/backup receipt](<deployment.json>) are provided. The final guard also
restricts interpolation to BF16 stored embeddings, keeping unmeasured non-BF16
NPU paths unchanged. [Portable diff](<../../design/patches/encoder_vision_fp32_optin.patch>)
reproduces the installed final source exactly.

## Completed native49-step request — runtime pass, quality still unresolved

The same [native49-step FL2VA form](<e2e_vision_precision_probe.py>) completes with
opt-in enabled, no capture environment, identical prompt/images/seed/weights, and
existing exact-prefix attention. Explicit PERF-only NPU drains remain enabled.
Independent CPU reference work overlaps some denoise steps, and cache-state
variation remains, so this is **not a controlled E2E speed A/B**.

| Actual complete request | Observation |
|---|---:|
| Request wall, startup excluded | **1225.764 s** |
| Encode + conditioning | **296.97 s** |
| Denoise | **786.86 s** |
| Decode, 4 rank records captured | **136.50–141.80 s** |
| First/second complete rank-max step | **62.623 / 21.126 s** |
| Steps2–49 mean rank maximum | **14.9902 s** |
| Steps3–49 mean rank maximum | **14.8597 s** |
| Steps3–49 rank-max range | **14.7690–15.1255 s**, **45/47 below15 s** |

The encode/conditioning stage is lower than the previous330 s observation, but
its components are not separately instrumented; **do not call296.97 s isolated
text time** or attribute all33.03 s to the operator change. The isolated full
vision-tower9.605× A/B is the controlled speed evidence. The step window does not
establish a strict15 s guarantee; the vision change is not a denoiser speed A/B.
Encoder2 s and decoder20 s targets are still unmet.

All **124 frames / 5.1667 s at1344×768 with audio** decode and checksum is verified.
Selected [first](<e2e01/frame_000.png>), [middle](<e2e01/frame_062.png>) and
[last](<e2e01/frame_123.png>) frames are reviewed. Unprompted logo/lettering remains
at mid-frame, so reduced operator error and completed runtime are **not full-model
quality/lossless passes**. No activations are captured from this opt-in request;
the separate language comparisons below use the **old round54 captures**.

[Raw request](<e2e01/result.json>), [complete-step summary](<e2e_summary.json>),
[frame/checksum receipt](<e2e01/frame_summary.json>),
[manual quality record](<quality_review.json>) and [actual video](<e2e01/video.mp4>)
are retained. Stage/rank-record truncation is explicit rather than replaced with
made-up all-rank timings.

## Complete fifty-layer higher-precision CPU text oracle exposes late outliers

A faster HF-based CPU reference computes linear layers in **FP32 followed by BF16
output**, while keeping original BF16 parameters and other HF operations. It is
**not the unchanged BF16 HF implementation**. It shares round54's native visual/
DeepStack inputs to isolate language execution, not independently validate all
multimodal features. Comparison hooks are pre-DeepStack; no final LM norm is used
to compare H3's intentionally unnormalized layer50.

Three layers complete in **17.468 s**; all50 complete in **279.281 s**, with551
parameter tensors and350 adapted linear modules. These are CPU reference costs,
not deployment text timings. The previous unchanged BF16 CPU50-layer attempt
still has only partial results after its2400 s timeout; it has not been retroactively
turned into a pass by this different reference.

Against round54's actual native NPU layer states:

- Layers0/1/2 full RMS differences: **0.1547 / 0.2350 / 0.2617%**.
- Layer9: **0.2206%**.
- Layer24: **0.4605%**, visual positions1.0801%.
- Layer49 (the50th retained layer): **16.5580% overall**, **17.9988% visual tokens**,
  **0.1196% text tokens**, maxabs difference **6464**. Both sides are finite.

The last figure is **not a language correctness pass**. There is no captured native
input immediately before layer49 and no FP32-teacher hidden-state capture for all
late layers, so it does not isolate the discrepancy to that block alone. Its
large visual-token outliers are a concrete next diagnostic target, not proof of
a single implementation or video-quality cause.

[Full50-layer reference](<hf_text_fp32_layers50.json>),
[three-layer pilot](<hf_text_fp32_layers3.json>),
[reproducer](<reference_hf_text_fp32.py>) and
[actual native outlier statistics](<native_text_outlier_stats.json>) are preserved.
Native layer49/final RMS13.454 and absmax20736 are finite. Many largest native values
are in channel731; layer24 has absmax15360. This confirms real outliers, **not FP16
overflow of the captured activations**. The cosine/sine references also have
nonzero RMS differences0.1500%/0.3644%; they are not advertised as bit-exact text RoPE.

## Verified text RMSNorm rounding contract for subsequent fused-kernel work

Reading the actual [common fallback](<source/norm.py>) reveals a model-specific
rounding difference: it multiplies FP32 normalized values by FP32 gamma before
one BF16 cast; installed HF Qwen3-VL first casts normalized values to BF16, then
multiplies by BF16 gamma. These are different contracts even if both compute the
variance in FP32. The common NPU path defaults to this native fallback when
`VLLM_OMNI_910A_NO_FUSED_NORM` is unset/default1.

On **actual layer0 input** with layer0 input-norm gamma and **actual layer24 output**
(the next layer25 input) with layer25 gamma, CPU formula isolation measures
**0.2813% / 0.2774% RMS** between the native fallback and HF. Rounding the normalized
value to BF16 before gamma produces **bit-exact HF outputs** in both full-tensor
cases. This is a verified model-specific operator contract issue, **not proof it
alone causes16.56% late-layer or video errors**. NPU/fused implementation is not yet
validated; production normalization is **unchanged**.

[Norm formula receipt](<text_rms_rounding_contract.json>) and
[reproducer](<text_rms_rounding_contract.py>) give the exact points a future
model-specific RMSNorm/AddRMSNorm kernel must preserve. Do not blindly change the
shared DiT norm contract or call a one-cast FP32 norm lossless for the BF16 Qwen port.

## Final state and next concrete work

The installed encoder retains the verified **default-off** vision precision mode
and exclusive pre-patch backup. The wrapper still matches exact-prefix v3.
[Final source/runtime verification](<final_source_verification.json>) records the
exact encoder/attention/norm hashes and torch2.10.0+cpu, torch_npu2.10.0.post7,
transformers5.14.1; the existing CANN9.2.0 environment is used. All
relevant jobs are completed/collected and owned server groups stopped; all8 NPUs
are live-probed idle and the fleet lease released at round completion. An initial
CPU-only Torch inspection missed the CANN environment and could not autoload
libhccl; sourcing the existing environment fixes it, without installing packages.

The objective remains active. Next tests should capture late native text inputs
and teacher-force the last layer while separating norm rounding, RoPE tables,
linear precision and TP reduction; preserve each candidate as opt-in until verified.
The matched NPU block script now has an optional candidate mode for future probes;
only baseline **block0** is executed in this round. Remaining whole-vision error,
custom fused kernels, text2 s/decoder20 s targets and a properly task-matched
official lossless quality/performance comparison are still unresolved.
