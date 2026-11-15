# Round 54 — real activations expose a nonzero-padding attention contract bug

## Real captures, not random input substitutes

A default-disabled rank-zero diagnostic helper captures the actual processor input,
embedding, vision outputs, language-model input/rotary states, selected text-layer
outputs and two DiT attention calls from the same native 49-step FL2VA request.
Captured calls are ordinals **0 and 49 among matching long-sequence wrapper calls**;
the second is not denoise step 49. Each includes all **seven rank-local heads** at
N=41792, D=128, and is bounded below 512 MiB per file.

The complete capture request produces a verified 124-frame/5.1667 s MP4, but capture
copies synchronize and this is **not a clean performance benchmark**. CPU reference
work also runs on the host during this audit. Tensor binaries/media/full logs are
retained locally or on the host, while compact receipts are versioned.

[Helper](<h3_910a_audit.py>) and [hook tests](<test_capture_hooks.py>) validate no
writes when disabled, detached write-once bounded tensors and unchanged module
outputs on the first and second invocation. **Five AST contract tests** additionally
verify that kernel build/core math, SDPA call arguments and text/vision forward math
are unchanged by capture sites. A first hook sketch returned a Boolean on a skipped
capture; it was corrected to always return None and checked before deployment.

The **base text-SDPA capture does not execute**: the actual NPU model patch replaces
that function with native GQA. Reading the
[effective runtime patch](<source/npu_minimax_h3.py>) detects this; no text-attention
capture is claimed. The actual text *layer* and complete vision outputs are captured,
which remain useful for independent HF comparisons. Future attention instrumentation
must be attached to that effective NPU function, not just the base port.

## Actual modulated padding is not zero

The old wrapper subtracts `n_pad` from the softmax denominator assuming each padded
key and value is zero. The metadata correctly marks **23 trailing invalid keys**,
but the real model applies QKV projections/normalization/modulation, so those rows
are no longer zero:

| Captured attention | Invalid-tail K max abs | Invalid-tail V max abs |
|---|---:|---:|
| matching call 0 | **7.6602** | **2.3457** |
| matching call 49 | **80.875** | **240.625** |

All captured inputs/outputs are finite. Sampled full-key FP32 score ranges are
−10.32…15.51 and −54.32…129.85; this is not evidence of an actual FP16 overflow in
these two samples. They do expose a definite **incorrect padding assumption**.
[Original actual-tensor FP32 reference](<attention_reference.json>) uses eight
selected valid query rows per head and **all 41769 valid keys**, checking both exact
FP32 scaling and matching the existing FP16-rounded scaled Q.

## Exact arbitrary-column prefix mask

The candidate now uses `source_N - n_pad` as the kernel's true valid-key prefix.
Both original invalid keys and added tile-padding keys are excluded before rowmax
and explicitly zeroed after exp, before row-sum and P@V. There is **no zero-key LSE
subtraction**. The last NZ block uses a repeated 128-bit vector mask for arbitrary
1–15-column tails, so the true valid length **41769** need not align to 16. Cached
kernel names include the valid boundary and a new **v3** mask version; old aligned
or unmasked artifacts cannot alias the corrected kernel.

Seven real NPU adversarial tail tests cover valid lengths **1,15,17,303,305,2417,2431**,
including block-crossings and nonzero invalid K/V. The uniform-one result is exact.
A last-valid-key impulse also verifies that masking does not wrongly discard valid
columns. **Five host prefix-mask/cache tests pass**, exercising all 15 possible
sub-column masks and old-version rejection.

### Same captured Q/K/V numerical improvement

| Real captured attention, FP16-scaled-Q/full-key FP32 reference | Old relative RMS | Correct prefix relative RMS |
|---|---:|---:|
| matching call 0 | 0.41736% | **0.06323%** |
| matching call 49 | 1.53803% | **0.20122%** |

Sampled max absolute errors change **0.06507→0.00894** and **8.42201→0.64761**.
The correction measurably reduces actual attention error; remaining FP16
score/probability errors are not a lossless proof or a sole-cause explanation of
unprompted lettering. The reference checks selected queries on one rank, not every
query/head/layer of the full model.

The same real late-call wrapper timing is **155.356→154.475 ms** (three groups of
three calls per variant). This limited fixed-order check finds no slowdown, but
is not a reversed-order/full-model speed ablation. Correctness is the purpose of
this patch.

[Raw NPU validation](<prefix_validation.json>),
[prefix candidate](<candidate_prefix_fa910a.py>),
[reproducer](<validate_prefix_npu.py>) and [host tests](<test_prefix_masks.py>) are
provided. The first host invocation lacked NumPy; it was rerun in the already
available bundled dependency environment, not after installing dependencies.

## Real encoder reference work — full vision discrepancy remains

The first three installed **HF CPU BF16 language layers**, loaded with original
weights and captured native visual/DeepStack input, compare to actual TP8 outputs
with full-tensor relative RMS **0.155%,0.233%,0.260%**. Embedding rows and RoPE tables
are also checked. This isolates the language port and does not validate the input
vision features. The separate attempted 50-layer CPU teacher hits its **2400 s**
timeout; the last reported comparison is layer 24 (relative RMS **0.458%**, visual
positions **1.074%**). Later unreported execution progress is unknown. **No complete
50-layer reference pass exists.** Its observed partial stdout checkpoints and
exit code are retained rather than presented as completed validation. Future
reference code now preserves progress and handles SIGTERM; that persistence change
was not retroactively executed by the timed-out process.

The independent installed HF visual tower consumes actual processor pixel/grid
tensors and original vision weights, not captured native visual embeddings. At
**8064 input patches**, two **1×48×84** grids, its full merged output differs from
the deployed NPU output by **12.330% relative RMS**. DeepStack differences are
**2.456%,4.921%,7.732%**. Multimodal position IDs are **bit-exact**.

Running the original native visual port on CPU separates port differences from
NPU execution: full merged output differs from CPU HF by **9.523%**, and from the
captured NPU output by **10.824%**. These are direct comparisons with different
reference denominators, **not additive error components**. Therefore one cannot
blame the entire 12.33% on the NPU backend or declare the port correct based on the
small standalone block tests.

One concrete port-level difference is interpolation precision. The old BF16
weights/products/four-add sequence differs from the installed HF position embedding
by **0.2886% RMS**. FP32 weights/products/sum, then conversion back to BF16, matches
the CPU HF table **bit-exactly**; visual RoPE tables already match exactly. This
isolates a numerical operation difference, not the whole quality cause.

An isolated **NPU-only interpolation candidate**, keeping the original CPU path
bit-exact, changes full merged-output error **12.330%→10.777%** and DeepStack errors
**2.456%→2.130%, 4.921%→4.193%, 7.732%→6.632%**. The residual discrepancy is still
large. Its first observed standalone forward is **64.998 s**, not steady text
encoding or a controlled speedup. The candidate is **not deployed**; full vision
layer/attention and precision comparisons remain necessary.

Evidence: [completed three-layer HF comparison](<hf_text_reference_layers3.json>),
[partial 50-layer attempt](<hf_text_reference_layers50_partial.json>),
[independent HF vision](<hf_vision_reference.json>),
[CPU native-port isolation](<cpu_native_vision_reference.json>) and
[NPU interpolation candidate](<npu_vision_interpolation_candidate.json>).
[Executed language reference source](<reference_hf_text_executed.py>) is kept
separate from the [progress-preserving follow-up](<reference_hf_text.py>).
[Vision reference](<reference_hf_vision.py>),
[CPU isolation reproducer](<reference_cpu_native_vision.py>) and
[NPU candidate reproducer](<validate_npu_vision_interpolation.py>) state precisely
what they compare. CPU reference wall time is not an NPU encoder benchmark.

## Deployed prefix fix and completed repeat generation

[Prefix deployment](<prefix_deployment.json>) restores encoder capture sites to
source SHA256 `ac48b620a1246228342af777a037dffe5ff6a6f0d6370d0317b10eb72069d0a1`,
and installs the tested wrapper SHA256
`937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1`.
The canonical [wrapper](<../../design/patches/fa910a.py>) is byte-identical to the
tested candidate; [portable diff](<../../design/patches/fa910a_exact_prefix.patch>)
is provided. Original sources and capture patches are backed up. The capture helper
is retained on disk but unreferenced by restored production sources. The vision
interpolation prototype is not installed and must not be confused with this fix.

The same [49-step native request](<e2e_prefix_probe.py>) completes with no capture
environment, unchanged weights/sigma settings, and explicit PERF-only device drains:

| Complete prefix-corrected request | Observation |
|---|---:|
| Request wall, startup excluded | **1255.82 s** |
| Encode + conditioning | **330.00 s** |
| Full denoise | **780.42 s** |
| Decode, 7 rank records captured | **138.34–143.36 s** |
| Complete first/second step, rank maxima | **62.863 / 20.747 s** |
| Complete steps 2–49, mean rank maximum | **14.8423 s** |
| Complete steps 3–49, mean rank maximum | **14.7167 s** |
| Complete steps 3–49, rank-max range | **14.6197–14.8396 s**, all **47/47 below 15 s** |

This is an actual observed warm-step window below 15 s for one completed request,
not proof of stable margin on all requests, cold steps, or a validated lossless
model. Per-rank maxima are not globally barrier-timed boundaries. Concurrent CPU
reference/cache-state variation prevents attributing all E2E timing differences
to the prefix patch; the controlled operator result remains the same-input A/B.
Encoder/decode targets are still far from achieved.

The MP4 has **124 frames / 5.1667 s at 1344×768 with audio**, all frames decode,
and checksum matches. But [the middle frame](<e2e02_prefix/frame_062.png>) still
contains unprompted logo/lettering. The [quality record](<quality_review_prefix.json>)
therefore does **not** count runtime success or reduced attention error as a
model-quality/lossless pass.

[Raw completed request](<e2e02_prefix/result.json>),
[complete-step summary](<e2e_prefix_summary.json>),
[frame/checksum evidence](<e2e02_prefix/frame_summary.json>) and
[video](<e2e02_prefix/video.mp4>) are provided. The complete diagnostic request is
also retained in [its receipt](<e2e01/result.json>), but its capture-copy overhead
and concurrent reference work are not a speed benchmark.

All relevant CPU/NPU jobs have completed or timed out and their outcomes are
recorded. Owned servers are stopped, all eight cards verified without processes,
and the fleet lease released at round completion. The goal remains active for
full-model correctness, fused kernels, encoder/decode targets and a task-matched
lossless comparison. Next priority is per-layer vision precision/reference
isolation and a complete encoder reference, not another physical-limit assertion.
