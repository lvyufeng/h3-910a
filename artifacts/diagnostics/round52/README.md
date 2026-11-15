# Round 52 — integrated bounded tile padding with stable added-key exclusion

## Measured real wrapper, not just pre-padded tensors

The round-51 prototype's larger tiles are now integrated with input copying,
buffer clearing/reuse, kernel selection, and output trimming. On the same valid
`B=1,H=7,N=41792,D=128` Q/K/V inputs:

| Complete wrapper | First order median | Reversed-order median |
|---|---:|---:|
| Old 64×64 tiles | 421.142 ms | 421.160 ms |
| New internal padding, 128×304 tiles | **154.401 ms** | **154.382 ms** |

Mean group-median speedup is **2.7278×**, a **63.34% latency reduction**. Unlike the
prior prototype, these new timings **include internal padding copies, full buffer
clearing and output cropping/contiguity**. They remain isolated, synchronized
single-device wrapper measurements, not DiT-step or video latency claims.

The sequence is expanded only for inefficient divisors at head_dim=128, and only
when added rows are at most 10% of the input length. The actual case expands to
43776 (=18×lcm(128,304)). Efficient shapes such as N=63232 retain their old kernels.
`VLLM_OMNI_910A_FA_PAD_TILES=0` selects the old exact-divisor path for rollback.

Evidence: [final raw validation](<validation_result05.json>),
[compact summary](<wrapper_summary.json>), [reproducer](<validate_wrapper.py>), and
[deployed wrapper source](<../../design/patches/fa910a.py>).

## Added-key exclusion is not the unstable prototype's LSE subtraction

A deliberately adversarial test makes every valid score **−45.25** and every
valid V row one. The correct attention result is one. The old extra-zero-key/LSE
subtraction prototype returns finite **zero** (absolute error 1), because its
small valid probability mass is lost before denominator subtraction.

The new compiled kernel excludes added keys instead:

1. Replace added-column scores with FP16's finite minimum before rowmax.
2. Explicitly zero those columns **after exp and before both row-sum and P@V**.

TIK rejects a literal −inf immediate, so using a finite sentinel alone would not
be enough: a second adversarial case has all valid scores equal to **−65504**.
Explicit post-exp masking removes the otherwise tied padded probability mass.
Both adversarial cases return exactly one with finite output, max error 0.

This stable exclusion applies to **new internally added columns**. The caller's
preexisting zero-padding `n_pad` still retains the old wrapper's LSE contract;
that contract assumes those original K/V rows are zero. It is not a new guarantee
for arbitrary masked/causal attention or for pathological old caller padding.
No key/head/block sparsity or attention approximation is introduced.

## Numerical scope and cache safeguards

Three amplitudes/original-padding cases (including 64 original padded keys) plus
length changes 41792→41920→41792 are checked. Whole valid-output pair relative
RMS difference from the old tiling is at most **0.01385%**. The sampled FP32
full-key reference relative RMS error for the new path is at most **0.03979%**;
outputs are finite. Reference sampling covers eight query rows of one head, not
every query/head of a real H3 block. Tilings are **not bit-exact** and these checks
are not a complete model-quality or official lossless evaluation.

Safety measures exercised:

- kernel cache identity includes runtime core count **and compiled key boundary**;
- masked artifacts cannot alias the unmasked round-51 prototype;
- cached metadata must identify the expected boundary and mask version;
- metadata is published before the object readiness marker;
- private build paths are resolved/verified before move/delete, including macOS
  temporary-directory aliases;
- padding buffers are reused but fully cleared each call, before prefix copying;
- this avoids the installed zero_ kernel's nonzero-storage-offset warning;
- original output shape/dtype and efficient-shape/explicit-opt-out behavior stay
  unchanged;
- a lazy scaling buffer fixes the old scale=1→scaled-call allocation bug.

**16 host-only regressions pass**, including an uncached fake-CANN build test as
well as cached metadata checks. Real NPU validation covers tail clearing, changes
in compiled boundaries, scale=1 followed by normal scaling, and adversarial score
ranges. Reuse assumes ordered single-worker/current-stream execution as in the
old wrapper; no new multi-stream/reentrancy guarantee is made.

[Test sources](<test_padding_plan.py>),
[core-count regressions](<test_core_budget.py>), and
[final deployment receipt](<deployment.json>) are included. Earlier failed probes
are preserved: deferred Bc initialization, TIK's unsupported infinity immediate,
and a local `import json` initialization error. They were fixed and the final path
was revalidated rather than counted as successes. The subsequent host fresh-build
test also caught and resolved legitimate macOS temp-directory symlink handling.

## Deployment and real FL2VA run

The checked deployed source SHA256 is
`c98b9064b4a01d0fa7166b292d8a7e6abf0793918d4479a8025b9715532da4a8`.
The original wrapper is preserved in an exclusive backup. The canonical repository
patch is byte-identical to the deployed candidate; weights, precision, schedule,
vision compatibility and VAE padding patches are otherwise unchanged.

## Complete real FL2VA validation — first successful verified video

The same actual first/last-image FL2VA form completes on a fresh 8-rank engine:
1344×768, `seconds=5`, fps=24, seed=1234. The real MP4 has **124 frames at 24 fps,
5.1667 s**, as required by the native 17n+5 alignment, plus a 32 kHz audio track.
Startup is 245.05 s (excluded from request wall). **Request wall is 975.59 s**.
All eight ranks confirm source_N=41792, kernel_N=43776, original padding=23 and
30 cores / 128×304 tiles.

| Stage reporter | This complete request |
|---|---:|
| Encode + conditioning preparation | **333.59 s** |
| Full denoise | **166.99 s** |
| Decode | **465.52–466.88 s** |

These stage records do not isolate text encoding from image/media preparation.
Neither encoder nor VAE decode targets are achieved. The different cold/cache
state across prior requests prevents attributing every end-to-end reduction to
this wrapper; the **2.7278× wrapper A/B** remains the controlled gain evidence.

### Step timing correction

The last two sub-timer sums are **14.674 and 14.736 s**, with model sub-timers
12.904 and 12.961 s. However, reading the actual
[step reporter source](<source/denoise_loop.py#L485-L509>) shows the timers stop
**before** callbacks and `check_request_cancellation(synchronize=True)`. Therefore
they are not complete synchronized step latency and do **not** prove the 15 s
whole-step target. The old `steady_avg=29.99` averages steps 2–4 and still includes
step-2 cold effects (~60.56 s by sub-timers). Its `steady_total` is average×all
steps, not an actual measured total; its 49-step projection is not an official
lossless benchmark. A follow-up should place the complete-step timer after the
existing synchronization and validate on more steps without changing the work.

This run uses the **four-step distilled checkpoint**, not the original official
49-step lossless schedule. Completion proves the real FL2VA path works, not that
quality or official-comparison objectives have been met. The goal remains active.

[Raw completed request](<e2e01/result.json>),
[complete-stage/step qualification](<e2e_summary.json>),
[downloaded MP4 metadata](<e2e01/downloaded_mp4_metadata.json>), and
[content sanity/decode record](<e2e01/frame_summary.json>) are saved. The complete
MP4 is [available locally](<e2e01/video.mp4>); its SHA256 matches the decoded remote
artifact and all 124 frames decode. **Visual inspection is not a quality pass:**
[the first frame](<e2e01/frame_000.png>) and [last frame](<e2e01/frame_123.png>) retain
the reference red-ball placement, while [the middle frame](<e2e01/frame_062.png>)
contains prominent unprompted lettering. This one example cannot identify whether
the cause is checkpoint/schedule behavior, conditioning/encoder correctness, or
another backend difference. A controlled reference comparison is required before
claiming preserved generation quality or an official lossless result.
An early parser attempt on
the still-running media transfer was correctly rejected as truncated; validation
was repeated only after transfer completion and exact byte-count/checksum checks.

The tested wrapper remains deployed with the original backed up. All managed
engine processes are stopped, all eight cards are verified without processes,
and the fleet lease is released at round completion.
