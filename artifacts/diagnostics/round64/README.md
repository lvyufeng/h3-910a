# Round64 — opt-in instance-local VAE finite-select and actual service-stage gate

## Measured result and rejected acceptance

On actual8×910A FL2VA HTTP requests, complete warm **service decode** (video+audio+
residency transitions) improves from **22.84865s to13.81159s,1.654×**. Candidate
rank-max times are13.87823/13.77996/13.81159s, all under20s. This is not the previous
resident11.59s result and not a cold151s→13s controlled comparison.

The strict whole-audio/video identity gate **fails and remains rejected**. Its
same input latents stay immutable and all **72 complete124-frame video outputs**
match raw bytes across baseline/candidate/all ranks, but the unchanged native
audio path is itself not repeat-bit-exact (five different hashes in five baseline
calls on every rank). The ninth HTTP request therefore returns500 after the
explicit assertion; the eight preceding normal requests completed successfully.
Timing evidence and video identity are retained separately from the failed whole
A/V acceptance. No official-lossless or full-goal completion is claimed.

[Verified summary](<summary.json>), [normal HTTP receipts](<http/result.json>),
[rank0 same-input failure](<timing/paired_rank0.json>) and
[bounded failure log](<http/failure_excerpt.txt>) preserve this distinction.

## What this round changes

[Round63](<../round63/README.md>) measured11.59s for resident video VAE alone.
That excluded actual pipeline residency transitions and the audio decoder. This
round adds a default-off service implementation and tests the complete native
service decode boundary, not a multiplication from resident tiles.

- [Opt-in integration](<../../design/patches/h3_910a_vae.py>) activates only for
  `VLLM_OMNI_910A_VAE_FINITE_SELECT=1`. Query chunking has its own VAE-only knob
  `VLLM_OMNI_910A_VAE_QUERY_CHUNK=256`; global SDPA chunking stays128. The
  [minimal adapter patch](<../../design/patches/h3_910a_vae_install.patch>) excludes
  all service timing code; copy the two dedicated service/kernel modules beside
  the adapter before applying it. This reuses this project's existing
  `fa910a` launcher/runtime-core query, `/tmp/libfa910a.so`, CANN/TBE environment
  and current-stream contract; it is not an independent wheel for arbitrary NPU
  versions. Unknown SDPA return contracts or unsupported SoC suffixes fail closed.
- It clones the native attention/flash/SDPA function namespaces **per decoder
  instance**. It never replaces global `F.scaled_dot_product_attention`, remote
  module globals, text encoder attention, DiT attention or model weights.
- The [native-half TBE kernel](<../../design/patches/h3_910a_vae_kernel.py>) is the
  same finite-select algorithm checked in Round63; service kernels use a distinct
  source-hashed JIT cache. NaN/Inf are selected to positive zero. This differs
  from the currently configured native NPU behavior on exceptional half patterns;
  arbitrary-nonfinite native identity and original-BF16 equivalence are **not**
  claimed.
- A [hash-pinned deployment](<deployment.json>) installs two temporary adapter/
  timing entries and four dedicated modules. Originals have non-overwritten
  backups. The [deployer](<deploy_service64.py>) refuses unknown bytes; no global
  patches, model/checkpoint overwrite or package installation is performed.

The [actual tile smoke](<smoke.json>) runs baseline/candidate/candidate/baseline
against the real36-layer tile. All four5,505,024-element outputs match the actual
capture bits. Global SDPA stays identical and only the two candidate calls enter
the new72 attention calls. Reported runtime SoC is `Ascend910B` **without B1/B2/B3
suffix**, the genuine910A-generation configuration verified in the previous
hardware audit, not an A2/A3 model.

## Exact service timer versus old log lines

The existing unsynchronized `H3PERFP decode` line cannot certify this A/B:
it starts after an asynchronous denoise checkpoint and does not explicitly drain
all decode work before stopping. It is retained only as older diagnostic output.

[New timing entry](<h3_service_audit64.py>) surrounds the unchanged native
[decode method](<source/pipeline_minimax_h3.py#L2099-L2125>) with a start barrier+
NPU drain outside the timer and an end drain inside. Current-stream events and
host wall time are both recorded on all eight ranks. Each request's critical time
is `max(rank wall time)`; the warm result is a median of three such requests,
**not** `max(median(rank))`.

That method includes:

1. idle stage-cache release;
2. video decoder component load, actual native tiled/temporal decode and offload;
3. native output crop;
4. audio component load, decode, offload and the existing audio CPU stage-output.

It excludes later video D2H/frame preparation, postprocessing and MP4 mux/download.
The end-to-end HTTP wall time is retained separately. No resident decoder bypass
or altered model loader is substituted for this service method.

## Workload and independent acceptance checks

[HTTP driver](<service_http64.py>) starts a managed, loopback-only engine with
original unfused native weights, FL2VA, first/last images,1344×768,5s/24fps,
seed1234,USP8/ring1/textTP8/tile8/DLO. Aligned video has124 frames. **One denoise
interval** is intentional for this decoder stage gate; it is not49-step video
quality or the official warmed T2VA lossless benchmark.

Normal request sequence is baseline warmup, three baseline measurements,
candidate warmup, three candidate measurements. Capture/audit hooks are disabled.
Each downloaded MP4's full hash and video-track geometry/frame count are retained.
There is no claim that the same seed alone guarantees unchanged denoiser latents.
In fact the already completed baseline requests have different MP4 hashes from
one another, as do candidate requests. That means cross-request hashes cannot
establish decoder identity, and the normal E2E difference is not cleanly attributable
only to the decoder. It is not used as a quality or lossless-generation proof.

The final `paired` HTTP request checks stronger identity: on its **same actual
video and audio latent tensors**, it alternates the complete original service
stage baseline/candidate nine times (first baseline+candidate are warmup); its
intended fresh candidate return is **not reached**, because the all-A/V assertion
correctly rejects the unequal native audio outputs. Both latent hashes are checked
before/after; full output hashes, shapes, dtypes and bytes are retained on all
ranks/calls. CPU downloads/hashes are **outside** stage timers. This request does
extra work, fails overall, and is excluded from normal request/E2E latency comparisons.

[Aggregator](<summarize.py>) independently reconciles every rank/mode/call and
rejects failed, missing, source-mismatched, inactive-candidate or mutated-input
proofs. It does not accept `completed=true` as sufficient numerical correctness.
The retained HTTP failure is classified explicitly, **not** converted into a
success: `timing_only_target_met=true`, `video_only_identity_accepted=true`, but
`stage_accepted_under20s=false` and `official_lossless_goal_complete=false`.

The paired full-stage warm medians are22.75627s baseline and13.73018s candidate;
final baseline is22.76205s. Its video reference is383,975,424 uint8 bytes per call,
and audio is real nonempty stereoFP32 `[1,2,165600]`, not a placeholder. Failure
is not inferred from MP4 container metadata: it is in the full pre-codec audio
sample hashes. Hash differences alone do not measure error magnitude.

## Native audio control, not a hidden change of reference

A separate [audio-only probe](<audio_repeat64.py>) uses actual model weights and a
fixed **synthetic** FP32 `[2,32,207]` latent matching the HTTP geometry. It does not
use the lost actual denoised HTTP latent and cannot measure that request's audio
error. Video finite-select is disabled and production audio math is unmodified.

[Raw control receipts](<audio_repeat01.json>) show five distinct hashes in each
of resident/native, staged/native, resident/deterministic and staged/deterministic.
Input and immutable CPU weight-master hashes remain unchanged; CPU/NPU RNG states
also stay unchanged. `torch.use_deterministic_algorithms(True)` does not stabilize
these tested audio outputs. Warm decode itself is only0.213–0.223s; this is not the
large performance bottleneck removed by the video finite-select kernel. The
specific nondeterministic native operator has **not** been localized.

The observed **synthetic-control** audio differences versus the first call in
each group reach max absolute0.000232593 and relativeRMS0.00032101 (~0.0321%);
all samples remain finite. Those figures do **not** quantify the lost actual
HTTP audio latent's error or establish perceptual quality. They are not blamed
on the new video kernel, nor silently waived as a full A/V quality acceptance.
An operator-level native audio repeat audit and an explicitly justified numerical/
quality acceptance rule remain follow-up work.

## Restoration and reproduction

The [restoration receipt](<restoration.json>) restores exact original adapter and
pipeline bytes; [final verification](<final_verification.json>) checks23 production/
model/dedicated-module sources and12 executed diagnostic sources. The four new
modules remain present but unused by the restored baseline entry points; the
opt-in portable patch is saved for subsequent controlled deployment, not enabled
as the production default. No service is intentionally left running. The
[final fleet probe/release](<cleanup_receipt.json>) records all eight cards at0%
utilization with zero accelerator processes and the owned lease released.

Thirteen service reporting/isolation/timing/failure regressions and the preceding
22 kernel plus6 block regressions [pass](<test_receipt.json>) (41 total); these
portable checks do not replace the retained real-NPU timing and output receipts.

## Reproduction and remaining objective

```bash
python3 -m unittest discover -s artifacts/diagnostics/round64 -p 'test_*.py' -v
python3 artifacts/diagnostics/round64/summarize.py
```

Remote script paths/run IDs are exclusive; never replay the active engine manager
or overwrite an earlier result. A fresh owned910A lease is required for any NPU
follow-up. Normal stage-control reads happen once per request; changing a control
file does not affect an in-flight call. The single managed engine serves one
request at a time, and only its own verified process group is stopped.

The larger goal remains active: text≤2s, strict whole-step DiT<15s with enough
margin, complete49-step FL2VA quality, and a separate task/precision/warm-protocol
matched comparison against official A2/A3 still require evidence. A service decode
result, even under20s, does not complete those objectives.

A parallel read-only [text-stack audit](<text_next_gate.md>) pins the next
controlled candidate: exact FP32 shadow weights for non-K/V projections and one
shared FP32 conversion of the Q/K/V input, preserving current stable1024-output
K/V GEMM geometry, Norm rounding and communication order. It is a proposed A/B,
not an implemented text≤2s result. Current stable4.1s numbers are resident
experimental text-only, not the production stage used by this HTTP test.
