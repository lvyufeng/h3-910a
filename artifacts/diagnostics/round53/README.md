# Round 53 — task-correct native baseline and full-step timing

## The coordinator contract was checked, not guessed

The first suspicion was that ignored `all_reduce` returns explained poor generation
quality. Reading only vLLM's coordinator is insufficient: H3 creates **Omni's own
coordinator**. Its current HCCL path normalizes to contiguous storage and reduces
in-place for the actual contiguous embedding/linear outputs. Real eight-rank HCCL
checks confirm that **both old and return-value-fixed primitives match exact small
CPU references** on every card. Therefore no missing collective is established in
the previous complete video, and this is **not** an identified cause of the lettering.

The narrow patch now honors the return value anyway. Real HCCL with a noncontiguous
view produces a fresh contiguous output, and a return-only mock coordinator
reproduces the old primitive's wrong result on all eight logical ranks. The new
primitive is correct in both cases. Weight loaders, BF16 storage and row-projection
FP32 reduction remain unchanged. These are source-extracted primitive tests, not
an end-to-end reference check of all Qwen layers.

[CPU primitive checks](<cpu_primitives.json>), per-card receipts such as
[rank 0](<hccl_rank0.json>) and [rank 7](<hccl_rank7.json>),
[reproducer](<validate_primitives.py>) and
[portable patch](<../../design/patches/encoder_collective_return.patch>) are saved.

## The old FL2VA quality baseline used the wrong task-family artifact

The actual [fusion receipt](<source/fused_lora.json>) identifies the four-step
FlashGen LoRA's supported task as **`t2va`**, while the request supplied first/last
images (`fl2va`). The old externally fused checkpoint bypasses the normal native
LoRA task guard, so HTTP success did not establish that the model was appropriate
for that conditional task. This is a verified metadata/configuration mismatch,
not proof that it alone caused the generated lettering.

The original unfused FL2VA transformer is still present. A private alias links
its original weight/component directories and removes the previously locally
pinned four-interval schedule **only in the alias metadata**. Original files are
not overwritten, weights are not fused/copied, and the engine still requests
`--dtype half` as before. The HTTP form now explicitly uses the **top-level
`num_inference_steps=49`** field. Putting that value inside `extra_params` merely
changes `extra_args` in this server and would not reliably set the scheduler.
The server's actual handling was read before submission.

[Alias/weight-slice receipt](<native_model_receipt.json>) confirms native components
and unchanged original metadata hash. Small binary slices show original BF16
transformer blocks vs fused-FP16 blocks; they are not whole-weight equivalence
checks. A native weight/schedule run is a more appropriate FL2VA reference than
adding steps to the four-step student, but is not automatically an official
lossless-quality comparison.

## Complete-step timing: verify the helper, then explicitly drain in PERF mode

The previous `dstep` timers end before the callback and cancellation checkpoint.
The first replacement incorrectly assumed `synchronize=True` always drains queued
work. Reading the [actual helper](<source/cancellation.py#L98-L119>) disproves that:
**only locally requested cancellation drains**; successful steps deliberately stay
asynchronous. The earlier synthetic test that faked an unconditional drain did
not validate runtime synchronization. Its [receipt](<timer_test_assumed_sync.json>)
and [rejected patch](<timer_assumed_sync.patch>) are preserved but are not successes.
The second engine attempt was intentionally stopped before submitting a request,
so its [interruption receipt](<native49_e2e02/result.json>) is not a model/operator
failure or a usable step benchmark.

The final timer explicitly calls `current_omni_platform.synchronize()` **inside
PERF mode only**, after the same callback and cancellation checkpoint, then ends
the wall timer. The [NPU platform implementation](<source/npu_platform.py#L309-L311>)
really invokes `torch.npu.synchronize()`. PERF=0 retains the original asynchronous
path. PERF=1 now introduces a measured step-boundary drain and can reduce
cross-step overlap; reported results must identify that measurement protocol.
There is no new per-rank global barrier. Each step, the actual sum and the observed
after-first average are recorded without calling the average a proven steady
state. The old misleading `steady_total`/49-step projection is removed.

The corrected synthetic-clock test executes the extracted old/new loops with
successful-cancellation semantics: model 2 s + callback 3 s + explicit measurement
drain 5 s. Complete timers correctly report **10 s per step / 20 s total**. CPU
outputs and model/callback invocation order match the old loop; PERF-disabled
execution has no extra drain. **Nine dependency-free contract regressions pass**,
including executing the actual cancellation helper with uncancelled signals.

[Corrected timer test](<timer_test.json>), [test source](<test_complete_timer.py>),
[structural regressions](<test_contracts.py>) and
[final portable timer patch](<../../design/patches/denoise_complete_timer.patch>)
are saved. A host test initially failed because bundled Python has no Torch;
CPU/NPU checks use the existing remote Torch environment, not a package install.

## Checkpoint storage mismatch made the first native startup unusable

The first native attempt reaches DLO preflight but rejects direct checkpoint mmap:
checkpoint BF16 does not match the existing FP16 runtime. The ordinary eight-worker
loader materializes/casts large weights. Before the 600 s startup deadline, host
usage reaches **605 GiB RAM / 111 GiB swap**; no video request is submitted. The
manager's own startup timeout stops and cleans up its engine. Afterwards, host
usage returns to 8.5 GiB and swap to 46 MiB; all eight cards have no processes.
This was a loading/storage problem, not a measured DiT hardware ceiling.

[Failed native startup receipt](<native49_e2e01/result.json>) is retained. A small
attempt to stop the exact owned manager found no matching process because it had
already exited at the harness deadline; no blanket process kill was used.

The unfused checkpoint is now converted once into separate storage that matches
**exactly the cast the unchanged `--dtype half` runtime would have made**. Every
saved tensor is checked byte-value-wise against that online cast, with all FP32
parameters retained as FP32. No LoRA is fused, original files are not overwritten,
and no inference settings are changed by storage conversion. This lets the mmap
loader avoid large per-worker CPU copies/casts.

The [completed conversion receipt](<native_storage_conversion.json>) covers
**13 shards / 66,280,485,328 bytes**. A separate
[raw-bit verification](<native_storage_bitcheck.json>) checks **all 535 tensors**
against the existing online cast, including signed-zero bit patterns. The
[13-tiny-shard test receipt](<storage_test.json>) checks deliberate BF16 underflow,
FP32 preservation (including a value above FP16 range), and source immutability.

This conversion is **not bit-exact BF16 inference**: **1,119,754 elements** change
on BF16→FP16→BF16 roundtrip, including tiny underflow/subnormal rounding. The
comparison target is the existing FP16 runtime cast, not the original BF16 compute
graph. The [conversion script](<convert_native_storage.py>) and
[test](<test_native_storage.py>) document that distinction. Its initial CPU
invocation failed at import because the CANN library environment was absent; it
was retried with the existing CANN environment, not with an installation or global
precision change.

## Completed native 49-step FL2VA — measured, not projected

The final third attempt completes with task-compatible original, unfused weights
in verified FP16-runtime storage and **49 actual native intervals**. The form is
1344×768, `seconds=5`, fps=24, seed=1234 and the same deterministic first/last images
as round 52. All eight ranks realize **535 checkpoint_mmap tensor views** without
the ordinary-loader fallback. Startup is **305.05 s** (excluded from request wall).
The request completes in **1265.77 s** and produces a verified **124-frame,
5.1667 s MP4 with audio**. All 124 frames decode and the downloaded SHA256 matches.

| Measured complete request / explicit-drain step protocol | Result |
|---|---:|
| Encode + conditioning reporter | **332.59 s**, 8 ranks |
| Full denoise reporter | **791.34 s**, 8 ranks |
| Decode reporter | **135.27–139.35 s**, 5 rank records captured |
| First complete step, rank maximum | **62.404 s** |
| Second complete step, rank maximum | **20.805 s** |
| Steps 2–49, mean of per-step rank maxima | **15.0810 s** |
| Steps 3–49, mean of per-step rank maxima | **14.9592 s** |
| Steps 3–49, per-step rank-max range | **14.8276–15.0832 s** |
| Steps 3–49 below 15 s | **35 / 47** |
| Last complete step, rank maximum | **14.8843 s** |

This is materially better evidence than the old four-step enqueue sub-timers:
**392 actual complete-step records** end after explicit PERF-only NPU drains.
Nevertheless, the after-first average is slightly above 15 s and 12 late steps
are above 15 s, so a strict always-below-15 s target is **not met**. Per-step maxima
are a conservative proxy, not a globally barrier-synchronized start/end clock.
The protocol also changes cross-step overlap in PERF mode; it is not an untouched
asynchronous-server latency measurement. Encode is not isolated text-encoder time,
and decode is still far above 20 s. The decode difference from round 52 is not a
controlled decoder A/B and is not attributed to a specific optimization.

[Raw completed request](<native49_e2e03/result.json>),
[392-step-record summary](<native49_summary.json>),
[mmap activation evidence](<mmap_activation.json>),
[downloaded MP4 metadata](<native49_e2e03/downloaded_mp4_metadata.json>) and
[full-frame decode/checksum](<native49_e2e03/frame_summary.json>) are saved. The
[video is available locally](<native49_e2e03/video.mp4>). The
[managed benchmark](<e2e_native49_probe.py>) preserves both earlier failed attempts.
The [final explicit-sync deployment](<deployment_explicit_sync.json>) supersedes
the earlier conditional-sync timer receipt; original sources stay backed up.

### Quality still fails this sanity check

The [first frame](<native49_e2e03/frame_000.png>) and
[last frame](<native49_e2e03/frame_123.png>) retain the red-ball anchors. The
[middle frame](<native49_e2e03/frame_062.png>) still contains a large unprompted
logo and lettering. Thus correcting the T2VA-student task mismatch and using
native 49 steps is **not sufficient to fix the issue**. No sole cause is established.
This is runtime-path/performance evidence, **not preserved generation quality or
an official lossless pass**. The [review record](<quality_review.json>) prioritizes
actual CPU/HF vs TP8 encoder hidden states, followed by real attention Q/K/V
full-key FP32 reference checks. Synthetic small matrices/random amplitudes alone
cannot validate all real model activations.

### Official comparison scope is now pinned

The [official NPU recipe](https://github.com/vllm-project/vllm-omni/blob/8ff8b02c152a6217c169719d56a4d581782fd1af/recipes/MiniMaxAI/MiniMax-H3-NPU.md)
is saved at commit `8ff8b02c152a6217c169719d56a4d581782fd1af`, with matching Git blob
and SHA256. Its **A2 150.06 s / A3 114.19 s** rows are **warmed lossless T2VA**, not
FL2VA with first/last images. Their DiT totals are 139.90/105.87 s, per-step
2.85/2.16 s; dividing total E2E by 49 is not the official per-step metric. There is
no FL2VA benchmark row in that recipe. Equal 49 steps and resolution do not make
this cold FL2VA request directly comparable in task, precision, measurement mode
or validated quality. The official HTTP example asks 50 while its 0.28 benchmark
table reports 49 DiT forwards; our current
[scheduler](<source/time_request.py#L39-L73>) explicitly generates N+1 sigma
boundaries for N evaluations. The observed 49-forward count here is real, but
cross-version equality of the full sigma ladder has not been demonstrated.
No hardware-only latency ratio is asserted.

The same recipe confirms FlashGen's T2VA-only restriction, dense/fused lossless
configuration, and exact power-of-two **LaserAttention input pre-scaling 256** to
avoid FP16 workspace overflow on outlier activations. Our real-layer numerical
checks must assess that class of issue; this source note is not proof that an
overflow caused the current lettering. [Pinned source](<official_npu_recipe.md>)
and [scope receipt](<official_reference.json>) retain the precise baseline.

### Persistent model for the next round

The separate native storage now has persistent metadata and component links at
`/home/mseco/models/MiniMax-H3-native-fp16-audit53`. The
[publication receipt](<persistent_model.json>) checks that metadata bytes and all
physical component targets match the temporary alias used by the completed test.
The persistent path itself has not yet had a separate inference request; no
model bytes were changed when publishing it. Use this native path and explicit
49-step form for follow-up correctness work, **not the old T2VA-fused checkpoint**.

All owned engine processes are stopped, all eight cards are verified without
processes, and the fleet lease is released at round completion. The goal remains
active for full-model quality, strict step target margin, fused operators, encoder
and VAE decode improvements, and a properly task-matched official comparison.
