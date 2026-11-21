# R127 actual result — formal cold/warm/reload lifecycle

Source: [complete lifecycle summary](lifecycle_summary127c.json),
[actual three-request receipt](executed_lifecycle127c_compressed/engine_lifecycle127c/lifecycle_result127.json),
[raw worker audits](executed_lifecycle127c_compressed/engine_lifecycle127c/worker_runtime_pid1947666127.jsonl),
[native repeat media](native_repeat_media127.json).
This is a completed **observation**, not an optimization promotion or quality acceptance.

## What actually completed

- One `Omni`, the same eight worker PIDs and the same pipeline/transformer objects
  executed all three requests, each original50 blocks×49 steps.
- Every rank has step0..48 and cumulative indices0..146; **182,280 strict Half checks** pass.
- The transformer construction proves actual TP2 parameter bytes33,168,334,080/rank;
  the current offloaded placeholder sizes are not used as logical model sizes.
- Live FP16 encoder runs freshly each request; original FP32 VAE parameters and
  FP32 front/time/final/norm/residual/range contracts stay unchanged.
- TP2+USP4 / encoderTP8 stays enabled. No R126 operand cache, approximation,
  quantization, graph switch, stream policy or NUMA binding was applied.
- Three124-frame1344×768/24fps videos and Float32/32000Hz native audio were generated
  and fingerprinted on-host. No weights, latents or activation archives were transferred here.

## Measured costs — do not substitute scopes

One-time initialization is **347.514s** in this successful run.
Driver generation wall is **1339.217 / 1009.406 / 1008.572s** for cold/warm/native reload.
Native worker request rank-max is **1339.747 / 1007.953 / 1007.146s**; this includes
observational metadata/audit work. It is not pure kernel time.

Actual whole-forward hook medians (steps1..48, each step's maximum across ranks)
are **19.897 / 19.911 / 19.901s**. Cold first step is **93.945s**, while later
request first steps are **19.936 / 19.918s**. The prior17.556s isolated replay
or19.372s candidate-integratedR126 timer is not this original-safety/profile scope.
Warming removed the large first-step startup, not the steady-state attention/GEMM cost.

Formal pipeline `encode_prompt` host boundary is **64.092 / 3.191 / 3.146s**;
request preparation is **122.243 / 5.083 / 4.999s**. These include stage loading,
encoder work and existing hidden-state audits, not pure loaded encoder compute.

Video `decode_latent` host submission spans are **75.886 / 21.350 / 21.365s**;
consumer-stream event spans are **76.768 / 20.885 / 20.866s**.
Audio host submission is **0.491 / 0.205 / 0.220s**, but actual cold consumer-stream
interval is **63.434s**, then **0.252 / 0.219s**. Calling cold audio0.491s would
ignore its asynchronous completion/first-kernel work.
Full native video+audio decode stage is **142.268 / 23.170 / 23.135s**.
Neither warm video-only nor the completewarmVAE stage meets the user's<20s goal
under a strict stage boundary. Nested stage/event times must not be added.

## Residency and streams — direct findings for Issue#4

- Every request records **40 actual `_load_once` calls and40 offloads** across8 ranks.
  On each rank: text non-block state1, videoencoder1, videodecoder2, audioVAE1.
  Thus both later requests **still reload weights**; compiler state is warm,
  but the whole model/component set is not resident between requests.
- All127 layerwise hooks (50DiT+27vision+50text) are observed on each rank.
  Every observed copy handle equals that executing thread's current handle;
  **no independent copy stream is active**. Total actual prefetch calls are20,232/request
  (2,529/rank), including encoder hooks; do not call all of them DiT-only transfers.
- The legacy factory's shared-stream behavior is a safety shim, not evidence that
  prefetch overlaps device compute. Independent stream work in Issue#4 must respect
  allocator registration/creation thread and consumer/ready fences; a global env toggle
  is not a validated optimization.
- All sampled stager masters stay pinned and retain the same pointer/dtype/size
  metadata during each load/offload transition. This is **not** a fullbytehash at every
  transition; prior exact pool proofs remain separately scoped.
- Warm videodecoder reload consumer-stream interval is about **0.785 / 0.770s**;
  audio reload about **0.077 / 0.078s**. This is not enough to explain the~19.9s
  steady whole-step. Improving persistent process/caches benefits startup, while FA/GEMM
  remain the main line for the<15s/step objective.
- Runtime thread affinity covers all192 CPUs: no measured binding policy. NUMA VMA
  samples are captured but exact per-tensor physical pages and NUMA causal benefit remainUnknown.
- Peak allocated across ranks is about **18.34GiB**, peak reserved about **19.45GiB**.
  TP2 entire weights alone need30.89GiB/rank, so simply pinning the entire DiT on
  a32GiB device leaves insufficient realistic activation budget. No global residency flip.
- Exact compiler-kernel enumeration and per-kernelformat cost remainUnknown in this
  low-interference observation; cold/warm deltas alone are not causal compile proofs.

## Numeric status

Unchanged live requests are **not bit-identical**. Cold→warm encoded overallPSNR is
53.995dB (minimum46.188dB), warm→reload53.986dB (minimum48.959dB).
Native Float audio is not bit-identical either: relativeRMS error1.049/1.050,
absoluteRMS~1.209e-5, correlation~0.444. These are native self-repeat findings,
not an authorization to accept candidate error or ignore low-volume audio.
No quality policy changed and no alternative model/cache was adopted.

## Failures and cleanup

- [127a](failed_lifecycle127a_summary.json): cold model completed8×49/60,760checks,
  driver read rank audits before all flushed. Fixed only audit completion with atomic
  markers after JSONLclose and event-driven inotify waiting outside generation timing.
- [127b](failed_lifecycle127b_summary.json): nativeCPU Halfpacking SIGSEGV before requests,
  rank5−11. [Issue#14](https://github.com/lvyufeng/h3-910a/issues/14) preserves this unresolved
  reliability risk. The successful127c diagnostic run does **not** prove its root cause fixed.
- [Cleanup](cleanup127c.json): all private processes exited, auth key removed, original
  source/noise/private launched sourceSHA matches; own ts217lease released.
- The first slow read-only evidence transfer was stopped and resumed with compression
  into a fresh destination. It did not restart inference or move model/activation files.

**Issue#3 observation acceptance is met** with the explicitly statedUnknown items.
Issue#4 can proceed with worker-local independent copy-stream safety, memory and exact
fence proofs; FA/GEMM and VAEtile batching stay separate tasks and still require their own gates.
