# R127 — GitHub Issue #3 formal engine lifecycle

Live task: [Issue #3](https://github.com/lvyufeng/h3-910a/issues/3) in
[Project #10](https://github.com/users/lvyufeng/projects/10).
This round observes the original accepted R124+FA mathematics. It does **not**
adopt R126 private candidates, disable TP, change stream policies/NUMA affinity,
claim a new precision baseline or mark any latency target achieved.

## Scope

The [driver](generate_lifecycle127.py) creates one `Omni` and submits three
fresh requests: cold, warm, and another **native-stage reload with warm compiler
cache**. Each request remains 50 original DiT blocks ×49 evaluations, with all
155 strict Half checks per step; total expected checks are 182,280 across 8 ranks.
FP16 matrix operands and the original FP32 front/time/final/VAE/norm/residual
exceptions stay unchanged. TP2+USP4, live encoderTP8, all eight cards are retained.

[Private integration](integration_hooks127.py) fixes only the old single-request
audit's lifetime: request-local step index and output names, plus independent
cumulative step/check counters. It never resets the cumulative safety evidence,
substitutes hidden states or changes the sampler. Encoder and final latent captures
stay on the host and are not transferred to the client.

## Measurement limitations

- [Runtime observer](observe_runtime127.py) records native load/_load_once/offload,
  actual current/copy handles in each runtime thread, pinned storage metadata,
  allocator-release decisions, memory and inclusive current-stream events.
- There is no per-module `synchronize`, no `torch.npu.Stream()` policy change,
  no affinity binding or new weight-pool allocation in the observer.
- Current-stream event intervals include waiting and host launch gaps. They are
  not pure SDMA/HBM activity and nested spans must not be added.
- The accepted safety hook performs complete prediction D2H for reliable
  finiteness checks on legacy NPU. The same whole-forward audit scope is retained;
  it is not pure device compute and is not the isolated replay timer.
- Pinned NUMA sampling uses `/proc` VMA statistics because `numactl` is absent.
  A VMA's page-node counts are not exact per-tensor physical-page placement.
- Native stagers may unload after every component stage. A hot request can still
  perform H2D weight reload; only actual `loaded` transitions substantiate that.
- Interface names return `Ascend910B`; the existing legacy910A kernel path and
  two4-card HCCS topology are separately evidenced. Do not infer silicon from the
  roster label or rewrite the queried device name.

## Sources and safety

- [Preflight](preflight127.json): actual runtime versions, source fingerprints,
  current device/topology, NUMA cpulists and parent-only stream probe.
- [Initial launch pins](launch_source_pins127.json) and
  [corrected launch pins](launch_source_pins127b.json): actual uploaded source SHA snapshots.
- [Launcher](launch_lifecycle127.py): fresh-root/receipt guards, immutable source
  checks, 32-byte0600 private IPC and original sources; refuses uncertain reruns.
- [Bootstrap](full_bootstrap127.py) and [startup](sitecustomize.py): install model
  hooks only in real DiffusionWorkers, not CANN compiler helper processes.
- [Start contract](issue3_start.md): resource and compute budget published in GitHub.
- [Summary verifier](summarize_lifecycle127.py): completed actual audit only, same
  worker and pipeline identities, 0..48 steps, 155 checks, state transitions.
- [Portable tests](test_contracts127.py): 14 tests including counter isolation,
  native math/TP, no new candidates, bootstrap security, rank-max aggregation,
  atomic audit-completion markers, matched session roots and unchanged native packing.

```bash
python3 artifacts/diagnostics/round127/test_contracts127.py
```

Weights are only the verified on-host checkpoint, with no client download/relay.
The worker/source snapshots are textual evidence, not a model archive.
Only own leased `ts-217` is used for this experiment; other sessions' machine leases
are not authorization for this one. Shared SDK/driver/model sources stay untouched.

## Execution status

The first run `engine_lifecycle127a`, PID1038756, actually completed the cold
request on all eight workers: 49 original forwards each and **60,760 strict
checks**. Its whole-forward warm rank-max median was **19.563s**; first step
**90.761s**, native worker request **1313.224s**. The driver then read the audit
before the other ranks had flushed and asserted at its eight-PID gate, before
media export or either repeat. It is **not** a completed three-request lifecycle.
See [failed cold-segment summary](failed_lifecycle127a_summary.json),
[failed driver receipt](lifecycle_result127a_failed.json) and
[exact failed sources](executed_sources127a/generate_lifecycle127.py).

The corrected [audit completion protocol](audit_completion127.py) publishes one
atomic marker **after** each rank's JSONL closes. The driver waits using inotify,
with a 120s timeout and no device collective or busy-poll; the wait is outside
generation timing. A real Linux thread/marker unit and all12 portable tests pass.
The [failed run's private IPC](cleanup_failed127a.json) has been removed after
all live process environments exited; failures and sources are preserved.

Fresh corrected `engine_lifecycle127b`, PID1941731, failed **before** the first
request: rank5 exited −11 while the native stack was in a CPU Half `copy_` during
layerwise initialization. No Python tensor/line was available to identify the
root cause; this is not evidence that the new audit marker or a numerical kernel
failed. See [native init failure](failed_lifecycle127b_summary.json) and
[its exact source](executed_sources127b/launch_lifecycle127.py).
Its [private IPC cleanup](cleanup_failed127b.json) preserves model/source evidence.

A bounded diagnostic `engine_lifecycle127c`, PID1947359, has completed.
It adds [CPU packing metadata and faulthandler](init_diagnostics127.py) while
still calling the original native `copy_` path. The
[synthetic native-copy unit](init_packing_unit127.json) proves six small
contiguous/transpose/offset copies word-identical; it does **not** establish the
full-model segfault root cause. The launcher/driver/collector/media roots are
consistent and all14 portable tests pass. [Pinned127c sources](launch_source_pins127c.json)
include those guards; the process receipt is now atomically published with its
launcher PID to avoid collector receipt races.

The three-request observation has now **completed**, with the same8 workers and
model objects, **182,280 strict Half checks**, and124frames for every request.
Driver generation wall is **1339.217 / 1009.406 / 1008.572s**, while whole-forward
warm medians remain about **19.90s**. Full VAE stage is **142.268 / 23.170 / 23.135s**;
all requests still perform native stage reloads and every observedcopy stream is
shared with the consumer stream. See [complete findings and next-step contract](RESULTS.md),
[actual lifecycle summary](lifecycle_summary127c.json) and
[unchanged native-repeat media differences](native_repeat_media127.json).
This is not a precision promotion or attainment of latency targets.

[Successful cleanup](cleanup127c.json) verifies all private processes exited,
private IPC removed and accepted source/noise fingerprints unchanged. The owned
machine lease is released. The nativeCPU packing failure remains unresolved in
[Issue#14](https://github.com/lvyufeng/h3-910a/issues/14), not hidden by the later
successfulrun. Follow [Issue#3](https://github.com/lvyufeng/h3-910a/issues/3) for live
closure and [Issue#4](https://github.com/lvyufeng/h3-910a/issues/4) for next development.
