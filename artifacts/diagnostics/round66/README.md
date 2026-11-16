# Round66 — current-stream attribution of the stable resident text stack

## Result: Norm is the largest observed whole-call interval

The clean same-run resident forward is4.14475s baseline and3.60056s combined
cache (1.151×), consistent with Round65 rather than a new kernel speedup. Clean
after-profiling medians are4.14543/3.59405s; restored original math is4.16308s.
The profiler preserves **all192 complete final text tensors** raw-bit-identically,
as well as every actual input/parameter byte; the baseline again reproduces the
Round60 complete reference tensor. This remains an experimental-stack identity,
not originalBF16/HF/official generation losslessness.

For the combined mode, the median rank-max **profiled** call is3.61642s
(call1,rank4). Its actual same-forward, same-rank non-overlapping top intervals:

- All200 Norm calls: **1.54511s** (~42.7% of the profiled wall).
- All100 row calls, **including** collective-visible work/casts:0.76047s.
- All200 Q/K/V/merged gate-up local calls:0.46186s.
- FP32 causal attention,50 calls:0.35726s.
- Shared activation entry casts,50 calls:0.08036s.
- DeepStack injections:0.04208s; SwiGLU:0.02156s; RoPE pairs:0.01406s.

Those top intervals sum3.28275s; **0.33367s remains unassigned**, not labelled
as CPU or communication. Nested `all_reduce` intervals total0.34264s, **already
inside the row time**, and are not added again. Nested `F.linear` intervals total
0.14881s but are not a reliable pure-FP32-compute budget: auxiliary streams,
consumer waits, dispatch and producer layout work can move outside those narrow
brackets. Subtracting children from parents does not prove the remaining time is
only casts, nor that all those intervals are removable.

Within the four Norm families, residual-add hidden Norm is0.81286s, hidden-input
Norm0.49695s, Q-head Norm0.15334s and K-head Norm0.08197s. This makes the actual
prescribed **residual/FP32-statistics/BF16-round/gamma path** the next controlled
optimization target. It does not justify adopting the previously rejected
fully-fused Norm whose changed reductions yielded6.318% final-tensor RMS error.

The median event-instrumented wall exceeds clean-before by0.578% in baseline and
0.441% in combined mode. Clean after drift is+0.0165%/-0.181%, respectively.
[Complete summary](<summary.json>), [same-vector compact breakdown](<breakdown.json>)
and [rank0 raw events](<ranks/profile_rank0.json>) retain the other calls/ranks and
both host-dispatch/current-stream quantities. No mixed-rank or independent
per-category maximum is presented as an actual forward.

## Scope

This is diagnostic profiling, **not a new optimization or a full text-encode
stage measurement**. It replays the same real checkpoint and actual2043-token
captured inputs as [Round65](<../round65/README.md>). Its fixed experimental stack
uses FP32 projection/causal math,1024-output K/V padding, the prescribed BF16/FP32
Norm rounding sequence, native BF16-GM SwiGLU and native-round TBE RoPE. No model
or production source is modified. Processor, vision encoding, loading/offload,
DiT/VAE and official49-step generation are excluded.

## Measurement design

[Reproducer](<text_profile_tp8.py>) copies the exact prior loading, local/row
projection, merged-MLP, parameter/input hashing and whole-forward timing logic.
The only new behavior wraps original functions for measurement. Norm arithmetic
bodies, original baseline row math and the converted-cache row math are unchanged.

For baseline and combined shadow/shared-activation modes:

1. One first forward is excluded (kernel/padding/shadow initialization).
2. Three clean forwards with event recording disabled.
3. Three profiled forwards with NPU event pairs and CPU dispatch wall.
4. Three more clean forwards to expose profiling perturbation/drift.

Finally original math/dispatch is restored for one excluded and three clean
forwards. As in Round65, already allocated FP32 shadows stay resident until final
cleanup; restoration is not a claim about returning to baseline HBM residency.

[Event recorder](<event_profile66.py>) preallocates/primes1400 event pairs before
measured forwards. It reserves a parent's event pair **before** nested calls,
preventing children from overwriting it. No component call synchronizes. The
original complete-forward end drain happens inside its wall timer; only after
that wall timer and CPU download does reporting read event timestamps. A second
untimed full drain makes event readiness explicit.

Clean calls still traverse disabled Python measurement wrappers, but **create/
record/read no events**. They are same-run controls, not a claim that the wrapper
indirection itself costs zero. Comparison with previous uninstrumented timings
is kept as a stability check, not an additive optimization gain.

## Non-overlapping parent intervals and nested children

Per profiled forward, top intervals contain50 each of hidden-input Norm,
residual-add hidden Norm, Q-head Norm, K-head Norm;50 Q projections;100 padded K/V
projections;50 merged gate/up projections;50 O rows;50 down rows;50 causal calls;
50 RoPE pairs;50 SwiGLU calls; and3 DeepStack injections. Combined mode additionally
has50 activation-scope **entry-only** casts, not50 enclosing attention spans.

A row interval includes its FP32 input/weight preparation, GEMM, collective
submission/visible dependency and output cast. Each local projection includes
its original casts/GEMM/crop/contiguous/output cast. These scopes do not overlap
other **top** intervals on the program's logical current stream.

`child.linear_*` (300 calls total) and `child.all_reduce` (100 calls) are nested
inside those parents. Their timings are provided for inspection but **never
added to the parent sum**. Norm is initially profiled as a whole, not assumed to
be a particular reduction or cast bottleneck.

## What event elapsed time means—and does not mean

- NPU event elapsed is a **current-stream interval**, potentially including
  device work, stalls, host enqueue gaps, dispatch and dependency waits. It is
  not a pure kernel occupancy or compute-utilization measurement.
- CPU `perf_counter` around the original function is dispatch/call wall, not
  pure CPU work. It overlaps the device interval and cannot be added to it.
- HCCL may use internal communication streams. Current-stream event intervals
  around `all_reduce` measure only what is visible there; waits may appear in
  subsequent output casts/consumers. They do **not** identify network bandwidth,
  sum of HCCL kernel times or a removable pure communication cost.
- Position-RoPE generation, attention output transpose/layout, final residual
  adds, Python dispatch gaps, event bookkeeping and other unwrapped work are not
  forced into an invented category. The signed `wall − sum(top events)` remains
  explicitly unassigned. No extra stream fence is inserted to force closure.
- For each profiled call, every rank's entire vector is retained. The displayed
  representative comes from that call's **slowest complete-wall rank**. We never
  add per-category maxima that might come from different ranks/forwards.

[Aggregator](<summarize.py>) rejects missing ranks/events, incorrect exact event
counts, nonfinite/negative durations, changed source or tensor/parameter bytes,
incomplete clean/profile repeats and changed current streams. Full clean and
profiled outputs must preserve all10,460,160 values, not a sampled checksum or
relative-RMS-only gate. Clean performance remains median of **per-call rank
maxima**; profiled duration and instrumentation perturbation are reported
separately.

## Reproduction and acceptance

```bash
python3 -m unittest discover -s artifacts/diagnostics/round66 -p 'test_*.py' -v
python3 artifacts/diagnostics/round66/summarize.py
```

Real-NPU execution uses a fresh owned lease and a fresh output/claim ID; do not
replay or overwrite an active/evidenced run. Profiling can motivate the next
exact Norm/projection/collective experiment; it is not proof of text≤2s, original
BF16/HF quality, whole-A/V identity or official task-matched generation speed.

[Final source verification](<final_sources.json>) matches23 unchanged production/
model/dedicated-module fingerprints and13 profiling/prior-dependency fingerprints.
The [live fleet probe/release receipt](<cleanup_receipt.json>) records eight idle
cards, no accelerator processes and the owned lease released. The
[regression receipt](<test_receipt.json>) records16 new event/math/attribution/
actual-proof tests plus58 prior cache/service/kernel/block tests, all74 passing.

## Next exact-kernel gate

Do **not** replace the FP32 statistics/reduction tree as the first experiment.
Keep the actual residual sum's BF16 rounding, `xf.square().mean(...)` and
`rsqrt(variance+eps)` operators unchanged. A smaller candidate can consume that
native FP32 inverse-RMS and fuse only:

1. BF16 input decoded exactly to FP32 × native FP32 inverse-RMS.
2. Explicit round-to-nearest-even **normalized BF16** boundary.
3. Decode normalized BF16 → FP32 gamma multiplication → BF16 output rounding.

It must first match actual same-input outputs for hidden/input, hidden/residual,
Q-head and K-head fixtures, with sign/subnormal/tie/exception policies explicit.
Preserving statistics is not by itself a proof that a new vector multiply or
codec has identical semantics. Only after those checks may full50-layer TP8
first/repeats/all-rank complete identity and clean timing count as a benefit.
This partial fusion may leave much of Norm cost intact, and even the observed
whole Norm interval is not a literal clean-time removable budget or a guarantee
of2s. Residual-add fusion, reduction changes or cache co-residency would each be
separate controlled experiments.
