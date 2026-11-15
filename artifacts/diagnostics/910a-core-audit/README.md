# 910A core-budget audit — 2026-11-15

## Main result

The old static FlashAttention kernel launched 32 equal-work tasks on a device with
**30 available AI Cores**. TBE's `CORE_NUM=32` is a compile-time architecture
value; both ACL's `ACL_DEV_ATTR_AICORE_CORE_NUM` and DCMI report 30 on **all eight
cards**. A compile-time task partition based on the wrong budget can leave two
long tasks for a second wave. It is not equivalent to a fine-grained scheduler
running many short tasks on 30 cores.

Fresh builds changing **only `core_num`**, all with Br=128, Bc=304, B=1, H=7,
N=63232, D=128, FP16 Q/K/V and FP32 L0C accumulation:

| launch partition | median raw-kernel time, order 1 | order 2 (reversed) | effective dense FLOP/s |
|---|---:|---:|---:|
| 32 tasks | 558.047 ms | 558.069 ms | 25.68 TFLOPS |
| **30 tasks** | **316.983 ms** | **316.985 ms** | **45.21 TFLOPS** |
| 16 tasks (control) | 565.814 ms | 565.818 ms | 25.33 TFLOPS |

The speedup is **1.761x**, a **43.2% latency reduction**, without changing
attention semantics or tensor contents. The 16-task control is as slow as the
32-task version despite using half as many launch tasks. This, the 30-core runtime
queries, and the low-utilization tail in the 32-task telemetry support the
second-wave scheduling diagnosis. A per-core hardware trace would be needed to
show the exact instruction/tail timeline; none was captured in this audit.

Whole-output 30-vs-32 and 16-vs-32 comparisons are **bit-exact** (max absolute
error 0). Independent FP32 sampled-query reference over all 63232 keys gives the
same max absolute error **8.78e-7** for all variants on the low-amplitude synthetic
inputs. This check preserves the original numerical behavior; it is not a new
model-quality or lossless benchmark claim.

Source: `core_count_ab_run01.json`. Each variant compiled under a unique name in
`/tmp/h3_corecount_ab_5m8a4pwk`, not from the production cache.

## Actual wrapper and integration

The deployed wrapper now queries the current device's available core count using
ACL, partitions the TIK kernel before compilation, and includes the core budget
in both the disk-cache name and in-memory cache key. It never launches an old
32-core artifact merely because the input shape matches. A bad/unsupported query
fails clearly rather than silently falling back to the wrong architecture count.
The old wrapper and old compiled object are preserved for rollback.

Whole-wrapper test, same input and output:

- old wrapper: **559.603 ms**, blockDim=32;
- corrected wrapper: **318.252 ms**, blockDim=30;
- output bit-exact, max absolute difference 0.

Eight host-only tests cover the runtime query, invalid counts, device-specific
caching, budget-specific binary cache identity, and rejecting an over-budget
cached artifact. `python -m unittest discover -s artifacts/diagnostics/910a-core-audit
-p test_core_budget.py -v` passes all eight.

## End-to-end result — and workload limitation

One managed fresh-engine request used exactly the old request form and normal
serving features (no attention/MLP bypass, no sync probe, no HWR change):

```text
prompt=a red ball bouncing
aspect_ratio=16:9
duration=5
8 devices, USP8, half, DLO, TORCH_SDPA with the custom FA override
```

It completed successfully. All eight ranks reported `core_budget=30 blockDim=30`
for the operative FA shape `B=1 H=7 N=63232 D=128 Br=128 Bc=304`.

- startup: 265.04 s (excluded from request wall time);
- request wall time: **365.25 s**, vs historical 412–423 s for that same request form;
- steady per-step reporter: **29.32 s**, vs historical 41.2–41.4 s;
- full denoise stage: **164.91 s**, vs historical about 211–213 s;
- encode: 39.47 s; decode: 156.11–160.12 s.

This is an integration success and a comparison to historical runs, **not a
fresh end-to-end old/new randomized A/B**. The raw kernel and whole wrapper have
controlled same-input A/B evidence; the older request baselines come from prior
session runs. The device processes were fully stopped after collecting the result.

**Important workload discrepancy:** the API response to the form above contains
`seconds="4"`, `num_frames=209`, `duration_s=8.7083`, and `size=null`. Thus the old
`duration=5`/`aspect_ratio` fields do not establish a literal 768p, 5-second request.
The FA token shape is confirmed, but a strict fixed-shape/fixed-duration official
lossless comparison still needs corrected request parameters and verified output
metadata. Do not label this result a new official-equivalent 5s benchmark.

Sources: `wrapper_validation.json`, `e2e_result.json`, remote
`/tmp/h3_corebudget_e2e_run01/serve.log` (also downloaded for local inspection).

## Clocks, power and GEMM

DCMI getter definitions were taken from this machine's installed SDK headers.
They are read-only. No clock, power limit, firmware or thermal setting changed.

For FA and all measured GEMM workloads:

- AI Core current frequency: **900 MHz**, equal to reported max **900 MHz**;
- HBM frequency: **1200 MHz**;
- tested temperatures: at most 56 C in the isolated suite;
- observed power peaks: about 289 W in the large GEMM, about 127 W in FA
  (DCMI raw units divided by 10, checked against `npu-smi`);
- no observed dynamic frequency drop during the sampled workloads.

This does **not** prove absence of all firmware/power limits: the driver's
throttle-cause/rated-power getters returned unsupported (`-8255`). A health getter
also reports `80F18003`, DDRA internal configuration error, alongside Health OK;
its performance relevance is unknown and it was not cleared or modified.

Isolated FP16 `torch.mm`, ND inputs/output, preallocated output, five synchronized
measurement groups after warmup, `2*M*K*N` FLOPs:

| M | K | N | median | throughput |
|---:|---:|---:|---:|---:|
| 11250 | 5376 | 28672 | 20.477 ms | **169.37 TFLOPS** |
| 4096 | 5376 | 8192 | 4.755 ms | 75.87 TFLOPS |
| 8192 | 8192 | 8192 | 14.118 ms | 77.88 TFLOPS |
| 4096 | 128 | 8192 | 0.299 ms | 28.77 TFLOPS |
| 8192 | 128 | 28672 | 1.458 ms | 41.25 TFLOPS |

Device-to-device copy: **543.42 GB/s** counting read+write traffic. Its benchmark
uses a copy engine; it is not a direct measurement of the FA kernel's memory
pipeline roofline.

The large GEMM reproducibly reaches about 169 TFLOPS at the same fixed clock, so
**73 TFLOPS was never an environment-wide compute ceiling**. Likewise, 25–28
TFLOPS at one short-K GEMM shape is a measurement of that implementation/shape,
not proof of a physical attention limit. FA already exceeds the previously claimed
31.5 TFLOPS ceiling after fixing just the launch partition.

The exact paper-peak accumulation contract (FP16 vs FP32 accumulation) was not
established by this suite. The FA kernel explicitly accumulates in FP32; GEMM API
outputs are FP16, but output dtype alone is not proof of internal accumulation
mode. No claim of a 2x accumulation penalty is made.

## Retractions and next useful work

The previous "15 s is physically impossible" and "82% of a hard ceiling" claims
are withdrawn. No impossibility has been shown, and the goal is active again.

The bypass runs in the historical report do not establish an additive hardware
budget: removing attention/MLP also changes offload overlap and synchronization.
In particular, the claimed 162 ms slow-HWR-transfer term was not directly measured;
HWR rejecting a dtype does not prove that term's cost or that changing dtype would
recover 20%. Failure to compile a valid BF16 probe is also not evidence of native
BF16 matrix support on 910A.

Next tasks are concrete rather than a guessed ceiling:

1. Correct the video API workload fields and reproduce a verified 768p/5s
   benchmark at the intended schedule, distinguishing first-step ramp and steady
   work without changing the computation.
2. Profile the **30-core** FA kernel's Cube/vector/MTE timeline using NPU hardware
   counters, with correct actual tile shapes. The old 32-core tail no longer masks
   that analysis. Evaluate genuine intra-core tiling and vector/Cube overlap.
3. Measure the remaining DiT/offloader, encoder and VAE stages under a consistent
   workload rather than attributing residual wall time by subtraction.

No ComfyUI route was used. Model weights/configuration and precision stayed
unchanged by this fix.
