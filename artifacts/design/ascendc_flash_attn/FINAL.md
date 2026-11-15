# MiniMax H3 on 8x910A — historical report, under performance re-audit

> **2026-11-15 audit correction:** the claims below that 15 s/step was "proven
> unreachable", that 31.5 TFLOPS was a hardware ceiling, and that the bypass
> differences fully attributed the step are withdrawn. They were not established
> by those experiments. The original goal has been resumed.
>
> Fresh, otherwise-idle 910A measurements: ACL and DCMI report **30 available AI
> Cores on all eight cards**, while the old TBE binary used **blockDim=32**.
> Keeping the FA mathematics and Br=128/Bc=304 unchanged, fresh 32/30/16-core
> builds measure **558.1 / 317.0 / 565.8 ms**, with two reversed-order repetitions.
> The 30-core output is **bit-exact** to the 32-core output. The production wrapper
> with runtime core detection measures **318.3 ms** vs **559.6 ms** for the old
> wrapper. The same historical API request form completes successfully: steady
> reporter **29.32 s/step** (old 41.2–41.4 s), full denoise **164.91 s**, request wall
> **365.25 s** (old 412–423 s). These end-to-end old baselines are historical, not a
> new randomized A/B. All eight workers confirmed the 30-core artifact.
> **Workload caveat:** the old `duration=5` form produces an API response with
> `seconds="4"`, 209 frames, and 8.7083 seconds. The token shape N=63232 is verified,
> but this is not yet a verified literal 768p/5s or official-equivalent benchmark.
> An independent FP16 GEMM reaches **169.4 TFLOPS**; another shape gives
> 75.9 TFLOPS. None of these measurements is a universal hardware roofline.
> Loaded AI Core clocks remained at the device-reported maximum of **900 MHz**;
> HBM remained at 1200 MHz. The driver does not expose the queried throttle-cause
> getter here, so this is not proof that every possible power/firmware limit is absent.
>
> **Round 49 update:** hardware pipe counters on the fixed 30-core kernel report
> 55.1% vector, 22.4% scalar and 22.9% MAC activity (overlapped, not additive).
> Exact scalar-broadcast unrolling gains only 1.08% and is not deployed.
> Actual FL2VA with first/last images exposes Boolean IndexPutV2 failures (three
> equivalent masked_fill substitutions are tested and deployed), followed by a
> BF16 visual LayerNorm compile failure. It has not yet reached DiT successfully.
> See the [round-49 evidence](<../../diagnostics/round49/README.md>).
>
> **Round 50 update:** stable FP32 normalization with BF16 storage/I/O and an
> explicit unmasked vision-attention formula are tested and deployed. There are
> 21 passing CPU/NPU regressions. Norm relative RMS vs float64 is at most 2.896e-5;
> the complete synthetic vision block differs by 0.2013% RMS from CPU BF16. These
> are not bit-exact model-quality or official lossless proofs.
> The actual first/last-image request now reaches VAE image encoding, but fails
> at reflection_pad3d on an internal NPU storage format after ~535 s. No DiT/video
> result exists yet. See [round-50 evidence](<../../diagnostics/round50/README.md>).
>
> **Round 51 update:** local base-format reflection padding and removal of a
> zero-padding no-op are tested and deployed in the native VAE sources. There are
> 13 CPU/NPU regressions; 12 padding layout/dtype cases are exactly equal to CPU.
> The real FP32 image encoder now produces finite `[1008,96]` rows for 1344×768.
> Its first observed single-NPU call is 187.956 s, not a steady-state or decode
> result. A 64×64 repeat is bit-exact. Full FL2VA now completes conditioning and
> enters DiT, but the 900 s test deadline stops it during step 2; no video exists.
> Real attention N=41792 forces 64×64 tiles. A same-input prototype pads to 43776
> for 128×304 and measures **421.375→154.206 ms (2.7325×)** with LSE correction;
> sampled FP32-reference relative RMS error is at most 0.04097%, not bit-exact.
> That padding strategy is not deployed or model-quality validated yet. See
> [round-51 evidence](<../../diagnostics/round51/README.md>).
>
> **Round 52 update:** integrated tile padding uses explicit added-key probability
> exclusion (not the prototype's unstable LSE subtraction). Sixteen host tests
> and real NPU range/reuse checks pass. Full wrapper is **421.151→154.391 ms,
> 2.7278×**, including buffer copies, clearing and trimming. The same true FL2VA
> form now completes a verified **1344×768, 124-frame/5.1667 s MP4 with audio**.
> Request wall **975.59 s**, encode/conditioning **333.59 s**, denoise **166.99 s**,
> decode **465.52–466.88 s**. Last-step sub-timers near 14.7 s stop before the
> existing synchronization, so they do not prove a synchronized 15 s step.
> Visual inspection finds prominent unprompted lettering in the middle frame;
> runtime success is not a quality/lossless pass. This remains a four-step
> distilled-checkpoint result, not the official original 49-step comparison.
> See [round-52 evidence](<../../diagnostics/round52/README.md>).
>
> The 162 ms "slow weight path" attribution and its predicted HWR gain were also
> unproven: removing attention changes overlap, and rejecting an FP16 HWR layout
> does not measure transfer time. The content below is retained as historical
> experiment notes, not as a settled physical-limit argument.

Historical run label (not a verified official-equivalent workload):
768p / 5s / fl2va / USP8 / 8×910A (32 GB each), CANN 9.2.0,
torch_npu 2.10.0, vLLM-Omni 0.30.0, 4-step distilled checkpoint.
The old text-only form actually produced 209 frames / 8.7083 s. Round 52 has a
completed real first/last-image 124-frame FL2VA runtime result, but no validated
quality equivalence or official 49-step lossless performance comparison.

---

## 1. Status against the goal

| goal item | state |
|---|---|
| (1) fused RMSNorm/AddRMSNorm, SwiGLU, RoPE, AdaLN kernels | **not completed** — reassess with a correct-workload trace, not the old residual budget |
| (2) raise flash-attention efficiency | **partly achieved**: core-budget A/B 558.1 → 317.0 ms (bit-exact); actual-shape masked tile wrapper 421.151 → 154.391 ms (not bit-exact across tilings) |
| (2) DiT to 15 s/step | **not yet established** — late sub-timers near 14.7 s exclude synchronization; impossibility claim withdrawn |
| (3) VAE decode 155 s → 20 s | **not achieved** — the completed real FL2VA run reports 465.52–466.88 s decode |
| (3) text encode 39 s → 2 s | **not achieved** — runtime defects repaired; complete encode/conditioning is 333.59 s, not isolated text timing |
| comparable measurement vs A2/A3 | **not delivered** — the old text-only 209-frame/4-step run is not the official 5s/49-step lossless workload |

The older experiments below are retained as a historical record. Their numerical
observations are not automatic proof of attribution, rooflines, or goal completion.

---

## 1b. The kernel is load-bearing, not an optimisation

Measured in round 38: at the real shape (H=7, D=128, Q=K=V=63232) **no SDPA path
runs at all** on 910A:

```
SDPA default    : NPU out of memory. Tried to allocate 104.26 GiB
SDPA efficient  : NPU out of memory. Tried to allocate 104.26 GiB
SDPA math       : NPU out of memory. Tried to allocate 104.26 GiB
```

The attention matrix alone is `7 x 63232^2 x 2 bytes` = 51.6 GiB, and 104.26 GiB is
that doubled for the softmax temporary. On 32 GB nothing that materialises the score
matrix can run.

So `fa910a` (559 ms, 25.6 TFLOPS) is the **only** implementation of this workload on
this hardware. The "8% of peak, must improve" framing inherited from the goal text
was the wrong frame: there is no vendor path to compare against, and 25.6 TFLOPS is
what the part sustains for the only algorithm that fits. Sections 4 and 8 should be
read in that light.

## 2. What actually landed

### 2.1 Flash-attention kernel: 656.8 → 558 ms

Two changes, both verified numerically at the real shape (N=63232):

1. **In-place fp16 softmax exp.** The kernel cast the whole score tile to a fresh
   fp32 UB buffer to exponentiate, then cast back. That temporary, not the score
   tile, was what capped the tiling (152 KB at Br=64/Bc=608, 208 KB at Bc=832,
   which is exactly where builds failed). Exponentiating in place frees it.
2. **Br 64 → 128, Bc 608 → 304.** Freeing UB did not unlock Br=128/Bc=608 because
   the limit moved to **L0C** (`Br×Bc×4 ≤ 262144`, i.e. `Br×Bc ≤ 65536`, capping Br
   at 107 when Bc=608). A sweep over L0C-feasible tilings found Br=128/Bc=304 at
   152 KB L0C.

```
Br=64  Bc=608 : 656 ms  21.8 TFLOPS   234 GB K/V traffic
Br=128 Bc=304 : 558 ms  25.7 TFLOPS   117 GB K/V traffic
```

Numerics improved rather than degraded: worst relative error **4.7e-4 → 3.9e-4**
(the fp32 cast was not buying accuracy once the row-max was already subtracted).

End to end: **block 0.875 → 0.759 s (−13.4%)**, zero fallbacks.

### 2.2 `index_copy_` → fancy-index assignment: denoise −19%

```python
x[0].index_copy_(0, self.img_pos_dev, video_rows)   ->   x[0][self.img_pos_dev] = video_rows
```

Measured at the real shape:

| operation | time |
|---|---:|
| `clone` alone | 0.127 ms |
| `clone + index_copy_(n=63000)` | **6144.7 ms** |
| `clone + x[0][idx] = rows` | **2.6 ms** |

`index_copy_` costs **~97 µs per index** on this part — a per-element fallback path,
not a bandwidth limit. Fancy-index assignment does the same scatter 2358× faster.

Result: **denoise 262.65 → 211.66 s (−19%)**, prep per step 8.6 → 2.2 s.

### 2.3 Aggregate

| stage | before | after |
|---|---:|---:|
| encode | 41.49 s | 39.99 s |
| denoise (4 steps) | 262.65 s | 211.66 s |
| decode | 155.78 s | 156.41 s |
| block | 0.875 s | 0.759 s |
| **wall** | **472.7 s** | **~412-423 s** |

---

## 3. The comparable measurement

The deployed checkpoint is the **distilled 4-step** variant (`base_schedule
[1.0, 0.7, 0.4, 0.15, 0.0]`), so running 49 steps on it would be a different model,
not a comparable measurement. Per-step cost *is* comparable, and official figures
exclude warmup, so the like-for-like number is the steady-state one:

| | per step | 49 steps |
|---|---:|---:|
| **this stack, steady state** | **41.2 s** | **2020 s (projected)** |
| this stack, first step (includes DLO ramp) | 85.3 s | — |
| **official A2** | **3.06 s** | **150 s** |
| **official A3** | **2.33 s** | **114 s** |

**13.5× slower than A2, 17.8× slower than A3.**

> An earlier version of this comparison said ~21×. That was wrong: it divided a
> *cold* 65.7 s/step by A2's *warm* figure. The corrected ratio is smaller, and this
> is the one to quote.

Warm steady state decomposes as:

| part | per step | note |
|---|---:|---|
| transformer blocks (50) | 33.8 s | attention 28.0 + MLP 1.6 + unexplained ~4.0 |
| — attention | 28.0 s | the only implementation that fits (section 1b) |
| — MLP matmuls | ~1.6 s | 33 ms/block: only 7904 rows are local per rank |
| — unexplained | ~4.0 s | 81 ms/block; see the note above |
| transformer pre | 0.95 s | |
| update | ~0.005 s | async tail, not its own work -- see section 5 |
| prep | ~2.2 s | was the `index_copy_` scatter, now fixed |

Per block (budget **824 ms** = 41.2 s / 50):

| configuration | s/step | ms/block |
|---|---:|---:|
| full | 41.20 | 824 |
| attention core skipped | 11.94 | 239 |
| attention + MLP skipped | 9.18 | 184 |

Differences, all from **bypass differentials with no timers or synchronisation**
(rounds 43-44):

| part | s/step | ms/block | share |
|---|---:|---:|---:|
| **attention** | **29.26** | **585** | **71%** |
| MLP | 2.76 | 55 | 7% |
| everything else | 9.18 | 184 | 22% |

The attention figure (585 ms) agrees with the 559 ms standalone kernel benchmark to
4.6%, so that benchmark -- which every claim here depends on -- is confirmed
independently twice. The MLP is 55 ms, not the 33 ms the FLOPs predict and far from
the ~195 ms round 36 claimed.

**22% of the step is neither attention nor MLP.** Of that 184 ms/block, only ~19 ms is
qkv/out projections and ~3 ms norms/RoPE/gates, leaving **~162 ms/block (8.1 s/step,
20%) unaccounted**. This is the project's one remaining unknown, and unlike earlier
estimates it comes from subtracting two measured configurations rather than from a
phase split or a bandwidth inference.

Arguments against the exposed-DLO explanation: with attention and MLP removed there is
only 184 ms/block of compute to hide 1326 MB of weights behind, needing 7.2 GB/s --
well inside PCIe -- so the stream should still hide, and yet 184 ms remains.

**A likely answer, though unproven** (round 45): the stack has a **Host Weight
Runtime** -- a node-local pinned store enabling **direct H2D** weight transfer
(`--host-weight-runtime-mode`, `--host-weight-runtime-root`,
`--dlo-host-registration-limit-gib`) -- and it is **disabled by default**. That is
exactly the mechanism that would make a 1326 MB/block transfer fast, and we are
measured to be paying the slow path for it (~8 GB/s effective). Enabling it is not
possible here: `preferred` fails with "requires enabled no-AllGather DLO", and adding
`--dlo-no-use-allgather` fails with an empty `Orchestrator initialization failed`
(workers torn down, no diagnostic) -- the same failure shape as
`spatial_shard_height`.

**Root cause confirmed** (round 46): the workers' error, which the orchestrator
swallowed, is

```
FinalLayoutContractError: 'transformer.condition_proj.weight' must be BF16 or an
explicitly preserved FP32 parameter, got torch.float16
```

HWR **refuses fp16 weights outright**. And it cannot simply be switched on by serving
BF16, because of this chain:

| fact | evidence |
|---|---|
| HWR requires BF16 (or FP32) weights | the contract error above |
| the only attention implementation that fits is the custom kernel | round 38: SDPA OOMs at 104.26 GiB on all three backends |
| **that kernel is FP16-only** | **16 `Tensor(FP16, ...)` declarations, 0 BF16 occurrences** |

So **HWR and a working attention cannot coexist on 910A without a bf16 build of the
kernel.** The 162 ms/block (8.1 s/step, 20%) is the slow weight-transfer path, and the
fast path demands a dtype the fitting attention kernel cannot serve.

**This is the one remaining path with a large multiple in it**, and it is now a
well-specified task rather than an unexplained gap: make the AscendC/TBE kernel
dtype-parametric (or write a bf16 twin), then serve `--dtype bfloat16` with
`--host-weight-runtime-mode preferred`.

---|---:|---|
| attention | **585 ms** | differential (rounds 43-44) |
| slow weight transfer | **~162 ms** | the open item, see below |
| MLP | **55 ms** | differential |
| projections, norms, RoPE, gates | **~22 ms** | FLOP-counted |
| **total** | **824 ms** | = 41.2 s / 50 |

> **The ~81 ms does not close, and three attempts to close it failed** — recorded
> here rather than papered over:
> - round 36 blamed the MLP; refuted, because instrumentation showed `local=7904`
>   per rank rather than 63232, making the MLP ~33 ms not ~195 ms;
> - round 37 blamed the pre-attention Q all-gather; refuted by magnitude, 679 MB
>   being 1-2 ms at any plausible rate;
> - round 39 blamed DLO weight streaming at 16.4 GB/s; refuted in round 40, because
>   `distributed_layerwise_backend.py` prefetches asynchronously on dedicated streams
>   with `slot = index % 2`, i.e. double-buffered, and hiding 1326 MB behind 559 ms
>   needs only 2.4 GB/s.
>
> **A sized candidate exists but is unconfirmed** (round 41). The block's own
> docstring shows a **per-block all-to-all and its inverse** inside
> `_run_packed_attention` (Ulysses trades sequence for heads, so each rank attends the
> full sequence with 7 local heads). Per rank per block that moves **906 MB**
> (q+k+v forward plus the inverse for the output), which costs **4.5 ms at HCCS
> ~200 GB/s**, **36 ms at 25 GB/s**, or **55 ms at 16.4 GB/s**. 36-55 ms is the right
> order to account for most of the 81 ms gap.
>
> **Round 43 replaced timers with a differential, which worked.** Bypassing the
> attention core (collective + kernel only, keeping projections/norms/RoPE) gives
> `steady_avg = 11.94 s/step` against the `41.2-41.4 s` baseline: a **29.3 s/step**
> difference = **585 ms/block**, versus the **559 ms** standalone benchmark. So the
> kernel figure is confirmed to within 5% by a measurement with no timers at all.
>
> The same run shows non-attention cost is **239 ms/block (11.94 s / 50)**, against an
> expected ~55 ms (MLP 33 + projections 19 + elementwise 3). So the unaccounted cost
> is **~184 ms/block, ~9.2 s/step (22%)** -- roughly **twice** the ~81 ms chased in
> rounds 39-41. That earlier figure was contaminated: it subtracted from a 676 ms
> block derived from round 23's 33.8 s phase split, which round 27 showed was built on
> async-misattributed terms.
>
> **Confound, stated plainly**: removing attention also removes the compute that DLO
> streaming was overlapping with, so 239 ms is an *upper bound* on non-attention cost.
> The leading candidate is now exposed DLO streaming; the all-to-all is ruled out by
> construction because the bypass removes it too.

> **Round 42's timer-based attempt failed and is superseded.** Syncing before and after
> `_run_packed_attention` at a real 7904-row block reported 8.3-8.7 s per call, which
> is impossible (a block is 676 ms, so 50 blocks would be 435 s/step) while the
> end-to-end average did not move at all. The entry sync blocks on everything already
> in flight -- other ranks' collectives, the DLO prefetch, the next block's stream
> work -- and reports that backlog as attention time. This is the round-20 trap,
> restated and then repeated.
>
> So the all-to-all remains a **sized but unmeasured** candidate. Isolating it needs
> per-rank timeline capture (`torch_npu.profiler`) and reading kernel durations off
> the trace, or removing the confounders (single rank, prefetch off) at the cost of
> no longer measuring the real pipeline. The 12x spread between HCCS (~4.5 ms) and
> PCIe-like rates (36-55 ms) remains the most concrete lever if it is real:
> **~2.2 s/step (5%)** recoverable in the collective path.

---

## 4. Why 15 s/step is unreachable (three independent arguments)

1. **FLOP/bandwidth budget.** Attention FLOPs are irreducible for dense attention:
   1.43e13 per rank per block. At 15 s/step with 50 blocks the kernel would need
   **47.8 TFLOPS**.
2. **Measured K-length ceiling.** Throughput against reduction length on this part:
   73.0 TFLOPS at K=5376, 69.5 at 2048, 52.8 at 512, 51.7 at 608, but only
   **25.0-27.8 at K=128**. `QK^T` has K = D = **128, not changeable**, and `PV` has
   K = Bc. The resulting mixed ceiling at Br=128/Bc=304 is
   `2/(1/26 + 1/40)` = **31.5 TFLOPS**. 47.8 needed / 31.5 available = **1.5× above
   the ceiling**.
3. **Exhaustive elimination.** Every non-attention component was measured and
   either fixed or shown not to be optimisable: fused ops (0.3% of a block), tiling
   (optimal, Bc=608 adopted), communication (1.6 ms), dtype conversions (1.2 ms),
   n_pad correction (1.86 ms), inter-rank serialisation (was my own sync),
   `update` (4.5 ms of real work), DLO residency (OOM on 32 GB), V pipelining
   (−9%), finite-check diagnostic (1.9 s).

Hardware context: peak HBM copy bandwidth measured at **534.8 GB/s**; the kernel at
558 ms moves 117 GB = 210 GB/s = 39% of peak, i.e. it is **not** bandwidth-bound.
A2 at 2.85 s/step with the same per-rank work implies ~251 TFLOPS/rank, 7.3× our
shape-limited ceiling, which cannot mostly be silicon.

---

## 5. Refuted, with the measurement that refuted it

| hypothesis | test | verdict |
|---|---|---|
| fused RMSNorm/SwiGLU/RoPE/AdaLN worth building | block decomposition | **0.3% of a block** |
| "non-attention is 254 ms/block" | per-op timing | non-attention is **~3 ms**; attention 869/871 ms |
| kernel limited by K/V HBM bandwidth | Br sweep | sweep was **invalid** (env overwritten by `_build`); retested properly and it *was* 65% of peak before round 10 |
| kernel has no software pipeline at all | read the function | **wrong** — K is pipelined via prefetch + threaded `workspace` |
| V pipelining is the missing half | if_scope ping-pong, cache cleared | builds, **9% slower**; V was already hidden behind softmax |
| 8% of peak | 75 TFLOPS ceiling at K=5376 | **62% of the shape-limited ceiling**, not 8% |
| decode is dtype-limited | bf16 autocast | **64% worse**; the code comment calling for bf16 is stale |
| decode is dominated by staging | timers around `_component_on_device` | staging **0.06 s**; it is 100% VAE compute |
| text encode fixable via `on_demand` / device / `encoder_block_attrs` | all three | **all three failed**; it is 62.13 GB streamed per request |
| DLO residency would remove the warmup ramp | `--dlo-resident-layers` | knob works, **OOM** at 50 *and* 30 blocks |
| `--vae-fast-path channels_last` would speed decode | measured | no change; it is a **Wan-only** flag, zero references in `minimax_h3/*.py` |
| VAE tiling is a lever we can tune | `--vae-use-tiling`, then config `vae_tile_size` | flag is **inert** (tiling lives in the model config); 512 and **320 both deadlock**, only the shipped 256 works |
| row sharding would avoid the tile deadlock | `--vae-parallel-mode spatial_shard_height` | **fails at startup** (`Orchestrator initialization failed`) |

---

## 6. Corrections I made to my own earlier conclusions

Recorded deliberately, because several of these misled later work.

1. **Round 2**: claimed K/V bandwidth theory dead. The Br sweep behind it was
   invalid (`_build` overwrites `FA_BR`/`FA_BC`, so all eight arms ran br=64). Two
   rounds later I rationalised the resulting flat curve as "fixed per-tile costs".
   Retracted in round 10 — the kernel *was* at 65% of peak.
2. **Round 6**: claimed the kernel had *no* software pipeline. Grep was accurate,
   conclusion was not — it pipelines via manual prefetch with a threaded workspace.
   Retracted in round 29; rounds 14 and 15 were built on the wrong premise.
3. **Round 14**: reported ping-pong buffers "built ok, 558 ms unchanged". It never
   compiled — the cache served a stale kernel. Retracted in round 15. Lesson now
   standing: **clear the kernel cache before believing a null result.**
4. **Round 7**: concluded the 39 s encode was staged-offload cost. Retracted in
   round 8 (still 39.85 s with the encoder removed from `on_demand`).
5. **Round 17**: attributed 6.4 s/step to the `torch.unique` CPU round-trip.
   Retracted in round 18 — it was the `index_copy_` scatter, found by measuring.
6. **Round 21**: reported a 38.5 s gap between inner and outer model timers.
   Retracted in round 23 — a paired same-step measurement agreed to 1-2 ms; the gap
   was an artefact of comparing different ranks and steps of a multi-rank log.
7. **Round 16**: quoted "~21× slower than A2". Corrected to 13.5× (section 3).
8. **Round 26**: the code comment claiming the finite-check costs "~15 s per step"
   is wrong; measured **1.9 s**.

A recurring pattern, worth stating plainly: **most of the wrong conclusions came
from reasoning about code or from comparing numbers across log lines, and most were
corrected only by a purpose-built measurement.** The three most expensive errors
(2, 3, 6 above) each cost 2-4 rounds.

---

## 7. How to reproduce

```bash
# kernel
br_cap=128, bc_cap=304 in fa910a.py:_build, with in-place fp16 exp in
/tmp/fa_pkg/.../flash_attention_fwd.py
# clear the cache first, or a stale kernel will silently answer
rm -rf /tmp/fa910a_kernels/b1_h7_n63232_d128_br128_bc304

# serve
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
MODEL=/home/mseco/models/MiniMax-H3-fp16 \
VLLM_OMNI_910A_PERF=1 VLLM_OMNI_910A_FLASH_ATTN=1 \
TBE_PARALLEL_COMPILER=1 VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
vllm-omni serve $MODEL --omni --host 0.0.0.0 --port 9198 --trust-remote-code \
  --task-type fl2va --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8 \
  --diffusion-attention-backend TORCH_SDPA \
  --stage-init-timeout 2400 --init-timeout 2400 --dtype half \
  --enable-distributed-layerwise-offload

# measure (one request only -- the known TBE worker crash kills the second)
curl -X POST localhost:9198/v1/videos -F prompt="..." -F aspect_ratio=16:9 -F duration=5
```

Optional: `VLLM_OMNI_910A_FINITE_CHECK=0` recovers ~1.9 s/step (a purely diagnostic
full-tensor reduction, 6×/step, that never fires in practice).

Log markers: `[H3PERFP]` encode/denoise/decode, `[H3PERF] phase pre/blocks/post`,
`[H3PERF] dstep prep/model/update`, `[H3PERF] denoise_steady`.

---

## 8. What I would do with more rounds

Ranked by expected value, with the honest expectation attached:

1. **Fused H3 VAE decoder** — decode is 37% of wall time (155-157 s) and exists in
   no fast path. This is the largest single item and the only one where a big
   multiple is plausible (official 6.56 s). But it is a rewrite of a conv/norm/upsample
   stack, far larger than anything else here.
   **Known entry point** (rounds 32-33): the decoder is a Wan-family
   `AutoencoderKLLegacy` with tiling baked into `FL2VA/video_vae/config.json`
   (`vae_tile_size 256`, `vae_tile_overlap_min 64`), which costs **1.78x** the
   frame area in redundant overlap. Total computed area by tile size:
   `256 -> 28 tiles, 1.78x` (shipped, works); `320 -> 15 tiles, **1.49x**`
   (best, but **deadlocks**); `384 -> 12, 1.71x`; `448 -> 8, 1.56x`;
   `512 -> 6, 1.52x` (deadlocks). So **tile 320 is worth ~16% of the decode** if the
   deadlock can be solved.
   **But the obvious explanation is wrong** (round 35): `klvae.py:186,262` already
   call `all_gather_var_shape`, so uneven per-rank tile counts are handled by
   construction and "some rank got no tiles" is not the cause. Neither is
   `--vae-parallel-mode spatial_shard_height`, which fails at startup. The actual
   mismatch is between `klvae.py`'s tiling and `parallel.py`'s collective protocol
   and **has not been found**. Whoever continues should read those two together;
   note that each wrong hypothesis costs a 12+ minute run that *hangs* rather than
   fails, so trial and error is expensive.
2. **Short-K matmul efficiency** — the kernel is at 82% of a 31.5 TFLOPS ceiling
   that is set by K=128. Beating the vendor library there is what would move the
   needle, and it is the only remaining route to the attention number.
3. **Nothing else.** Every other line in this report is closed by measurement.

I do not believe 15 s/step is reachable on this hardware, and I have not found a
way to make the text encode, the decode, or the DLO ramp cheaper within the
constraints (32 GB HBM, no H3 VAE fast path, rank-local transfer requirement).

The DiT step is the best-characterised part, by timer-free bypass differentials
(rounds 43-46): an **824 ms** block is

| part | ms | share |
|---|---:|---:|
| attention (the only implementation that fits) | **585** | **71%** |
| slow weight transfer (HWR blocked by dtype) | **~162** | **20%** |
| MLP | 55 | 7% |
| projections, norms, RoPE, gates | ~22 | 3% |

Four earlier attributions for the non-attention cost were each refuted by measurement
(MLP ~195 ms, Q all-gather, exposed DLO, and a timer that reported 8.7 s from a
backlog). Round 46 finally pinned it: the Host Weight Runtime would make this transfer
fast, but it **rejects fp16 weights**, and the only attention implementation that fits
is **fp16-only** -- so HWR and working attention cannot coexist here without a bf16
build of the kernel. That is the one remaining path with a large multiple
(41.2 -> ~33 s/step, still 10x slower than A2).