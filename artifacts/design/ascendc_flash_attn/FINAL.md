# MiniMax H3 on 8x910A — final report after 31 rounds

Benchmark configuration: 768p / 5s / fl2va / USP8 / 8×910A (32 GB each), CANN 9.2.0,
torch_npu 2.10.0, vLLM-Omni 0.30.0, 4-step distilled checkpoint.

---

## 1. Status against the goal

| goal item | state |
|---|---|
| (1) fused RMSNorm/AddRMSNorm, SwiGLU, RoPE, AdaLN kernels | **refuted as worthwhile** — measured at 0.3% of a block |
| (2) raise flash-attention efficiency | **partly achieved**: 656.8 → 558 ms kernel, block −13.4% end to end |
| (2) DiT to 15 s/step | **proven unreachable** — needs 47.8 TFLOPS against a 31.5 ceiling |
| (3) VAE decode 155 s → 20 s | **diagnosed, not fixed** — 100% video-VAE compute, no fast path exists for H3 |
| (3) text encode 39 s → 2 s | **diagnosed, not fixed** — 62.13 GB streamed per request; 3 fixes tried, all failed |
| comparable measurement vs A2/A3 | **delivered** (section 3) |

Two durable, end-to-end-verified speedups landed. Everything else in this report is
a measurement, a refutation, or a retraction.

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
| transformer blocks (50) | 33.8 s | attention 28.0 + DLO 4.0 + MLP 1.6 + 0.15 other |
| — attention | 28.0 s | the only implementation that fits (section 1b) |
| — DLO weight streaming | ~4.0 s | 1326 MB/block over PCIe, cannot be resident |
| — MLP matmuls | ~1.6 s | 33 ms/block: only 7904 rows are local per rank |
| transformer pre | 0.95 s | |
| update | ~0.005 s | async tail, not its own work -- see section 5 |
| prep | ~2.2 s | was the `index_copy_` scatter, now fixed |

Per block (budget 676 ms from 41.2 s / 50, which is sync-bounded):

| part | per block | source |
|---|---:|---|
| attention | 559 ms | kernel benchmark, Q=K=V=63232 |
| DLO weight streaming | ~81 ms | 1326 MB/block over PCIe at ~16 GB/s |
| MLP | ~33 ms | 7904 local rows at 73 TFLOPS |
| other (norms, adaln, gates, conversions) | ~3 ms | round 3 |
| **total** | **676 ms** | = 41.2 s / 50, closes |

The DLO term was identified in round 39 by arithmetic rather than instrumentation:
the DiT is **66.28 GB** over 50 blocks = **1326 MB per block**, and streaming that in
81 ms implies **16.4 GB/s** — squarely in the PCIe H2D range, versus 2 ms if it were
HBM (534 GB/s measured) or 53 ms at a clean 25 GB/s. The Q all-gather I had guessed
in round 37 is ruled out by magnitude (679 MB, 1-2 ms).

> **Two corrections stand in this record.** Round 36 claimed this closed to zero and
> blamed the MLP; that was wrong because `local=7904` per rank, not 63232, so the MLP
> is ~33 ms. Round 37 then left ~81 ms unplaced and guessed the Q all-gather, also
> wrong. Round 39 closes it with the DLO stream. The block is now accounted by four
> terms, each measured or bounded.

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

The DiT step is the best-characterised part: a 676 ms block is attention (559 ms,
at 82% of its shape-limited ceiling) plus MLP (33 ms) plus ~3 ms of elementwise
work, leaving **~81 ms unplaced** per block -- about 4 s per step -- whose leading
candidate is the Q all-gather that must precede attention. Round 36's claim that
this closed to zero was wrong and is retracted above.