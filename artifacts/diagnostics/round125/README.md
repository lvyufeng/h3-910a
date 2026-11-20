# Round 125 — restoring acceleration on top of the R124 precision baseline

Baseline: commit `f3f1328`, the R124 real 49-step original-50-layer 910A generation
(own FP16 encoder, original FP32 VAE/branch parameters, verified FP16 range protections).
The user's instruction for this round: this version is the precision baseline, restore
every previously removed acceleration, keep what is lossless, stop and ask for anything
measurably lossy. A deviation inside the earlier 1% GPU-comparison tolerance is **not**
auto-approved.

Everything here is a same-input single-evaluation comparison plus wall-clock timing. No
lossy candidate was enabled in a generation, and no full 49-step run was made in this round.

## The bar had to be re-derived: this device is not bit-reproducible

Re-running the *identical* R124 graph twice gives different bytes:

| step | video relRMS (repeat A) | video relRMS (repeat B) | audio relRMS (repeat A) | audio relRMS (repeat B) |
|---:|---:|---:|---:|---:|
| 0 | 0.080% | 0.103% | 0.161% | 0.217% |
| 10 | 0.052% | 0.055% | 0.074% | 0.104% |
| 25 | 0.027% | 0.026% | 0.034% | 0.041% |
| 40 | 0.032% | 0.032% | 0.055% | 0.057% |
| 48 | 0.053% | 0.054% | 0.166% | 0.134% |

`noise_floor125.json`. Consequence: bit identity cannot gate an optimization here, so a
candidate is called lossless only when it stays inside this measured self-repeat band.

This also reframes R124 honestly: its five same-input GPU gates (0.036%–0.26% video,
0.056%–0.26% audio) sit **inside** this band. They demonstrate the absence of a gross
implementation error; they are not evidence of operator-level bitwise agreement.

## Candidates measured

| candidate | class | result |
|---|---|---|
| whole DiT resident (drop per-block CPU staging) | unavailable | NPU OOM: 31.06 GiB allocated, 404 MiB request failed on all 8 ranks |
| static device tensor cache | lossless, no gain | 41.99 s median vs 40.47 / 43.84 s baseline; 11 keys, 21.3 MB |
| drop the FP32 patch/time projection guards | lossy → asked | video relRMS **2.29%**, ~22× the noise band |
| region cost attribution (one warm step) | measurement | attention **58.1%** of the step; 0.694 s/block vs 0.057 s/block MLP |

Residency is blocked by arithmetic, not by tuning: TP2 leaves 16 539 411 712 FP16
parameter elements (30.8 GiB) per rank, which cannot share a 32 GiB card with
activations. Making it resident needs TP8/USP8 — a reduction-order change, therefore lossy.

## Why the remaining lossless set is small

The step is not transfer- or elementwise-bound, it is attention-bound (12× the MLP per
block). Residency, tensor caching, geometry memoization and pointwise fusion therefore
cannot move 40 s/step toward the 15 s/step target. The lossless items the reviewers did
identify are real but marginal:

- R124-native vision positional-interpolation / rotary geometry memoization
- R124-native text RoPE table memoization with fail-closed keys
- VAE geometry-only caches (token ids, tabulated RoPE cos/sin, tile and temporal-frame plans)
- VAE mean/std and blend-weight caching where the original tensors are reused unchanged
- fixed-shape tile gather workspace removing the `all_gather_var_shape` shape-to-list sync
- static Half layout pre-packing of the original parameters (layout only, identical values)

## What needs a decision

See `STATE125.json` → `lossy_candidates_waiting_for_user_decision`: custom 910A TBE flash
attention (round63 measured 293 102 differing FP16 words), the VAE Half-parameter + fused
decoder fast path, an FP16 residual stream, raised sharding, and fused pointwise kernels.
Round 63/64/76/108/118 failure receipts are preserved and are not reused as correctness or
performance baselines.

## Not claimed

No speed target was met; no full generation was produced this round; no A2/A3 comparison
was measured; no universal picture or audio quality claim is made.