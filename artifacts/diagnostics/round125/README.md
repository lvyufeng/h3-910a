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

Residency is blocked by arithmetic, not by tuning: every rank holds the full
16 539 411 712 FP16 parameter elements (30.8 GiB), which cannot share a 32 GiB card with
activations. Raising tensor parallelism does **not** help, because this DiT is not actually
TP-sharded: per-rank parameter bytes are identical at TP2 and TP8, so the weights are
replicated on all eight ranks and only sequence parallelism splits activations. That also
explains the low compute utilisation early in the round.

## Why the remaining lossless set is small

Before the flash-attention kernel the step was attention-bound: attention took 58% of it
and cost 12x the MLP per block, which is why residency, tensor caching, geometry
memoization and pointwise fusion could not move 40 s/step. After the kernel that ranking
inverts: attention falls by about 2.7x, and the remaining cost is per-block weight staging
through the CPU offload plus one-off pieces (token refiner, time embedder, final layer),
which is why the pointwise and residual candidates measured later gained nothing. The
lossless items the reviewers did identify are real but marginal either way:

- R124-native vision positional-interpolation / rotary geometry memoization
- R124-native text RoPE table memoization with fail-closed keys
- VAE geometry-only caches (token ids, tabulated RoPE cos/sin, tile and temporal-frame plans)
- VAE mean/std and blend-weight caching where the original tensors are reused unchanged
- fixed-shape tile gather workspace removing the `all_gather_var_shape` shape-to-list sync
- static Half layout pre-packing of the original parameters (layout only, identical values)

## Decision record

The user approved measuring all five lossy candidates, with the explicit rule that every
item reports its own deviation and timing and the user decides — an error inside the
earlier 1% GPU-comparison tolerance is not auto-approved. Only the flash-attention kernel
was then kept. Round 63/64/76/108/118 failure receipts are preserved and are not reused as
correctness or performance baselines.

## Not claimed

No speed target was met (18.41 s/step against a 15 s target). The encoder target was not
worked on at all (about 65 s against 2 s), and the VAE decode target is still about 132 s
against 20 s — that was measured, including a Half-parameter fast path that did not help,
but never brought near the target. No A2/A3 comparison was measured. No universal picture or audio
quality claim is made: the audio track stays at about -75 dBFS in every configuration, so
sample-level audio differences are never used as a quality signal. All three rejected
candidates were measured only; none of them is enabled anywhere.

## Restored acceleration: custom 910A flash attention (user-approved)

The R124 install already routes `TORCH_SDPA` to the chunked matmul+softmax backend;
the fused TBE kernel sits behind `VLLM_OMNI_910A_FLASH_ATTN=1` and, for this shape,
reuses `b1_h7_n43776_d128_br128_bc304_c30_k41769_v3`. Engagement is proven by the
`[FA910A] plan ... B=1 H=7 N=43776 D=128 Br=128 Bc=304 core_budget=30` line.

| measurement | R124 baseline | with FA |
|---|---:|---:|
| engine warm step (H3PERF) | 38.33 s | **17.76 s** (first 169.16 s) |
| replay harness warm step | 40.47 s | **18.41 s** |
| 49-step generation wall | 2316.96 s | **1329.96 s** |
| end-to-end | 1.00x | **1.74x** |

Same-input deviations stayed at or near the device's own run-to-run noise band
(video 0.027%-0.116%, audio 0.040%-0.291%). Over the full trajectory the generated
media does not drift away from the baseline:

- 124/124 frames at **PSNR >= 51.18 dB** (median 54.98 dB), mean absolute pixel
  difference 0.104 of one 8-bit level; no frame below 40 dB.
- Both audio tracks remain inaudible noise near **-75 dBFS** (peak 1.86e-4 vs
  1.75e-4), so sample-level audio correlation is not used as a quality claim.

Artifacts: [video](<artifacts/diagnostics/round125/NPU_generated_FA/h3_910a_vllm_omni_fp16.mp4>),
[audio](<artifacts/diagnostics/round125/NPU_generated_FA/h3_910a_vllm_omni_audio.wav>),
[contact sheet](<artifacts/diagnostics/round125/NPU_generated_FA/contact_sheet.png>),
[comparison](<artifacts/diagnostics/round125/R124_vs_FA_media125.json>),
[kernel evidence](<artifacts/diagnostics/round125/fa910a_candidate125.json>).

Two harness failures were preserved rather than reinterpreted: attempt 124h died in
my media-capture path assertion after a complete 49-step DiT and VAE decode, and
attempt 124i died because the private IPC key lived outside `/tmp/h3_align12401`, so
`configure_python_startup`'s assertion was swallowed at interpreter startup and the
worker audit hook never installed. Both were harness errors, not model failures.

## Candidate verdicts (all five approved candidates measured)

Every candidate was measured against the accepted FA base with identical inputs;
the user judges the numbers, nothing is auto-accepted. Full detail in
[candidates_measured125.json](<artifacts/diagnostics/round125/candidates_measured125.json>).

| candidate | speed | numerics | verdict |
|---|---|---|---|
| custom 910A TBE flash attention | step 40.47 -> **18.41 s**, wall 2316.96 -> **1329.96 s** | media PSNR >= 51.18 dB | **kept** (user-approved) |
| VAE Half params + native fused decoder | decode 138.6 -> **132.4 s**, wall unchanged | media PSNR >= 51.68 dB | rejected: no real gain |
| DiT FP16 residual stream | 18.41 -> **25.93 s** | same as FA base | rejected: 40% slower |
| fused FP16 pointwise (SwiGLU / SiluAndMul) | 18.41 -> **18.83 s** | video 0.08-0.22%, audio to 1.64% at step 48, growing with step | rejected: no gain and drifting |
| raised TP/USP for DiT residency | residency OOMs; USP8 slower (22.64 s) | — | unavailable: TP does not shard this DiT |

Two negative results are worth keeping in mind:

- The VAE decode is **not** parameter-precision bound. Rewriting the original FP32
  decoder parameters to Half and enabling every fused decoder path moved ~132 s by
  at most 4%, so that cost lives in tile staging / temporal streaming / audio.
- The FP32 residual stream costs **no** speed. Narrowing it to FP16 was slower, so the
  R124 numerical protection is not a performance liability.

## Still outstanding

Encoder (~65 s against a 2 s target), VAE decode (~132 s against 20 s) and the A2/A3
comparison remain open. The DiT step is now dominated by per-block weight staging: with
attention fixed, pointwise fusion and residual narrowing both did nothing, because the
cost is streaming 50 blocks of weights through a CPU offload that cannot be dropped
(the DiT does not fit 32 GiB and tensor parallelism does not shard it). Closing the
remaining gap therefore needs a different mechanism — weight pre-packing or a different
staging strategy, quantized transfer, or a larger-memory part — not another precision
approximation.

Also unmeasured: the AdaLN / QK-norm+RoPE fusions specifically (the highest-volume
pointwise fusion, SwiGLU, was measured and gained nothing), and encoder acceleration.

## Files in this directory

- `check_NPU_candidate125.py`, `r125_*.py` — the same-input candidate harness and the
  variant patches (`half_residual`, `fused_mlp`, `naive`).
- `launch_candidate125.py` — spawns the pinned R124 driver with a fresh private IPC key,
  work directory and output label for candidate full generations.
- `noise_floor125.json`, `baseline_rank0*.json` — the device self-repeat band.
- `resident_oom_rank0.json`, `cost_attribution125.json` — residency limit and the
  attention-dominance profile.
- `fa910a_candidate125.json`, `full_generation_with_FA125.json`,
  `R124_vs_FA_media125.json` — the accepted acceleration, end to end.
- `half_residual_candidate125.json`, `VAEfast_vs_baselines125.json`,
  `fused_mlp_fa_rank0.json`, `candidates_measured125.json` — the rejected candidates.
- `encoder_inventory125.json`, `vae_inventory125.json` — source inventories produced by
  two read-only reviewers.
- `NPU_generated_FA/` — the accepted FA generation and its contact sheet.
- `NPU_generated_VAEfast/` — the VAE fast-path generation used for the comparison.

## Reproducing the accepted configuration

The accepted configuration is the pinned R124 driver plus one environment variable:

```bash
VLLM_OMNI_910A_FLASH_ATTN=1
```

Everything else is the R124 reproduction recipe (see
[round124 README](<artifacts/diagnostics/round124/README.md>)): `PYTHONPATH=/tmp/h3_align12401`,
`H3_NPU_ALIGNMENT124_SELECTIVE_BOOTSTRAP=1`, a fresh 32-byte `0600` private IPC key **inside
`/tmp/h3_align12401`** (the bootstrap asserts that parent), a fresh `ASCEND_WORK_PATH`, a
fresh `H3_NPU_RAW_OUTPUT_DIR124` whose parent is `/tmp/h3_align12401` and whose name starts
with `NPU_full_generation124`, and a new `--output-label` (the driver refuses to overwrite).

Two failure modes cost real time here and are worth knowing:

- The IPC key must live under `/tmp/h3_align12401`. If it does not, the assertion inside
  `configure_python_startup` is swallowed at interpreter startup, no worker audit file is
  written, and the driver fails later on its `49`-evaluations-per-worker assertion.
- The raw output directory must satisfy the capture hook's path assertion, otherwise the
  run dies *after* a complete 49-step DiT and VAE decode.

`launch_candidate125.py` wraps exactly this for candidate full generations.
