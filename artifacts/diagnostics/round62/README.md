# Round62: actual whole-block AdaLN acceptance and rejection on8×910A

## Main result

Self-developed FP16 point AdaLN/gated-residual kernels now pass a **complete
teacher-forced DiT block** check, not merely local operator comparison.

- Three real layers0/24/49, all eight ranks, each5224×5376 local rows.
- Native replay first matches the actual full-run block captures **bit-for-bit**.
- Point-gate and exact-Norm-plus-point-modulation modes also match all24 fixtures;
  each accepted mode checks **674,021,376 scalar outputs**, with four forwards
  per case repeat-exact.
- The lossless-to-current-FP16 variant improves these warmed resident blocks by
  only about0.6–0.7ms/~0.3%. This is small and **not** a new full-step/E2E result.
- Full fused Norm is faster but **fails lossless acceptance** on every captured
  layer/rank; layer49 reaches relativeRMS0.12346%, max absolute error1840. It was
  not installed in production.

The reference is current native **FP16** H3 on genuine **Ascend910A**, not A2,
original BF16 H3 or the official task/precision/warm-protocol lossless recipe.

## Fixture capture and loader equivalence

[Capture receipt](<e2e_capture/result.json>) is one instrumented FL2VA interval,
1344×768, two first/last references,5s/24fps/seed1234,124 aligned frames. It only
supplies actual fixtures; its~555s request is neither quality proof nor clean
latency evidence. Complete rank-local block inputs/outputs are retained on-host
under `/tmp/h3_round62_blocks01/capture`; the
[24-fixture manifest](<capture_metadata/capture_metadata.json>) pins shapes,
metadata and raw hashes. Each fixture contains all5224 rows, without resampling.

Native [configuration](<source/transformer_config.json>) and
[weight index](<source/transformer_weight_index.json>) are preserved. Standalone
replay loads exactly10 tensors for the requested layer through the **actual native
model loader**: grouped per-head QKV is reordered once, fc1 loads first-half gate
and second-half up, all other parameters use their attached loader. No alternate
FP32 projection, padding, quantization, VSA, LoRA or cache approximation is added.
The full-run output match validates this reconstruction for the three fixtures.

All eight ranks replay together with **USP8/Ulysses8, ring1, TP1**, a full replica
of only one resident block at a time. Captured local rows are not split again.
Global packed metadata stays41792 rows, valid keys41769/padding23; the packed
96-wide DiT RoPE table and two FP32 timestep embeddings are unchanged.

[Attempt01](<attempt01/block_replay_rank0.json>) failed before any accepted A/B:
attention construction lacked a forward context, so the
[parallel factory](<source/attention_parallel_factory.py#L40-L43>) cached
NoParallel; a global41792-key mask then mismatched local5224 rows. The final
[reproducer](<block_replay_tp8.py>) establishes the forward context during **both
construction and execution**, and asserts the constructed strategy is Ulysses.
The failed source and all-rank receipts remain retained, not overwritten.

## Exact and exploratory modes

All modes use the same model, captured inputs, attention backend and weights:

1. `baseline`: original eager indexed operations.
2. `point_gate`: only the final MLP gated residual uses native-half TBE.
3. `lossless_split`: preserve the original FP32 variance/rsqrt/gamma sequence and
   its normalized FP16 rounding, then use separate native-half point modulation
   and gated residual kernels. This is **not** a single fused Norm kernel.
4. `norm_fused_exploratory`: self-developed fused RMSNorm+modulation and gated
   residual+RMSNorm+modulation; FP32 reduction order changes, so no lossless claim.

Actual projection chunk views have row stride32256 and nonzero offsets. The final
point wrapper consumes those views directly. Norm prototypes still require dense
rows, and their exploratory mode explicitly includes tiny table `.contiguous()`
copies; it is not an implicit stride-unsafe launch.

The [all-rank receipts](<block_replay/block_replay02_rank0.json>) and
[verified summary](<summary.json>) contain each mode's raw-bit comparisons,
repeatability and three warm latencies. The summary checks every rank and exact
executed source, not equality of displayed norms/hashes alone.

## Warmed latency: include communication, exclude offload

Take the maximum rank latency per synchronized warm forward, then its median
across three calls. Complete block includes native QKV/norm/RoPE, full-sequence
custom FA, both Ulysses exchanges, projections, MLP and surrounding modulation:

- Layer0: baseline216.215ms; point-gate216.038ms; lossless-split215.554ms;
  exploratory fused Norm214.024ms.
- Layer24: baseline217.044ms; point-gate216.591ms; lossless-split216.349ms;
  exploratory fused Norm214.666ms.
- Layer49: baseline216.312ms; point-gate216.361ms; lossless-split215.722ms;
  exploratory fused Norm214.758ms.

Only three warmed calls, fixed mode order, no statistically robust/sustained
service speedup claim. The point-gate benefit alone is at the timing-noise scale.
**Do not multiply these resident block numbers into a new50-layer/DLO/49-step
performance claim.** No encoder, VAE, model-transfer or request overhead is timed
in this replay.

## Why the faster Norm candidate remains rejected

All exploratory cases repeat exactly on their own; deterministic does not mean
reference-correct. RelativeRMS versus the same-input native block:

- Layer0:0.00786–0.02688%, max absolute error2.
- Layer24:0.01063–0.01427%, max absolute error32.
- Layer49:0.03188–0.12346%, max absolute error1840.

The [round61 operator proof](<../round61/README.md>) had~6e-6 relativeRMS local
modulation differences; actual attention/MLP amplify those differences. This
experiment neither establishes nor excuses a generation-quality equivalence.
The faster Norm path is not a lossless integration success.

## Reproducibility, tests and production restoration

```bash
python3 artifacts/diagnostics/round62/summarize.py
python3 -m unittest discover -s artifacts/diagnostics/round62 -p 'test_*.py' -v
# Existing runtime on ts-217, only after a fresh owned lease:
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
/home/mseco/vllm910a/bin/python -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29663 \
  /tmp/h3_round62_blocks01/block_replay_tp8.py
```

Scripts refuse existing proof files: use a new exact receipt name for a new run.
The replay needs the final
[AdaLN wrapper](<../round61/adaln_vector61.py>),
[point builder](<../round61/point_vector61.py>) and isolated
[C launcher](<../round61/vec61_launcher.c>) under the diagnostic root; the existing
CANN/runtime and shared production FA are reused, not replaced.

[Restoration receipt](<capture_restoration.json>) and
[final source receipt](<final_source_verification.json>) verify unchanged production
DiT/encoder/noncausalFA/sharedNorm/coordinator/rotary. Additional indexed modulation,
QK-norm/RoPE, attention and parallel-factory sources match retained snapshots.
Local final kernel, C launcher, guard and block replay bytes match actual executed
source. Capture-only hooks are no longer imported by production; kernels remain
confined to standalone processes. All jobs are collected, groups shut down,
eight cards were live-probed0%/zero processes and the owned lease was released.

## Remaining objective

No new49-step video, encoder≤2s, decoder≤20s, native-BF16 quality acceptance or
official warmed T2VA-matched comparison. The goal stays active. AdaLN point kernels
now have whole-block identity proof; full Norm fusion and the larger encoder/VAE
and matched generation work remain open.
