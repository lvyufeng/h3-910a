# R126 — actual warm execution and corrected acceleration evidence

This round keeps the user-accepted R124 + FA precision baseline. Experiments are
private, on owned ts-217, with the already-present original checkpoint. No model
weights are transferred to the client. Only source/JSON and final small media
are retained locally; an interrupted 3.78MB prediction transfer was removed and
never used as model input. A fresh real 49-step combined regression has completed
and its worker/media evidence is verified. This is **not** automatic acceptance
of a new precision policy, equivalence of independent live trajectories, or
completion of the encoder <2s / VAE <20s / DiT <15s targets.

## Corrections to R125

- **TP really shards the DiT.** Live vLLM groups and actual tensor shapes were
  checked on every rank for TP2/4/8. Parameter memory is 30.890 / 15.446 / 7.723
  GiB per rank; QKV row sizes are 10752 / 5376 / 2688, with 28 / 14 / 7 heads.
  The old TP8 failure was not evidence of replication. See the
  [eight-rank meta evidence](<true_TP_meta_summary126.json>) and actual
  [TP4 forward receipt](<resident_TP4_FA126_rank0.json>).
- **The old VAE "fast" switch did not invoke a NPU fusion.** The correct
  `ops.vae.dispatch` capability lookup returns None, the actual installer returns
  False, and the original FP32 parameters remain FP32. See the
  [runtime source facts](<runtime_extensions126.json>) and
  [same-latent native VAE receipts](<vae_observe_native126/rank0.json>).
- **The old Half-residual patch did not force a Half residual**, and
  `SiluAndMul.forward_npu` defaults to the eager fallback on this host. The old
  measurements cannot reject a genuine Half stream or a genuine fused kernel.
- **A cold, per-module-synchronized profile cannot identify the warm bottleneck.**
  Native immutable pinned staging and prefetch already exist. New measurements
  use explicit warmups, rank-max elapsed time, and a separate event-only profile.

## 910A interconnect constraint

**Tensor parallelism remains ON.** The accepted/current full-generation baseline
is **DiT TP2 + USP4 and encoder TP8**, using all eight devices. Rejection of a
slower TP4/TP8 *candidate* does not mean disabling TP or switching to TP1. Live
TP groups and actual sharded tensors were verified, not inferred from a switch.

The user explicitly notes the **2D full-mesh layout: two four-card full-mesh
islands, with communication between the islands going through CPU**. The
[actual `npu-smi` query](<topology_query126.json>) confirms islands **0–3** and
**4–7**. Cross-island pairs 0↔4, 1↔5, 2↔6 and 3↔7 are PHB; other cross-island
pairs are SYS paths through PCIe/NUMA/SMP. Do not model this host as a single
eight-card full mesh. Map physical card IDs before
choosing groups; prioritize high-frequency TP collectives within an island and
measure unavoidable cross-island USP/HCCL traffic separately. TP8 parameter
residency is not a reason to push every block's all-reduce through the CPU bridge.
The earlier TP2/USP4, TP4/USP2 and TP8/USP1 timings are observations, not a pure
causal topology comparison: their GEMM and collective shapes differ too.

## Executed warm experiments

- TP2/USP4 native staging is about 17.9–18.0s per representative step. True TP4
  residency is about 18.98s and TP8 residency about 21.02s; neither is faster.
  All 155 Half safety checks remain enabled. See
  [TP2 paired checks](<sync_native_TP2_FA126_rank0.json>),
  [TP4](<resident_TP4_FA126_rank0.json>) and
  [TP8](<resident_TP8_FA126b_rank0.json>).
- The actual FA path has 50 successful ACL launches per measured forward,
  the expected 41792→43776 sequence plan, valid key prefix 41769, 30 cores, and
  the same pinned kernel binary. The kernel itself is about 7.5s; USP pre/post
  spans total about 1.08s. Nested spans must not be added together. See
  [kernel-level execution evidence](<actual_FA_profile126_rank0.json>).
- Deferring host reads while preserving the checks, then selecting the exact
  power-of-two scale on device, saves only about 0.07s in the first paired run.
  All 8×104 real same-input wide-GEMM leaves match word-for-word. The
  [range-selector NPU unit](<device_range_unit126.json>) also includes all
  threshold boundaries. This is not an automatic whole-trajectory acceptance.
- A two-slot immutable weight pool, explicit compute-complete fences and
  optional eight-block residency are implemented privately. Clean native /
  serialized pool / asynchronous pool / eight-resident-block medians are
  about 18.012 / 19.778 / 18.026 / 17.955s. The gain is small, not evidence that
  staging dominates. All held storage prefixes match their pinned masters. See
  [clean pool evidence](<weight_pool_TP2_FA126_rank0.json>) and the separate
  [eight-rank event profile](<rank_receipts126/weight_pool_profile126/candidate_rank0.json>).
  Serial-copy body overlap is explicitly zero; the independent-stream experiment
  records actual copy timestamps and block-body intervals, not GPU busy-time.
- The equivalent integer Half safety audit has now tested **all 65,536 bit
  patterns on NPU**. Both original checks, their counts, and strict abort
  placement are preserved. Representative warm step 18.059→17.556s; restored
  original is 18.025s. See the [exhaustive unit](<equivalent_safety_unit126.json>)
  and [strict/simplified/restored measurement](<equivalent_safety_bench126_rank0.json>).
  This is a read-only check optimization, not looser range protection.
- Same accepted final latent, native FP32 VAE: video warm median about 20.7s,
  audio about 0.21s; their first cold calls are approximately 78s and 62s.
  Video loading is about 0.64s, outside resident decode timing. Do not compare
  these with the older approximately 132s whole-stage figure as if the scopes
  were identical. See [same-latent timing summary](<vae_summary126.json>).
- A real persistent Half **operand-copy** cache was tested for the 144 decoder
  block Linear modules, while every original FP32 parameter stays unchanged.
  All 1152 first-tile Linear comparisons and all 196 complete prequantized local
  tile pairs are word-identical. Clean video timing is 20.763→20.565s, restored
  native 20.753s; it needs an additional 4.502 GiB and primes in about 0.058s.
  This does not reach <20s and is not a fused kernel or a parameter rewrite. See
  [complete cache evidence](<VAE_operand_cache126c/rank0.json>) and the
  [cache implementation](<vae/autocast_operand_cache126.py>).
- Current live FP16 encoder geometry caching has real hits and both original
  geometry leaves match exactly, but clean 2.340→2.339s is not a useful gain.
  Separate full live-graph captures executed all 27 vision blocks, all three
  DeepStack mergers and all 50 text layers. Vision outputs are exact across two
  native repeats; the text difference starts in layer 0 and grows to about
  3.007% relative RMS at layer 50. This is the **unchanged native path**, not
  an attributable cache loss or an automatic tolerance. See
  [encoder measurement](<encoder_geometry126_rank0.json>) and the
  [complete native-repeat diagnostic](<encoder_native_diagnostic126_rank0.json>).
- After confirming the two physical HCCS islands, live **encoder TP4** was tested
  inside each island. Warm medians are **2.341 / 2.297s**, with **12.825 GiB**
  parameters per rank. All repeated and cross-island TP4 outputs are word-identical,
  but they differ from the fresh live TP8 reference by **2.783% relative RMS**.
  This is neither a useful speed win nor automatic precision equivalence. The
  candidate is stopped pending the user's numerical decision; **encoder TP8 and
  DiT TP2+USP4 remain enabled**. See
  [topology-aware encoder evidence](<mesh_encoder_summary126.json>).

## Full-regression lifecycle and timing scopes

The first integrated run completed all **8×49 DiT forwards and 60,760 strict
Half checks**, then failed before media completion because private Half copies
were created inside the generator's outer `inference_mode`; their version-counter
guard was unsupported. This was a metadata/lifecycle bug, not an observed numeric
delta. The [failed attempt](<NPU_full_generation126_failed_result.json>) is kept.
Copies are now constructed in a narrowly scoped `inference_mode(False)` +
`no_grad`, while model arithmetic still runs under the original outer mode and
all original FP32 parameters remain unchanged. The
[actual NPU nested-mode regression](<topology_encoder/cache_nested_inference_unit126.json>)
passes 144 synthetic exact Linear comparisons. A fresh corrected full49 run
completed; the earlier failed attempt remains labeled failed.

The successful [real generation](<NPU_generated126/generation_result126.json>)
produced **124 frames at 1344×768, 24fps**, with original 50-block DiT execution
on every one of 49 steps, live FP16 encoder, and original FP32 VAE parameters.
All **60,760 strict Half checks** pass and the actual cached-operand call count
is **4032/3024 per rank** (28/21 local tiles ×144 Linears), never just a flag.
Generation wall is **1314.58s**, compared with the previous FA run's **1329.96s**:
only about **15.38s / 1.16%** lower, not a large speedup. Initialization is
separate (**286.24s**). The full cold video+audio decode stage still costs
**142.14s**, including cache priming (up to **2.019s**), staging and first kernels.
Do not replace that cold-stage number with the independent **20.565s warm
resident video-only** measurement. See [full timing summary](<full_generation_summary126.json>).

Against the previous user-accepted FA media, encoded-video PSNR is at least
**51.52dB** across all 124 frames (median **53.89dB**). Native Float audio is not
word-identical: relative RMS difference **1.057**, absolute RMS difference
**1.2195e-5**, maximum absolute difference **6.5571e-5**, correlation **0.435**.
Both audio tracks have RMS around **−98.8dBFS**. These low amplitudes do **not**
authorize discarding the difference or declaring quality equivalence. Each run
recomputes its live encoder, and native repeats already vary; this full-media
comparison is not an isolated causal operand-cache experiment. See the
[unmodified media metrics](<R125_FA_vs_R126_media.json>) and
[generated video](<NPU_generated126/h3_910a_vllm_omni_fp16.mp4>). New precision/
quality acceptance remains the user's decision.

Do not mix the older H3PERF main-block stack timer with the whole-forward hook.
The accepted R125 FA run's **same whole-step hook** has a warm rank-max median
**19.732s**, and the successful R126 run is **19.372s** (first step **87.51s**).
Those are different scopes from the **17.556s isolated same-input check trial**.
See [comparable baseline scope](<R125_FA_same_scope_timing126.json>) and
[successful whole-forward timings](<full_generation_summary126.json>).

## Runtime safety and scope

- Actual FA flag: `VLLM_OMNI_910A_FLASH_ATTN=1` (not `FAST_ATTN`).
- No BF16 roundtrip, no clipping, no forced Half residual, no unverified NPU
  activation fusion, no teacher/student switch, and no hidden/vision-feature
  substitution. Same-input DiT replay fixtures are diagnostics, not proof of
  a new live 49-step trajectory.
- Shared SDK/model sources are read-only. Pool and operand-cache changes are
  private instance hooks; original values, dtype contracts and strict stop
  checks are preserved. Copies and proof/profile forwards are outside clean
  timing. Original parameters are not widened back after a lossy rewrite.
- GPU capture budgets are explicit. Only JSON/source evidence is versioned;
  private IPC material, model tensors and activation archives are ignored.
- VAE ranks own different numbers of tiles: 28 on ranks 0–3 and 21 on ranks
  4–7. Every rank must still enter **28 proof gates**. The first prototype's
  mismatched gate/decoder-barrier sequence was a harness failure, not a numeric
  loss. Its [exact executed source](<vae/executed_cache126b/benchmark_operand_cache126.py>)
  is retained, and the corrected [scheduling contract](<vae/tile_rounds126.py>)
  has portable regressions.
- An unchanged audio decoder has non-bit-identical self repeats. Report the
  actual absolute/relative errors; do not call this equivalence or invent a
  universal audio tolerance. A2/A3 timings are still not measured.

## Validation

The [evidence verifier](<verify_round126_evidence.py>) checks real groups, shapes,
completed rank receipts, safety counts, actual kernel launches, serial overlap
controls, immutable pool storage and complete prequantization comparisons. A pass
means the **evidence** is consistent with execution, not that a speed/precision
candidate has been accepted or that the long-term latency objective is complete.

```bash
python3 artifacts/diagnostics/round126/vae/test_contracts126.py
python3 artifacts/diagnostics/round126/vae/test_tile_rounds126.py
python3 artifacts/diagnostics/round126/encoder/test_encoder126_contracts.py
python3 artifacts/diagnostics/round126/verify_round126_evidence.py --require-final --require-combined
```

The successful combined49 run is fully verified in
[the final verifier result](<verified_round126_final.json>). This verifies real
execution and evidence integrity, not global tensor identity between independent
live trajectories or automatic promotion. Shared accepted runtime remains
unchanged; only the private combined experiment includes the new candidates.
All private model processes have exited, the private IPC key is removed, and
owned ts-217's lease has been released. Accepted source SHAs still match;
[cleanup evidence](<cleanup126.json>) is retained.
