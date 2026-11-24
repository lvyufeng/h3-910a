# R219 — R218 trim candidate passes step0/48 real-input gates and finite edge tests

**Supplemental precision evidence only. No adoption, composition, candidate-speed claim or full49-step acceptance.**
The model always receives original FA outputs in this round. The existing candidate remains default OFF and awaiting the user's per-item decision. The original graph/50blocks/TP2+USP4/FP16 matrices/FP32 exceptions/155 strict Half checks are unchanged. Live encoder TP8 is the contract, but this isolated DiT replay does not execute the encoder or generate media.

## 1. Real layer Q/K/V gates at the start and end of sampling

The unchanged R218 padding-task candidate was recompiled from the same pinned wrapper/source and produced the **same binary SHA**:

```text
052c49f2df2235bac2136b4630a05a71bd1b9b454d88c354bbbf5e254ab6e327
trim218_q41792_b1_h7_n43776_d128_br128_bc304_c30_k41769_v3__kernel0
```

Storage stays B1H7,N43776,D128; source Q41792, valid keys41769, Br128/Bc304,30cores. The only logical schedule is Tr327/Tc138 vs original342/144; original masks/full partly-valid tiles and all actual token pairs remain. This is not causal/sparse/approximate attention.

| Sampling position | Actual ranks ×layers compared | Compared FP16 output words | Different words |
|---|---:|---:|---:|
| step0 |8×50=400|14,978,252,800|0|
| step48 |8×50=400|14,978,252,800|0|
| R218 step25 (prior evidence) |8×50=400|14,978,252,800|0|
| **Combined gate coverage** |**1,200 original/candidate pairs**|**44,934,758,400**|**0**|

Every candidate ACL call is counted, not inferred from backend selection. Each comparison uses identical **actual Q/K/V in the same call**. The gate always returns original output to the model, including on failure; a mismatch would disable subsequent candidate calls and stop promotion across all ranks. Therefore these tests do **not** depend on a self-repeat noise tolerance or change the trajectory.

- Gate forwards preserve155strict checks/rank; all counts zero for nonfinite/saturation.
- Candidate result returned to model: **0**.
- Each rank really uses TP2 and USP4; all original50blocks are replayed.
- Baseline and gate wrappers are restored and distributed groups closed; final process exit0.

[All gates and counts](gate_summary219.json) · [Rank0 raw receipt](rank_receipts/candidate_rank0.json) · [Fail-closed gate source](gate_stage219.py) · [Process/source pins](process219.json).

These positions strengthen exact local equivalence but **do not prove all49steps, all inputs or combined runtime state**. A separate candidate49step trajectory run, user gate decision and full pipeline scope remain outstanding.

## 2. Extreme finite values and mask boundary tests

After the eight-rank process ended, standalone FA tests used N2048,Br128/Bc256 and valid-key counts1792(exact block boundary) /1791(partial boundary), so one completely invalid key block is omitted. All original GM shapes and strides are unchanged.

Four finite test patterns ×two mask boundaries = eight tests,1,835,008 output words each:

1. Random valid Q/K/V; invalid K=+65504 and V=−65504 (finite input, retained-mask behavior checked even when invalid-column dot products can overflow).
2. Zero Q/K with alternating positive/negative zero V.
3. Alternating smallest positive/negative Half subnormal V.
4. Q=32,K=−32 with alternating-sign V, testing very negative finite scores/exponent handling.

**14,680,064 output words compared: zero differences**, zero nonfinite words and zero saturated output words. The signed-zero output bit patterns also agree. Maximal invalid K/V reproduce the random valid-only output hashes, showing exact mask exclusion for these cases.

No nonfinite input injection was used: the original strict QKV check would fail before attention. These are representative finite-edge tests, not an exhaustive proof over all IEEE bit combinations.

[Edge-case comparisons](edge_summary219.json) · [Raw candidate result](edge_trim219.json) · [Standalone test source](edge_trim219.py).

## 3. Performance and decision boundary

This round **does not remeasure candidate performance**. R218's single-candidate result remains:

- Local FA kernel152.235→140.853ms (1.0808×), restored152.229ms.
- Independent8rank rank-max DiT replay17.944→17.376s, restored17.977s.
- Measured saving0.568s/step (3.17% time reduction) with all155strict checks/FP32guards.

These timings are isolated replay, not live encoder/full49step pipeline. Do not combine them arithmetically with R185's different-run15.473s as if composition were measured. R218's complete prediction scatter is still reported; native self-repeat is not a numeric tolerance. This round feeds no candidate outputs, so it neither removes nor adjudicates that separate full-model issue.

**State: Awaiting user decision, default OFF.** The useful next choice is whether the user accepts these exact same-input FA gates as the local criterion and authorizes **this one candidate alone** for a full49-step verification. No permission for combining candidates or relaxing full-model tolerance is implied.

[R218 speed and numeric evidence](../round218/RESULTS.md) · [Prior full-prediction scatter](../round218/trim_prediction_metrics218.json).

## 4. Scope, failures and cleanup

- Source pins: transformer5d94ba09…, native FA937671ef…, candidate wrapper52365cbf…, fwd a21c52ea…, utils6aa8949d…. Exact step0/48 snapshot hashes are checked on host before launch.
- Only original checkpoints and sealed on-host conditioning/snapshots reused; no new weights or activations downloaded/transferred to the client.
- R219 generator, launch, model gates and edge tests all completed without a runtime/code failure. Static contracts check that `_attention` returns only `reference` and there is no candidate-measurement mode, but are explicitly not numerical evidence.
- Private model/test processes gone; all host-only prediction/edge tensor files, private candidate/cache and package removed. Original sources unchanged; no driver/device/SDK reset, CI/dormant use or unrelated process signaling.
- Owned ts-217 lease released (`released=true`); CUDA grants released unused. Another session's dormant host lease was not touched.

[Cleanup](cleanup219.json) · [Static scope contracts](static_gate_contracts219.json) · [Issue #5](https://github.com/lvyufeng/h3-910a/issues/5).

The 10s/step, encoder<2s and hardware A2/A3 objectives remain open; videoVAE work remains user-paused. Goal stays active.
