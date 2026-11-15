# Round 49 — 30-core hardware trace and real FL2VA workload repair

## Hardware counters: the Cube is not busy for the entire kernel

[Raw hardware CSV](<pipe_profile/kernel_details.csv>) contains exactly three
completed launches of the deployed 30-core object, all at B=1/H=7/N=63232/D=128,
Br=128/Bc=304. Allocation, registration, and warmup are outside profiling.
Unprofiled time is 317.024 ms; profiled mean is **316.952 ms**.

| Hardware pipeline | Reported activity time | Reported cycle ratio |
|---|---:|---:|
| Vector | 173.175 ms | **55.1%** |
| Scalar | 70.254 ms | **22.4%** |
| Cube/MAC | 71.795 ms | **22.9%** |
| MTE1 | 52.301 ms | 16.6–16.7% |
| MTE2 | 35.691 ms | 11.4% |
| MTE3 | 30.558 ms | 9.7% |

These pipelines overlap. Their ratios are **not additive stage-time shares**, nor
proof of a new roofline. The export also has a field named `cube_utilization(%)`
at 99.1%; that must not be mistaken for MAC duty cycle when `mac_ratio` separately
reports 0.229. High card/AI Core activity does not establish high arithmetic duty.

The generic profiler stop-in-RECORD warning was emitted. Completeness was checked
for the measured part: [task records](<pipe_profile/task_time.csv>) contain all
three starts/stops, full ~317 ms durations, and matching task IDs 28/29/30.
The measured versions are torch 2.10.0+cpu and torch_npu 2.10.0.post7, as saved in
[invocation metadata](<pipe_profile/invocation.json>); these current runtime facts
supersede older package-version narration. No package/driver upgrade was done.

Reproducer: [profile_fa30.py](<profile_fa30.py>), with Level1/PipeUtilization and
exactly the installed profiler API. Summary: [pipe_summary.json](<pipe_summary.json>).

## One trace-guided controlled experiment

The kernel broadcasts an FP32 scale with 128 scalar loads/duplicates in each
noninitial K/V tile. A high-level `h_mul(matrix, matrix, column)` alternative was
rejected at TIK validation (operands must have the same shape). This API rejection
is **not** a hardware impossibility result.

A private exact-math variant then statically unrolled only that scalar broadcast:

| Variant | Order 1 median | Reversed-order median |
|---|---:|---:|
| Deployed 30-core baseline | 316.637 ms | 316.621 ms |
| Static broadcast unroll | 313.201 ms | 313.196 ms |

The output is **bit-exact**, max absolute difference 0. The gain is **1.083%**,
not the tens of percent suggested by naively attributing the full scalar ratio to
this loop. The object grows from 36,808 to 48,832 bytes. It is **not deployed**;
further shapes, cache identity, and integration would need validation before a
production change. The next FA work should target genuine vector/Cube/MTE overlap
or substantial softmax work, not claim that loop unrolling solves the bottleneck.

Evidence: [controlled A/B](<broadcast_unroll_result.json>),
[reproducer](<broadcast_unroll_ab.py>), [rejected API probe](<broadcast_probe.py>).
Production FA still uses the runtime-budgeted 30-core version from commit b495919.

## The old form was not a 5-second FL2VA benchmark

The complete remote MP4 from the earlier successful request has actual track
metadata **1344×768, 209 frames, 24 fps, 8.7083 seconds**, and an audio track of
8.732 seconds. An interrupted local media copy was correctly rejected as truncated;
the parser ran on the complete remote file. See [metadata](<old_video_metadata.json>).

The inspected FastAPI form accepts **`seconds`**, not `duration`. Model processing
uses 209 frames as the missing-duration fallback for non-ref2va tasks. Real FL2VA
also requires image conditioning; a text-only request is not a FL2VA validation
merely because the engine was started with `--task-type fl2va`.

For an explicit five-second request at 24 fps, the model's native **17n+5** rule
aligns 120 requested frames to **124 frames, 5.1667 s**. Neither 209-frame timings
nor a four-step distilled run can be directly labeled the official 49-step lossless
5s result. Four source-contract regressions pass; see
[test_workload_contract.py](<test_workload_contract.py>),
[form signature](<source/helpers.py#L852-L890>), and
[frame planner](<source/time_request.py#L10-L23>).

## Corrected request found a real encoder indexing defect

The first corrected 768p/5s form supplied two deterministic first/last reference
images. It failed before DiT, during Qwen encoding, with AICPU `IndexPutV2` exception
0x2a surfaced at `NonZero`/Boolean input masking. That request was stopped and is
**not a performance result**. Receipt: [failed run](<e2e5s01/result.json>).

The inspected encoder had three dynamic Boolean assignments in its input setup.
They are replaced by exactly equivalent `masked_fill_` operations: TP vocabulary
input masking, image token type, and video token type. The token override order,
dtypes, weights, precision, collectives, and media conditioning are unchanged.
A hash-checked deployment preserves the original encoder in an exclusive backup.

**Nine CPU/NPU regressions pass**, including 8-way shard reconstruction, all token
boundaries, invalid/empty inputs, and NPU int64/int32 mask operations. A small
[portable patch](<../../design/patches/encoder_masked_fill.patch>),
[exact candidate](<candidate_encoder.py>), [tests](<test_embedding_mask.py>),
[test receipt](<mask_tests_result.json>), and [deployment receipt](<deployment.json>)
are provided. The failed first run did not identify which earlier asynchronous
Boolean assignment first poisoned the stream; all three dispatch that same dynamic
indexing family and are tested independently after repair.

A fresh normal-engine request after this repair used
[e2e_5s_probe.py](<e2e_5s_probe.py>), kept existing model precision/schedule, set
`seconds=5`, size 1344×768 and fps=24, and supplied real first/last images.
It passed the former masking point but failed in the visual encoder's **BF16
LayerNorm**: the installed TBE implementation rejects its bfloat16-to-float32
cast. Receipt: [second run](<e2e5s02/result.json>). The model never reached DiT, so
this is **not** a new end-to-end performance result and no final MP4 was produced.
The requested pre-generation metadata (120 frames/5.0 s) in the failed response is
not evidence of a completed video; the native shape rule remains 124/5.1667 s.

This is a concrete implementation compatibility defect, not an impossibility
statement about the chip. The next round should validate an exact BF16-compatible
LayerNorm route (including input/output cast behavior) against the CPU/reference
norm before repeating the verified request. Silently switching encoder precision
would not establish an official lossless comparison. Source inspection also shows
that the encoder's default dtype is BF16 unless its explicit TE_DTYPE flag is set;
`--dtype half` on the engine does not itself prove FP16 encoder execution.

The whole optimization goal remains active; no 15 s/step, 2 s encoder, 20 s VAE,
or official-equivalent completion claim is made. The 30-core FA fix and the tested
three masking substitutions remain deployed, with their originals backed up.
The 1.08% broadcast variant remains private and undeployed. Managed engines were
stopped after each failed request; this round releases the fleet lease after
verifying that all eight devices have no remaining processes.
