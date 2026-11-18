# Round87 vision — portable FP32 bias + ONE Half cast GEMM (source only)

This directory contains **source only**. Nothing here was compiled, launched or
executed by its author: no SSH, no NPU/GPU work, no model run, no CANN build, no
installation, no production edit and no commit. It proves no precision and no
speed. It is not a full-encoder result and not an official-lossless claim.

## What it is

[`vision_bias87.py`](vision_bias87.py) builds a cached direct-TIK GEMM for the H3
vision tower whose epilogue is the arithmetic that Round83 **actually executed**:

```
Half A (K1, Mp, 16)  x  Half W (K1, N, 16)
   -> one FP32 L0C accumulator live across ALL K blocks   (no Half partial)
   -> tensor_mov to FP32 UB                               (no narrowing)
   -> FP32 UB bias add (16 values broadcast over mb rows)
   -> exactly ONE h_cast(dst, src, 'none')                (single rounding)
   -> Half C (N1, Mp, 16) GM store
```

Five GM pointers, in the order the already-proven launcher bridge expects:

| # | role | dtype | shape |
|---|---|---|---|
| 0 | `a_gm` | float16 | `(K1, M_pad, 16)` activation, zero padded M rows |
| 1 | `b_gm` | float16 | `(K1, N_pad, 16)` weight, values untouched |
| 2 | `bias_gm` | **float32** | `(N_pad,)` exact original Half bias, zero for added N |
| 3 | `aux_gm` | float16 | `(16,)` unused placeholder keeping the proven 5-GM ABI |
| 4 | `c_gm` | float16 | `(N1, M_pad, 16)` the single Half cast |

`matmul` is always called with `bias=None`: TIK's hardware bias operand is
rejected on short-SoC Ascend910, so it is never used and never faked.

## Planned real vision geometries (all 30 cores)

| case | M | K | N | M_pad | K tiles | tasks | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 8064 | 1152 | 3456 | 8064 | 9 | 1701 | 57 |
| proj | 8064 | 1152 | 1152 | 8064 | 9 | 567 | 19 |
| mlp1 | 8064 | 1152 | **4352** | 8064 | 9 | 2142 | 72 |
| mlp2 | 8064 | **4352** | 1152 | 8064 | 34 | 567 | 19 |
| merger1 | 2016 | 4608 | 4608 | 2048 | 36 | 576 | 20 |
| merger2 | 2016 | 4608 | 5120 | 2048 | 36 | 640 | 22 |

The real checkpoint attention/MLP widths used here are the **vision** tower's:
hidden 1152, vision MLP intermediate **4304 → 4352** (host zero padding), merger
4608 → 4608/5120, N rows 8064 (pre-merge) / 2016 (post-merge). The 27648/3456
figures belong to a different projection and are rejected by the padding guards.

## Host contract owned by the root adapter (never by this kernel)

* **M** — zero A rows to a multiple of 128. Those padded output rows are **the
  bias, not zero**; the host must crop them before the model sees them.
* **K** — zero A columns *and* zero W K-rows to a multiple of 128
  (vision MLP `4304 → 4352`).
* **N** — zero W trailing rows *and* zero bias entries to a multiple of 128
  (vision MLP `4304 → 4352`); crop the output columns.
* **bias** — the original model bias is Half; `pack_bias` promotes it to Float32
  exactly (Half→Float is lossless) and the added N entries are zero.
* **C store stride** — `M_pad - Mb` in 32-byte units, because a Half `K0=16` row
  is 32 B. This is deliberately *not* the FP32-output kernel's
  `2 * (M_pad - Mb)`; using the FP32 factor here would corrupt the Half output.

The module exposes torch-lazy helpers for all of the above (`pack_a`, `pack_w`,
`pack_bias`, `pad_k_activation`, `pad_n_weight`, `pad_n_bias`, `unpack_c`) and
torch-free index maps (`pack_a_index`, `pack_w_index`, `c_index`) so the layout
can be cross-checked without a device.

## Cache contract

`build_kernel(m, k, n, out_dir)` validates the plan and the on-device capability
report, verifies every locatable Round83 reference source against its pinned
SHA-256, and then:

* reuses an existing `obj`+`meta` pair only after re-checking `blockDim`;
* serialises build+reuse with `fcntl.flock` on `.vision87.compile.lock`;
* compiles inside a unique `vision_bias87_build_*` workspace;
* **refuses to overwrite** an existing kernel output directory;
* copies (`shutil.copyfile`) the artefacts out and retains a failed build's
  workspace for diagnosis instead of deleting or moving computed paths;
* writes the informational `spec87.json` atomically, in place, and only when the
  recorded object hash disagrees.

Kernel names are `vision_bias87_m{M}_k{K}_n{N}_mb128_nb128_kb128_c{cores}`, so
they can never collide with the Round83 `blocked_*83_*` namespaces.

## Reference pins (re-verified bytes on disk)

| reference | sha256 |
|---|---|
| `blocked_fp32_gemm83.py` (reviewed base) | `78896ada…4c8c2ee` |
| `blocked_bias_epilogue83.py` (reused epilogue) | `2f6a4470…1c8306` |
| `tik_bias_epilogue83.py` (**actually executed** bias micro) | `124ba029…ee03a` |
| `runtime5gm83.py` (bridge used unchanged) | `c8890dbc…5c52a6` |
| launcher `/tmp/libfa910a.so` (unchanged) | `bd50d700…e8b043` |

A present reference copy with different bytes fails closed; a missing copy is
reported (`missing`) but does not silently claim verification.

## Tests

`python3 artifacts/diagnostics/round87/vision/test_vision_bias87.py` → **38 tests,
OK**, stdlib only, no torch/CANN/NPU, ~0.3 s. They cover: module-scope import
purity and lazy TIK/torch, the six real geometries (M_pad/K-tiles/tasks/steps/30
cores), MLP `4304→4352` padding and refusal of unpadded K/N, multi-K plans
(2/9/34/36 tiles), DMA strides (including the Half output stride being *not*
doubled), UB/L1/L0C budget, layout index bijectivity, emission order (`tensor_mov`
→ FP32 `vec_add` → exactly one `h_cast`), `bias=None`, static K loop with
`init_l1out=(kt==0)`, single L0C tensor, explicit 5-GM `BuildCCE` arguments, the
cache/lock/no-overwrite contract, reference pin hashes, tamper detection, and the
manifest's self consistency.

These tests do **not** build or run a kernel: they are structural contracts, not
numerics. Actual execution receipts belong to the root agent's run.

## Status

`executed: False` · `proves_precision: False` · `proves_speed: False` ·
`full_encoder_precision_accepted: False` · `whole_goal_complete: False`.