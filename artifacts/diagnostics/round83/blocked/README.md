# Round83 — blocked FP16×FP16 → FP32-output GEMM for the real H3 shapes

**Source only, deliberately not compiled and not executed.** The whole point of
this module is to let the root agent build and launch it on the accelerator under
its own supervisor; nothing here is a precision result, a speed result or an H3
improvement claim.

## What it adds over Round82

Round82 proved on this exact device that a direct-TIK kernel can do FP16 operands
→ FP32 L0C → FP32 GM, and really executed it: M128/K512/N128, bit-identical
repeats, 4.081e-7 relative RMS against an FP32 accumulator diagnostic. That
kernel loaded the *whole* A and B into L1, so it only worked for the micro shape.

This module keeps that arithmetic and adds explicit blocked tiling so the H3 row
shapes fit, with the L0C accumulator staying FP32 and alive across **all** K
blocks. There is no Half partial and no FP32-UB partial sum anywhere: the only
narrowing in the whole kernel happens later, outside it, after the FP32 all-reduce.

| shape | logical (M,K,N) | M pad | tiles (M×N) | cores | steps | K blocks |
|---|---|---|---|---|---|---|
| `kv_proj` | 2043, 5120, 128 | 2048 | 16 | 16 | 1 | 40 |
| `q_proj` | 2043, 5120, 1024 | 2048 | 128 | 30 | 5 | 40 |
| `mlp_up` | 2043, 5120, 3456 | 2048 | 432 | 30 | 15 | 40 |
| `row_parallel` | 2043, 3456, 5120 | 2048 | 640 | 30 | 22 | 27 |

## Layouts and the byte arithmetic that must be reviewed

```
A GM: (K1, Mp, 16) float16   (k1, m, k0) = A[m, k1*16+k0]
B GM: (K1, N,  16) float16   (k1, n, k0) = W[n, k1*16+k0]
C GM: (N1, Mp, 16) float32   (n1, m, n0) = C[m, n1*16+n0]
```

A Half `K0=16` row is exactly 32 B, so `data_move`'s 32-byte units mean the A/B
bursts need **no** factor of 2; a Float32 `K0=16` row is 64 B, so the C store
stride **does** need it. That asymmetry is the one thing most likely to be got
wrong by hand, so every stride is asserted against the byte layout in
`test_blocked_fp32_gemm83.DmaUnitTests`:

* A load: `nburst=kb/16`, `burst=mb` (=`mb*32 B`), `src_stride=mp-mb`, `dst_stride=0`
* B load: `nburst=kb/16`, `burst=nb`, `src_stride=n-nb`, `dst_stride=0`
* C store: `nburst=nb/16`, `burst=2*mb` (=`mb*64 B`), `src_stride=0`, `dst_stride=2*(mp-mb)`
* L0C→UB: `tensor_mov(..., 'm', 1, mb*nb*4//1024, 0, 0)` (its burst unit is 1 KiB)

All GM offsets are 32-byte aligned; the tests check that for every tile and K
block of every H3 shape. No access leaves its operand: `Mp` is a multiple of
`mb`, and N/K are tile multiples for all four shapes.

## Deliberate design choices a reviewer should check

1. **`init_l1out` is a build-time `bool`.** The SDK calls
   `check_type_match(self.init_l1out, bool, ...)`, so it cannot be a dynamic
   scalar. The K loop is therefore a *static* Python loop where only `kt == 0`
   initialises L0C (`init_l1out=(kt == 0)`). This is the same arithmetic the task
   described via an `if_scope` duplication, with less IR and no dynamic branch
   around a cube instruction.
2. **One L0C tensor**, allocated once per core outside the tile loop; the tests
   assert a single `scope_cc` allocation and that it precedes the tile loop.
3. **Synchronisation** comes from `new_stmt_scope(disable_sync=False)`, which the
   SDK documents as "automatically insert the sync signal into IR". This is the
   same boundary the deployed TIK flash-attention helper uses around
   `matmul`/`tensor_mov` while reusing the same L1 buffers across K blocks, which
   is the property this kernel needs (WAR between the next load and the previous
   cube read). This is the highest-risk construct that could not be compiled by
   its author; see the staged verification plan below.
4. **Bias is absent on purpose.** The TIK hardware/L1 bias argument was rejected
   on this part in Round82 (`matmul param bias, not support Ascend910`); a sibling
   task owns a portable FP32-UB bias epilogue.
5. **No FP32 weight shadow and no Half partial.** Operands stay the original Half
   values; the accumulator is FP32 in L0C and the destination is FP32 in GM.
6. **30 cores, never more.** The live DCMI audit reports 30 usable cores on all 8
   original 910B chips; the compiler's `Ascend910A` profile says 32, which is an
   older default. 32 GiB of memory is not 32 cores. `capability_report()` refuses
   to build if `CORE_NUM != 30`.

## Known inefficiency (not a claim of performance)

Each tile re-reads a full `mb×K` slice of A and `nb×K` slice of B, so A is read
`n_tiles` times. For `mlp_up` that is ~27 A reads (≈566 MB) and for
`row_parallel` ~40 (≈566 MB), on top of the B traffic; `kv_proj` reads A once.
Task order is row-major over N, so consecutive tasks reuse the same A tile and
should hit L2. This is acceptable for a first correct kernel; a later version
would keep A resident across N tiles in L1. No performance number is claimed.

## API

```python
import blocked_fp32_gemm83 as g

spec = g.build_kernel(2043, 5120, 128)   # compiles once, cached, exclusive lock
spec["executed"]                          # always False: build != run
spec["gm_shapes"]                         # allocate A/B/C from this
spec["kernel_symbol"], spec["obj_path"], spec["block_dim"]   # 5-GM ABI, blockDim=cores

a = g.pack_a(x_half_mk)      # (M,K) Half  -> (K1, Mp, 16) Half, zero padded rows
b = g.pack_w(w_half_nk)      # (N,K) Half  -> (K1, N, 16)  Half, values untouched
c = torch.empty(spec["gm_shapes"]["c"], dtype=torch.float32, device="npu:0")
# launch: [a, b, aux0(Half16), aux1(Half16), c] on the current torch stream
out = g.unpack_c(c, m=2043)  # (N1,Mp,16) Float32 -> (M,N) Float32, crops padding
```

`PaddedWeightCache` is available if the root wants to hoist the weight packing
out of the timed region. Its signature includes `storage_offset` because H3's K
and V projections are two views of the *same* QKV storage: a shared `data_ptr`
alone must never be treated as the same cache entry, and a changed signature for
one logical view raises instead of silently re-caching.

Only the launch is the root's; `build_kernel` also caches the binary under
`/tmp/h3_round83_blocked01/kernels/<name>/` with a `spec83.json` sidecar.

## Staged verification the root should run (in this order)

1. `--plan` and `--capability` (no device work): confirm the pinned SoC/core/UB
   guard passes on the live host.
2. Build the **smoke geometry** `(128, 256, 128)`: one core, one tile, **two** K
   blocks. This is the only cheap case that exercises real K-block accumulation;
   compare against the CPU Half Linear. If accumulation or the WAR sync were
   wrong, this fails first and tells you which.
3. Build `kv_proj` (16 tasks, 40 K blocks, one N tile) and compare against CPU
   Half Linear; then `q_proj`/`mlp_up` (multi-step round-robin), then
   `row_parallel`.
4. Only then consider either `mb=64` (halves A re-reads, doubles tasks) or a
   persistent-A variant. Timing must include packing, weight packing and the
   FP32→Half final cast to be comparable with the encoder budget.

## Explicit non-claims

* Not compiled, not executed, no numerics and no timing exist yet.
* Not H3 end-to-end: no encoder integration, no FP32 HCCL reduce, no final Half
  cast. The FP32 GM output is only useful once the root keeps it FP32 through the
  collective and narrows once at the end.
* `pack_*`/`unpack_c` are pure permutation plus zero padding — they must not be
  used to introduce any arithmetic, dtype shadow or new teacher. The acceptance
  reference stays the verified independent CPU FP16 oracle from Round78.