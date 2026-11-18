# Round85 — batched per-head Half→FP32 GEMM (source only)

This is a **fresh-fork, source-only** subtask for task-26. Nothing here was
compiled, launched, imported on an NPU, or run against the model. There is no
precision result, no speed result, and no H3 improvement claim in this
directory. The round85 root agent owns every runtime receipt.

## What was added

[batch_gemm85.py](batch_gemm85.py) adds a leading **head** dimension to the
already-executed Round83 blocked FP32-output GEMM, without changing its
arithmetic. Round83 remains byte-verified as the reference and is never edited
or imported for mutation:

| | Round83 (executed) | Round85 (this fork) |
|---|---|---|
| A GM | `(K1, Mp, 16)` Half | `(H, K1, Mp, 16)` Half |
| B GM | `(K1, N, 16)` Half | `(H, K1, N, 16)` Half |
| C GM | `(N1, Mp, 16)` Float | `(H, N1, Mp, 16)` Float |
| tasks | `m_tiles*n_tiles` | `H * m_tiles * n_tiles` |
| 5-GM ABI | `[A,B,aux0,aux1,C]` | identical, unchanged |
| accumulator | one FP32 L0C over all K | identical |

Two Round83 properties that cost the most to get right are preserved verbatim
because they are *measurement* properties, not style:

- the K loop stays a **static Python loop** with `init_l1out=(kt == 0)`, because
  the SDK requires a Python `bool` there — so a single FP32 L0C stays live across
  all K blocks and no Half partial or FP32-UB partial sum ever exists;
- each K block is wrapped in `new_stmt_scope(disable_sync=False)`, which orders
  the L1 re-load (WAR) against the previous cube read.

Only the outer tile-step iteration is dynamic (`for_range`). The head is folded
into the flattened task index
`head*(m_tiles*n_tiles) + m_tile*n_tiles + n_tile`, so compiled IR does not grow
with the head count — important because the real attention product is 8 heads.

## Target geometries

| case | H | M | K | N | m_pad | K tiles | m_tiles | n_tiles | tasks | cores | steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| score | 8 | 512 | 128 | 2048 | 512 | 1 | 4 | 16 | 512 | 30 | 18 |
| score last chunk | 8 | 507 | 128 | 2048 | 512 | 1 | 4 | 16 | 512 | 30 | 18 |
| P@V | 8 | 512 | 2048 | 128 | 512 | 16 | 4 | 1 | 32 | 30 | 2 |
| P@V last chunk | 8 | 507 | 2048 | 128 | 512 | 16 | 4 | 1 | 32 | 30 | 2 |
| smoke | 2 | 128 | 256 | 128 | 128 | 2 | 1 | 1 | 2 | 2 | 1 |

The score case pads the **key** axis (true keys 2043 → 2048) and the last query
chunk pads M (507 → 512, 5 zero rows). Both paddings are host-side: the kernel
has no masking semantics and the padded rows are exact zeros. The smoke case is
the smallest geometry that exercises real K accumulation (2 K blocks) *and*
distinct per-head B.

`P@V` is where batching actually matters: N=128 with 128×128 tiles gives only 4
output tiles per head, so 8 heads = 32 tasks ≈ 30 cores instead of 4 cubic tiles
on a 30-core part.

## Head isolation

Each head owns a **distinct dense B matrix**; no head ever reads another head's
weights. The head stride is added to all three offsets:

```
a_off = head*(K1*Mp*16) + (kt*kb1*m_pad + base_m)*16
b_off = head*(K1*N*16)  + (kt*kb1*n + base_n)*16
c_off = head*(N1*Mp*16) + ((base_n//16)*m_pad + base_m)*16
```

Per-head DMA burst/strides are therefore unchanged from Round83 (within a head
the layout is dense), and the tests verify every A/B/C span for every head stays
inside `[head*stride, (head+1)*stride)`.

## API for the root driver

```python
import batch_gemm85 as g
spec = g.build_kernel(8, 512, 128, 2048, out_dir=<root kernel dir>)
a = g.pack_a(q_half_h_m_k)   # (H, K1, Mp, 16) Half, zero M padding
b = g.pack_w(k_half_h_n_k)   # (H, K1, N, 16)  Half, original values untouched
c = torch.empty(spec['gm_shapes']['c'], dtype=torch.float32, device=dev)
# launch 5 GM pointers per spec['kernel_arg_order'] via runtime5gm83
out = g.unpack_c(c, m=512)   # (H, 512, 2048) float32
```

`spec` carries the keys `runtime5gm83` already consumes — `obj_path`,
`obj_sha256`, `kernel_symbol`, `block_dim`, `dst_dtype`, plus GM shapes/bytes —
so no new launcher header or C++ shim is needed. `build_kernel` returns
`'executed': False` unconditionally.

Also available: `plan_tiling`, `flatten_task`, `gm_offsets`, `kernel_name_for`,
`capability_report`, `check_reference_r83_source`, and `pack_p` (an alias of
`pack_a` for the probability operand).

Kernel names are `batch_gemm85_h{H}_m{M}_k{K}_n{N}_mb{mb}_nb{nb}_kb{kb}_c{cores}`
— a distinct namespace from Round83's `blocked_*83_*`, and every target shape
gets a unique name.

## Build safety

Compilation reuses the executed Round83 pattern: a per-out-dir `fcntl` lock
(`.compile.lock`), a **unique** `mkdtemp` workspace per compile, artefacts
`copyfile`d into the destination, the object SHA-256 recorded and re-validated
on cache hits, and a refusal to overwrite an existing kernel directory. The
source contains no `rmtree`/`os.remove`/`unlink`/`shutil.move` — a failed build
retains its exact workspace for diagnosis instead of deleting computed paths.

## Tests

[test_batch_gemm85.py](test_batch_gemm85.py) — **68 stdlib tests, all pass**,
no torch/CANN/NPU/network required (torch and `tbe` are imported lazily inside
functions; a module-scope import check enforces this). Coverage:

- plan arithmetic for all target shapes, budget/L1/L0C/UB bounds, edge rejections;
- task flattening proven **bijective and head-complete**;
- **head-leak simulation**: synthetic `(head, n, k)`-tagged B matrices are walked
  through the kernel's own offset math and every gathered tile is asserted to
  contain only its own head's tags;
- per-head A/B/C spans bounded, 16-element alignment, C-store bounds;
- DMA strides asserted equal to the Round83 formulas;
- pure-python reference implementations of `pack_a`/`pack_w`/`unpack_c` verify
  element placement, zero M padding (including that padding does not leak across
  heads) and round-trip ordering — without torch;
- static-K-init and single-L0C structural checks on the emitted source;
- pins: Round83 SHA-256 verified against the real file, and a modified copy is
  rejected;
- source-only invariants: no NPU/GPU artefacts, no numeric claim fields absent,
  no unsafe path deletion.

## Correction carried forward from Round84

R84 caught a stale planning assumption (an intermediate-size constant that did
not match the official checkpoint, giving K3456 instead of the actual K3200).
This fork deliberately derives **only** the declared attention geometry —
head_dim 128, 8 local heads per TP8 rank, sequence 2043 — and does not infer or
restate any model MLP width. The root should keep reading real checkpoint shapes
rather than reusing constants from earlier rounds.

## Open items for the root

- Prebuild serially on the leased owned 910A host before the 8-rank/native run.
- The FP32→Half decomposition of P (high + power-of-two-scaled residual, summed
  in Float before the single Half cast) is **not** in this file: it is two calls
  of this same operator plus an FP32 add, which is why no fused dual-PV kernel
  was written. Verify the residual scale on the real P distribution first; a
  small residual can round to zero or subnormal in Half.
- No numeric or timing claim is possible until the root executes the operator
  against a CPU FP16 `scaled_dot_product_attention` (MATH backend) reference on
  the same captured Half operands.