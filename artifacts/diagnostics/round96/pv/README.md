# Round96/pv — fused two-Half-probability P@V kernel (single launch, one Half store)

**SOURCE ONLY.** Authored here and never compiled, imported on an NPU, executed
against a model, installed into production or committed by its author: no SSH, no
CANN build, no torch import, no NPU/GPU work, no model run. Every runtime receipt
belongs to the root agent that compiles and launches it. Nothing in this directory
is a precision result, a speed result or an H3 improvement claim.

## Why this kernel exists

[Round88](../../round88/README.md) already made the vision attention leaf exact by
construction: original Half q/k/v are copied unchanged into zero-padded operands,
the executed Round85 batched GEMM keeps ONE FP32 L0C accumulator across every K
block, the Float32 softmax is unmasked and applies the original `72 ** -0.5`
scale, and the probability is split into `P_hi = Half(P)` /
`P_lo = Half((P - Float(P_hi)) * 1024)` before a Float32 combine and a single Half
cast.

What Round88 left on the table is structural, not numerical. Per query chunk it
issues **three** kernel launches (1 QK + 2 P@V) and brings **two** FP32 P@V
products back to the host, where torch scales, adds and casts. This directory
replaces that with **one** fused P@V launch: two independent FP32 L0C
accumulators, a Float32 UB scale+add, one `h_cast('none')`, one Half store.

## Arithmetic contract (deliberately unchanged)

```
P_hi = Half(P)                      P_lo = Half((P - Float(P_hi)) * 1024)
out  = Half( Float(out_hi) + Float(out_lo) * (1/1024) )       <!-- once -->
```

* `out_hi` and `out_lo` are the full-K contractions `P_hi @ V` and `P_lo @ V`,
  each accumulated in its own FP32 L0C across all 32 K blocks: **no Half partial
  and no FP32-UB partial sum**.
* `1/1024` is a power of two, so the downscale is exact for normal Float32 values;
  the combine is one IEEE Float32 addition, then **one** narrowing cast.
* No Float32 weight shadow, no BF16 gold cast, no Float32 input promotion, no
  learned Float32 parameter, no teacher.

The only difference from the executed Round88 path is *where* the two Float32
operations run (vector unit, before any narrowing) and that they happen inside the
kernel. Whether the vector unit rounds those exactly like torch is something the
root must **prove**, not assume.

## GM argument ABI (5 pointers — the proven fa910a launcher)

| order | operand | dtype | layout |
|---|---|---|---|
| 0 | `hi` (P_hi) | float16 | `(H, K1, Mp, 16)` |
| 1 | `lo` (P_lo) | float16 | `(H, K1, Mp, 16)` |
| 2 | `V` | float16 | `(H, K1, N, 16)` (host passes V as `(H, N, K)`) |
| 3 | `aux` | float32 | `(16,)` — **unused** five-GM ABI placeholder |
| 4 | `out` | float16 | `(H, N1, Mp, 16)` |

`aux` exists only because the verified launcher always passes five GM pointers; the
kernel never reads it, and it is passed explicitly to `BuildCCE` so operand
pruning cannot drop it.

Layout element maps (identical to the executed Round83/85 convention):

```
hi/lo[h, k1, m, k0] = P_[hi|lo][h, m, k1*16 + k0]
V    [h, k1, n, k0] = V[h, n, k1*16 + k0]
out  [h, n1, m, n0] = out[h, m, n1*16 + n0]
Mp = ceil(M/128)*128     rows M..Mp-1 of both A operands are exactly zero
```

## Target geometry (real H3 vision attention, per query chunk)

| M | K | N | m_pad | K tiles | tasks | cores | steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4096 | 128 | 512 | 32 | 64 | 30 | 3 |
| 448 | 4096 | 128 | 512 | 32 | 64 | 30 | 3 |

`K=4096` is the true 4032 keys plus 64 zero-padded keys; `N=128` is the true
head_dim 72 plus 56 zero-padded columns. Tasks are `16 heads x 4 M tiles x 1 N
tile = 64` over the 30 physical cores of the short `Ascend910` TIK SoC.

## Resource budget (checked, not hoped)

| resource | need | limit |
|---|---:|---:|
| L1 | 98,304 B (a_hi 32 KiB + a_lo 32 KiB + b 32 KiB) | 1,048,576 B |
| L0C | 131,072 B (two FP32 tiles) | 262,144 B |
| UB | 163,840 B (two Float32 tiles + one Half tile) | 196,608 B floor |
| vector repeats/chunk | 32 (mask 64) | 255 (8-bit immediate) |

The full Float32 tile is 16,384 elements = **256** repeats at mask 64, which
overflows the 8-bit repeat immediate, so the combine is split into 8 static
N-group chunks of 2,048 elements (32 repeats each). `mask=64` and `rep_stride=8`
and the L0C→UB `block_mode='m'`/64 KB transfer are the shapes the executed Round90
erf kernel and the executed R83/R87 bias epilogue used on this hardware.

## Store geometry (the Half-vs-FP32 trap)

The executed Round85/87 kernels stored **FP32** output, so their NZ store used
`burst = 2*mb` and `dst_stride = 2*(m_pad-mb)`. This kernel stores **Half**, so one
output row is exactly one 32 B unit:

```
nburst = nb//16 = 8, burst = mb = 128, src_stride = 0, dst_stride = m_pad - mb = 384
```

The stdlib test suite asserts this halving against the Round85 plan, so the FP32
factor cannot leak back in.

## Launch accounting (structural, not a measurement)

| | Round88 | Round96 |
|---|---:|---:|
| launches per enabled call | 24 (8 QK + 16 P@V) | **16** (8 QK + 8 fused P@V) |
| FP32 P@V products in GM per chunk | 2 | **0** |
| host combine / scale / cast per chunk | yes | **none** |

Fewer launches and less host traffic are an expectation from the structure. **No
speedup is claimed or measured here**; only the root's paired warm clocks inside a
real encode can say anything, and those must include all pixels, all 27 vision
blocks, all 50 text layers, HCCL, layout, packing and D2H.

## API

```python
import pv_high_low96 as pv
spec = pv.build_kernel(16, 512, 4096, 128, out_dir=kernel_dir)   # compiles once
meta = spec                                   # runtime5gm83-compatible fields:
                                              # obj_path/obj_sha256/kernel_symbol/block_dim
a_hi = pv.pack_a(p_high)                      # (H, M, K) Half -> (H, K1, Mp, 16)
a_lo = pv.pack_a(p_low)
b    = pv.pack_w(v_t_h_n_k)                   # (H, N, K) Half -> (H, K1, N, 16)
out  = torch.empty(spec["gm_shapes"]["out"], dtype=torch.float16, device=dev)
runtime.launch(spec, [a_hi, a_lo, b, aux_float, out], synchronize=True)
res  = pv.unpack_out_half(out, m=m, n=128)    # (H, M, N) Half
```

`accurate_vision_attention_pv96.AccurateVisionAttentionPv96(candidate, runtime,
kernel_dir)` mirrors the Round88 engine API: `prepare()`, `set_enabled(bool)`,
`forward(q, k, v)`, `snapshot()`, `uninstall()`, `padding_selfcheck()`, plus the
`install(...)` factory. It patches exactly one attribute, the candidate module's
own `_vision_sdpa_bf16_compatible`, and installs **disabled**; `set_enabled(True)`
refuses before a successful `prepare()`.

## What the root must still establish

1. Both binaries compile on the pinned 21.9-style TIK toolchain and `blockDim` is
   30.
2. Two live FP32 L0C accumulators coexist in one task body without an L0C
   allocation failure.
3. The Float32 vector combine reproduces the executed Round88 host combine: either
   bit-identical PV outputs against the frozen R88 QKV capture, or a documented,
   *measured* difference with no post-hoc tolerance.
4. The fused kernel accepts both `M=512` and the `M=448` tail with zero key
   padding, and the padded M rows come back as exact zeros that the host crops.
5. Any speed statement comes from paired full-encode warm clocks, never from this
   file.

## Reproduce the contract tests (stdlib only, 50 tests)

```
cd artifacts/diagnostics/round96/pv
python3 -m unittest test_pv_high_low96 test_accurate_vision_attention_pv96 \
    test_emission96 -v
python3 refresh_manifest96.py     # only after editing a delivered file
```

They import no torch, torch_npu, TIK or CANN, and check tiling against the
executed Round85 plan, budgets, repeat immediates, GM offsets, task flattening,
the five-pointer ABI, the Half store stride, the pinned source SHAs, the
fail-closed activation gate, the disabled-path delegation and the scoped
install/uninstall.

`test_emission96.py` runs the **real** `_emit_kernel` against a recording TIK stub
and evaluates every emitted address numerically with fixed `core`/`step` samples.
It therefore verifies, without compiling anything:

* the five GM operands in ABI order and their exact declared shapes;
* two float32 L0C accumulators plus three UB and three L1 tiles;
* 32 K blocks of `(a_hi load, a_lo load, shared V load, a_hi matmul, a_lo matmul)`;
* `init_l1out` true on the first K block only, for both products;
* L0C->UB low before high, 8 float32 scales then 8 float32 adds, exactly one
  `h_cast('none')`;
* the Half store `8 / 128 / 0 / 384`, never the FP32 `256 / 768`;
* emitted A/B/out addresses equal to the planned head-strided, K-block-stepped
  offsets.