# Round83 bias epilogue: FP32 UB bias add, exactly one Half cast

**Source only. Nothing here has been compiled or executed.** This is task 21's
deliverable: a minimal TIK micro kernel that removes one *identified* rounding
difference between the 910A path and the CPU FP16 reference. It is not a
precision pass, not a speed result and not an H3 claim.

## What problem it removes

Round80 measured that a native biased `Half` Linear behaves as *no-bias Half
GEMM followed by a Half bias add*, i.e. it rounds the bias contribution to Half
before the CPU reference would. Round82 tried the hardware alternative — the
`matmul` bias operand — and the SDK rejected it **before compilation**:

```
matmul param bias, not support Ascend910
```

The full/runtime SoC label is `Ascend910B`, but the TIK bias gate uses the short
SoC name and this part reports `Ascend910`. Forcing an A2 SoC profile to slip
past that gate is not acceptable, so this kernel adds the bias in FP32 instead.

## Arithmetic order in the kernel

```
FP16 A, B                     (original Half inputs, unchanged)
  -> t.matmul(acc_l0c, a_l1, b_l1, m, k, n, init_l1out=True, bias=None)
       FP32 L0C accumulator                      # no bias operand at all
  -> t.tensor_mov(c32, acc_l0c, "m", 1, m*n*4//1024, 0, 0)
       FP32 L0C -> FP32 UB                       # no narrowing, all bits kept
  -> t.vec_add(16, c32[g,:,:], c32[g,:,:], bias_ub[g*16], m, 2, 2, 0)
       FP32 bias add in UB, once per 16-column group
  -> t.h_cast(c16, c32, "none")
       exactly ONE FP32 -> FP16 conversion
  -> t.data_move(out_gm, c16, 0, 1, m*n*2//32, 0, 0)
       packed FP16 output (N1, M, 16)
```

That is the CPU `F.linear` Half rounding order: accumulate in FP32, add the
bias, narrow once.

`bias_gm` is **float32, but it must contain the exact original Half bias values**
promoted to FP32 by the caller. No FP32 weights and no FP32 shadow copies are
permitted — the whole point is that only the epilogue arithmetic changes.

## API for the root driver

```python
from tik_bias_epilogue83 import build_kernel

meta = build_kernel(variant="fp32_bias_fp16_out",
                    m=128, k=512, n=128,
                    out_dir=root / "kernels")
meta["obj_path"]      # kernel_meta/*.o
meta["obj_sha256"]
meta["kernel_symbol"] # e.g. tik_bias83_fp32_bias_fp16_out_m128_k512_n128__kernel0
meta["block_dim"]     # 1
meta["dst_dtype"]     # 'float16'
```

Five GM arguments, in this exact order (the proven Round82 launcher ABI):

| # | argument | dtype | layout | bytes |
|---|---|---|---|---|
| 0 | `a_gm` | float16 | `(K1, M, 16)` | 131072 |
| 1 | `b_gm` | float16 | `(K1, N, 16)` | 131072 |
| 2 | `bias_gm` | float32 | `(N,)` — exact original Half values | 512 |
| 3 | `aux_gm` | float16 | `(16,)` — **unused by the math** | 32 |
| 4 | `out_gm` | float16 (or float32 for the diagnostic variant) | `(N1, M, 16)` | 32768 |

The aux buffer exists only because the already-proven launcher takes five GM
pointers; it is never read by the kernel. No new C headers are needed: the
legacy runtime headers are absent on this install, which is why Round82's
separate C shim failed while the existing `/tmp/libfa910a.so`
(SHA `bd50d700…`) works.

Variants:

| variant | dst | bias | role |
|---|---|---|---|
| `fp32_bias_fp16_out` | float16 | FP32 add | primary target |
| `fp32_bias_fp32_out` | float32 | FP32 add | diagnostic: proves the bias add is exact, no cast |
| `fp16_out_unbiased` | float16 | none | control: must reproduce Round82's unbiased Half result |

## Layout conventions (host packing, validated by tests)

```python
packed_A[(k//16), m, 16][q, i, z] = A[i, q*16 + z]
packed_B[(k//16), n, 16][q, j, z] = W[j, q*16 + z]
packed_C[(n//16), m, 16][g, i, z] = C[i, g*16 + z]
bias slice for group g          = bias[g*16 : g*16+16]
```

These match the layout Round82 actually executed (that kernel's output was
verified against an FP32 numpy golden at 4.081e-7 relative RMS).

## SDK facts this kernel depends on

Every claim below was read from the CANN 9.2.0 sources copied under `sdk/`, with
per-file hashes in [manifest83.json](<manifest83.json>):

* `rep_stride` is in **32-byte blocks** (`tik_tensor_op.py` extent arithmetic
  multiplies strides by `ONE_BLK_SIZE`). `mask=16` FP32 elements = 64 B = 2
  blocks, so `rep_stride=2` advances exactly one packed row.
* An integer mask in normal mode means an **element count**
  (`common_check_func.get_32bit_dtype_mask_len`: `mask_len = mask[0]`, required
  `<= 64` for 32-bit dtypes). So `mask=16` really activates 16 FP32 elements.
* `check_tensor_op_overflow` asserts
  `extend_offset + offset + ori_offset <= reduce_mul(original_shape)`. For the
  last of eight groups that is `2048 + 7*2048 = 16384 ==` the tensor size, so
  exact-size UB buffers pass — no padding, no out-of-bounds read or write.
* In-place `vec_add` is allowed: the overlap check only rejects the pattern
  "iteration N's destination is iteration N+1's source". Here `dst` and `src0`
  are the identical view on the same iteration, whose intervals are adjacent.
* `round_mode "none"` is valid (`ROUND_MODE_MAP["none"] = ""`), building dtype
  string `f322f16` — round-to-nearest, matching CPU `.half()`.
* `h_cast` requires `dst` and `src` to have the **same shape**; both are
  `(N1, M, 16)` here.
* `h_cast(fp16_dst, fp32_src, "none")` is proven on this exact device:
  `artifacts/diagnostics/round57/source/flash_attention_fwd.py:251` used it in
  a kernel that executed correctly on this chip.
* The compile profile is `Ascend910B` with **30 cores** — Round82's live DCMI
  read all 8 chips at 30 cores. 32 GiB is memory, not core count. Nothing here
  overrides the SoC to `Ascend910A` or to an A2 profile.

## Tests

`python3 -m unittest test_tik_bias83 -v` — 22 stdlib-only tests, no torch, no
CANN, no device. They cover: the emitted instruction sequence (matmul without a
bias operand, FP32 accumulator preserved, one `vec_add` per group, exactly one
`h_cast` at the end), layout round-trips against Round82's executed convention,
and the SDK's own buffer-offset arithmetic proving the last group fits exactly.

What the tests **cannot** establish: that this compiles, that it runs, or that
the encoder matches the CPU FP16 reference. That is the root's execution.

## Explicit non-claims

- Not compiled, not executed, no `kernel_meta` artefact exists.
- Micro geometry M128/K512/N128 only. The H3 shapes (M=2043, K=5120) need
  blocked tiling, which does not exist yet.
- A micro rounding match does not prove full-encoder CPU FP16 agreement; the
  Round81 complete-encoder gap was 7.42% and Round82's conditioning isolation
  left 2.48% residual. Neither is waived here.
- No BF16 compute, no BF16 teacher, no BF16 agreement requirement.
- No speed measurement of any kind.