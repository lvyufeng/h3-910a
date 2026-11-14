# AscendC flash attention for H3 on 910A

Ported from `flash_attention-1.0.0-py3-none-any.whl` (TIK/TBE), which is a
910A-targeted FlashAttention-2 implementation. The wheel is the algorithm
reference; this directory is the AscendC rewrite.

## Why this shape of kernel

Measured on the real H3 attention shape (H=7, S=73408, D=128, fp16):

| op | per 1024-query chunk | throughput |
|---|---|---|
| QK^T | 0.0042 s | 32.2 TFLOPS |
| softmax | 0.0103 s | 205 GB/s |
| SV | 0.0017 s | 78.5 TFLOPS |

Per call that is matmul 0.425 s + softmax 0.742 s = 1.166 s, and over the
200 calls of a 4-step render softmax accounts for 148 s of 233 s.

So the win is **not** a faster matmul: it is never materialising the score
tile. At Br=Bc=128 a score tile is 128x128 fp32 = 64 KiB and stays in UB,
whereas the current PyTorch path streams a 1.05 GB tile per chunk four to
six times. That is the whole point of the port.

## Contract inherited from the TBE reference

Taken from `flash_attention_fwd.py`, `tik_ops_utils.py`, `constants.py`,
`tiling_strategy/sparse_tiling.py`.

Tiling (`SparseTiling`):

```
Br = min(128, Nq)      Bc = min(128, N)
Tr = ceil(Nq / Br)     Tc = ceil(N / Bc)
last_Br / last_Bc carry the tails; Tr/Tc are bumped when there is a tail
```

Fractal geometry (`constants.py`): `BLOCK_NUM = 16`, so `N0 = 16` and
`N1 = d / 16`. For H3 `d = 128`, giving `N1 = 8`.

Layouts:
* `Q` `(Nq, d)` fp16, `K` `(N, d)` fp16, `V` `(N, d)` fp16, all GM
* L1 holds `Qi` as `(d/16, Br, 16)`, i.e. K1 x M x K0 ("zN")
* `KjT` is loaded straight from `K_gm` with a strided burst, so the transpose
  is free — it is the same trick as `unroll_data_move` in the reference
* `Pij` is written to L1 as `(n1, m, n0)` and consumed by the second matmul
* `Oi` accumulates in **L1 as fp32** `(d/16, Br, 16)`, `li`/`mi` in UB

Softmax bookkeeping (FlashAttention-2, from `rowmax_compute` /
`rowsum_compute` / `output_compute`):

```
mi_new    = max(mi_old, rowmax(Sij))
scale     = exp(mi_old - mi_new)          # 1.0 on the first kv block
Pij       = exp(Sij - mi_new)
li        = li * scale + rowsum(Pij)
Oi        = Oi * scale + Pij @ Vj
```

Two details worth preserving:
* `rowsum(Pij)` is done **on the cube** by multiplying with an all-ones
  `(K, 16)` matrix in L1 and reading column 0, not by a vector reduction
  (`row_sum_cube_impl`). Cheaper and it avoids a UB transpose.
* `Pij @ Vj` accumulates in fp32 in L0C and is rescaled while still in L1,
  so the rescale is one pass over L1 rather than over UB.

Final normalisation (`output_norm_and_write`):

```
O = Oi / li            # calc_vec_rec gives 1/li
l = li * exp(mi)       # written as h_ln(li) + mi in log domain
```

The `l` output is the log-sum-exp; H3 does not consume it, so the AscendC
port writes only `O` and keeps `l` behind a compile-time flag.

Pipelining in the reference: `Sij` for kv block `j+1` is computed
(`flash_attention_fwd.py:371-388`) **before** block `j`'s output update, so
the cube and vector units overlap. The AscendC port uses the same skew
instead of a plain sequential loop.

## AscendC mapping

| TBE (TIK) | AscendC |
|---|---|
| `tik_instance.Tensor(..., scope=L1)` | `TBuf<TPosition::A1>` + `LocalTensor` |
| `scope=UB` | `TBuf<TPosition::VECCALC>` |
| `scope=L0C` | `TPosition::CO1` |
| `tik_instance.matmul(C_l0c, A_l1, B_l1, m, k, n)` | `Mmad(dst, aL0A, bL0B, ...)` after `LoadData` |
| `unroll_data_move` GM -> L1 strided | `DataCopy` with `DataCopyParams` |
| `tensor_mov(C_ub, C_l0c, "m", ...)` | `Fixpipe` with `DataCopyExtParams` |
| `h_exp` / `h_max` / `h_reduce_max` | `Exp` / `Max` / `ReduceMax` |
| `h_duplicate` | `Duplicate` |

Note the 910A vector-unit gaps found while writing the PyTorch fallback: a
whole-tensor fp16 `amax()` raises a ReduceMax inner error on some shapes,
so `ReduceMax` is always issued per-row with an explicit `srcShape`, never
as a full-tensor reduction.

## Second reference: ascend910a-extras

`https://github.com/monellz/ascend910a-extras` supplies the two pieces the TBE
wheel cannot: the AscendC cube plumbing and a working build path for custom ops
on this SoC. Not vendored here; fetch it directly.

**`csrc/opdev/op_kernel/matmul_core.h`** is the cube contract, and it differs
from what a CUDA-trained reader would guess:

```
A: GM ND -> L1 NZ   DataCopy(a1[dst], a_gm[src], {rows, 1, k/16-1, 0})
A: L1 NZ -> L0A ZZ  LoadData2d{repeatTimes=BLOCK_K/16, srcStride=BLOCK_M/16}
B (transposed): DN -> L1 ZN, then LoadData2d{repeatTimes=16, srcStride=1}
Mmad(acc, a2, b2, {m, n, k, cmatrixInitVal})   // cmatrixInitVal zeroes acc
L0C -> L0C2         DataCopyEnhancedParams{blockMode=BLOCK_MODE_MATRIX}
CO2 NZ -> GM ND     DataCopy(c_gm[dst], c[src], {rows, 1, 0, n/16-1})
```

Fractal block is 64 (`BLOCK_M=BLOCK_K=BLOCK_N=64`), and the whole transpose of
K is free because `K_gm` is already laid out transposed for the ZN path.

**`csrc/opdev/op_kernel/paged_attention_ex.cpp`** shows the attention skeleton —
separate `TQue` for Q/K/V/P/O, `row_max_buf`/`row_sum_buf` as the online-softmax
state initialised to `-5e4f` / `0.0f`, and `MmaQK`/`MmaPV` wrapping `Mmad` with
`cmatrixInitVal=true` only on the first KV step. Two caveats: it is decode-shaped
(`BLOCK_M=16`), and **its softmax body is commented-out pseudo-code** — the live
path is a plain `DataCopy`, so the vector softmax has to be written here rather
than copied.

## Build path

The repo's README pins the route, and it targets this exact SoC via
`-c ai_core-ascend910` (not `910b`):

```bash
msopgen gen -m 1 -i op_def.json -f pytorch -c ai_core-ascend910 -lan cpp -out ./csrc/opdev
```

so the flow is: `op_def.json` (written here) -> `msopgen` generates the host
tiling stub, kernel entry and aclnn registration -> fill in the compute ->
rebuild. That resolves the registration problem that otherwise blocks calling a
hand-written kernel from torch.

## Files

* `op_def.json` — operator declaration fed to `msopgen`
* `flash_attn_910a.h` — kernel body, template over tiling

## Status

Kernel and tiling are written to the contract above. Compilation for
`dav-c100` is the first gate; numerical validation needs the aclnn
registration (see `README` in this directory) and is not done yet.
