# Round97 — Float-NZ score row to true-4032 unmasked Float softmax with dual Half probability packing

**SOURCE ONLY.** The author of this directory never compiled, launched or measured
anything here: no CANN/TBE build, no NPU execution, no torch or model run, no SSH,
no install, no production edit, no commit. [softmax_pack97.py](<softmax_pack97.py>)
is a kernel source plus its planning and validation layer; every binary, timing and
numeric result belongs to the root agent that builds and runs it. Nothing here is an
accuracy result, a speed result or a lossless acceptance.

## What this stage replaces

The Round88 vision path unpacks a ~134 MB FP32 score tensor per chunk in Python, adds
hi/lo Half probability packs and re-packs the result. This kernel does that whole stage
in **one launch with five GM pointers**:

```
Float32 score NZ row -> * (72 ** -0.5) -> mask 64 key tail -> max -> exp
                     -> zero tail -> sum -> 1/sum -> p
                     -> hi = Half(p), lo = Half((p - float(hi)) * 1024)
                     -> both packed straight back into Half NZ
```

The score operand has exactly the layout the PV stage already consumes as `A_NZ`:
`(head, n1=256, m_pad, 16)`. So one query row of one head is 256 blocks of 16 values
= 4096 keys, and no Torch unpack, pad, softmax, pack or dtype conversion is needed
between the two stages.

## Fixed geometry and the five-pointer wire

| Item | Value |
|---|---|
| heads | 16 |
| padded query rows | 512 (real `m` is 512 or 448) |
| padded keys | 4096 = 256 NZ blocks × 16 |
| **true keys** | **4032** = one image's 48×84 patch grid, *before* the 2×2 merger |
| key tail | 64 (masked, never a real key) |
| head dim / scale | 72 / `72 ** -0.5` (the padded 128 is **not** the model scale) |
| lo rescale | 1024, so PV reconstructs `p ≈ hi + lo/1024` |
| core tasks | `16 × 512 = 8192` rows over 30 cores, 274 steps |

GM argument order, all five passed explicitly to `BuildCCE`:

```
[ scores float32 (16,256,512,16), aux0 float32 (16,), aux1 float32 (16,),
  hi float16 (16,256,512,16), lo float16 (16,256,512,16) ]
```

`aux0`/`aux1` are unused placeholders that keep the proven five-pointer ABI visible to
the compiler; auto-collection would prune them.

## Why the DMA descriptors differ between load and store

| | burst | stride | meaning |
|---|---|---|---|
| score load | 2 × 32 B | `src_stride = 2*(512-1)` | 16 FP32 = 64 B = 2 units |
| prob store | 1 × 32 B | `dst_stride = 512-1` | 16 Half = 32 B = 1 unit |

The factor of two is the **byte width**, not an element index. Using `2*(m_pad-1)` on
the Half store would corrupt every second block; `row_element_coverage()` proves both
descriptors tile one row exactly, and a stdlib test asserts the store never takes the
Float factor.

## Masking and the endpoint pad

- The 64 padded key slots are written to `-FLT_MAX` **before** max and exp, so they can
  never win the row maximum nor add a spurious `exp(0-max)` term.
- The tail is then **explicitly re-zeroed before the denominator**, so the sum is over
  exactly the 4032 real keys even if a hardware `vexp` did something unexpected on the
  extreme masked input.
- The mask value is finite (`-3.4028234663852886e38`), never NaN and never `-inf`.
- Rows at or beyond the real row count (the 512→448 pad) write **all-zero** hi/lo for
  the whole row. They must never become a uniform `1/4096` softmax row, because the PV
  stage reads every padded row. A stdlib test pins that predicate for both `m` choices.
- Attention is **unmasked**: there is no causal mask anywhere in this source.

## Arithmetic and honest limits

Float32 stable softmax, then two Half leaves. The denominator uses `vec_rec` plus two
Newton steps (the same refinement shape as the executed Round90 erf kernel) purely to
get an accurate `1/sum`; it is **not** the Round94 RMSNorm Newton that was measured
negative and is not repeated here.

- **Not a bit-replay of any Torch softmax.** The Float reduction order (`vcmax`/`vcadd`)
  and the reciprocal are hardware orders; low bits can differ. The authority for
  numerics is the root's frozen-leaver CPU Half MATH comparison against the unchanged
  Round88 QKV fixture, not this document.
- No Float linear weight, no BF16 gold cast, no teacher tensor, no cached
  pixel/patch/hidden/DeepStack/conditioning value, no added tolerance.
- The two auxiliary Float GM operands carry no data.

## SDK capability evidence (read from the in-repo SDK copies)

The author could not import `tbe` locally (no CANN on the authoring host), so primitive
and signature evidence was read from the SDK copies already committed under
`artifacts/diagnostics/round83/bias/sdk/` and from project kernels that actually
executed:

| Need | Evidence |
|---|---|
| `vec_muls`/`vec_adds` accept a TIK `Scalar` | `common_params_check_.py:193` `_check_scalar` accepts `(int, float, Scalar, Expr)` |
| `vector_dup` accepts a `Scalar` src | `common_params_check_.py:303` same tuple |
| `vec_muls(mask,dst,src,scalar,repeat,dst_str,src_str,mask_mode)` | `tik_vector_api.py:631` |
| `vec_adds(...)` same shape | `tik_vector_api.py:599` |
| `vec_max/vec_add/vec_mul` signatures | `tik_vector_api.py:113/53/173` |
| `vec_rec(mask,dst,src,repeat,dst_str,src_str)` | `tik_vector_api.py:414` |
| `vec_exp`, `vec_abs` | `tik_vector_api.py:315/341` |
| `vcmax`/`vcadd` exist as intrinsics | `tik_params.py:908-910`; `VREDUCE_DST_ALIGN["vcmax"]=4` at `tik_params.py:619-626` |
| vreduce repeat limit 4095 | `tik_params.py:607` (`MAX_VREDUCE_REPEAT_TIMES`) |
| vector repeat limit 255 | `tik_params.py:77` (`MAX_REPEAT_TIMES`) |
| `vec_reduce_add(mask,dst,src,work,repeat,src_rep_stride)` | executed Round61/62 AdaLN norm: relative RMS ~1e-4 vs baseline, repeats exact |
| `h_cast(dst,src,'none')` both directions | executed Round90 erf: Half→Float widen and the one Float→Half narrow |
| FP32 4096 tile = 64 repeats × `rep_stride 8` | executed Round90 (`FP32_MASK=64`, `FP32_REP_STRIDE=8`) |
| offset UB slice as a vector dst | project FA patch `vector_dup(..., scores[block,0,0], ...)` |
| `t.Scalar('float32', name=...)` + `.set_as(t[idx])` | executed Round61 AdaLN |
| `t.else_scope()` | FA patch, Round57 sources |

`vec_reduce_max` has **no** executed precedent in this project, so its exact call shape
is an explicitly recorded assumption: it is taken as the same vreduce signature as the
verified `vec_reduce_add`, with the single-call-over-all-repeats convention that
Round61/62 used (result in `dst[0]`). The receipt field
`reduction.vreduce_signature_assumption` records this verbatim, and `_emit_kernel`
refuses to emit if the installed TIK instance lacks any required method. **The root's
build and probe are the authority.**

Float vreduce is gated: `capability_report()` queries
`platform.intrinsic_check_support` for `Intrinsic_vec_reduce_max` /
`Intrinsic_vec_reduce_add` over the `float32`/`f32` spellings, resolves the compile SoC
(must be `Ascend910`) and `CORE_NUM` (must be 30). An unsupported Float reduce raises a
`SoftmaxPackError` naming the exact checks instead of substituting a non-Float
reduction, an A2-only instruction, or a silently different maximum.

## Budget

UB total 66,240 B (scores/aux/zeros 4096 FP32 each, hi/lo 4096 Half each, plus reduce
work and 8-slot scratch), well under the 196,608 B safe bound. 64 repeats covers the
row and is inside both the 255 vector-repeat and 4095 vreduce limits.

## Files

| File | Purpose |
|---|---|
| [softmax_pack97.py](<softmax_pack97.py>) | kernel source + `build_kernel(batch,m,n,valid_keys,out_dir)` + planning/validation |
| [test_softmax_pack97.py](<test_softmax_pack97.py>) | 29 stdlib-only contract tests (no torch/tbe/NPU) |
| [manifest97.json](<manifest97.json>) | file bytes + SHA256, API evidence, capability and non-claim receipts |
| [make_manifest97.py](<make_manifest97.py>) | stdlib generator for the manifest (records hashes only) |

29 tests pass. They cover geometry, the true-key/tail split, endpoint-pad zeros,
per-head offsets and NZ row arithmetic, both DMA descriptors, the math-order invariants,
the five-pointer wire, the UB/repeat budgets, the capability fail-closed paths, the
8-wide vs 64-wide mask discipline, and the non-claim receipts. They do not compile or
run the kernel and they say nothing about its numerics or speed.

## Not claimed

`executed: false`, `compiled_now: false` until the root builds it,
`not_a_speed_claim: true`, `full_encoder_precision_accepted: false`,
`whole_goal_complete: false`, `numerical_tolerance: null`. This is a direct
softmax+pack stage only — no online FlashAttention fusion, no PV ABI assumption, no
change to the frozen Round88 QKV fixture or the CPU Half reference.