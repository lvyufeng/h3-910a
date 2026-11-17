# 910A FP16→FP32 GEMM: what the SDK actually supports

**Post-review runtime correction:** the [direct TIK FP32-output kernel](<runtime06/probe_fp32_out.json>)
actually compiled and ran with bit-identical repeats. The [bias variant](<runtime06/probe_fp32_bias_fp16_out.json>)
was rejected before launch: `matmul param bias, not support Ascend910`.
Thus the authored inference below that the full `Ascend910B` label would enable
TIK bias is **not true on this part**; short-SoC normalization still rejects it.
Use a portable FP32UB bias epilogue instead of forcing past that gate.
The [live core audit](<live_core_count82.json>) confirms30 physical cores;
32 is the oldercompilerdefault,not this device's capacity.
The text below is the original source-only SDK analysis,not a substitute for
these actual execution receipts.

Source-only research. **Nothing here was compiled or executed on the device.**
Every claim below is quoted from CANN 9.2.0 source with the exact file, line and
SHA-256 of the copy inspected. This is a capability/API finding plus a prototype
that is ready for the root to compile and micro-execute; it is **not** a
precision result, a speed result, or a claim that the H3 encoder has improved.

## Why this matters

Round80 measured two real defects in the native path:

* native biased `Half` Linear equals *no-bias Half GEMM, then a Half bias add* —
  an extra rounding the CPU FP16 reference does not perform;
* row-parallel TP projections round the FP16 partial GEMM output before the FP32
  collective, so the collective cannot recover the lost precision.

Round80 also recorded that the framework op could not be asked for FP32 output:

```
npu_grouped_matmul([F16 x], [F16 w], output_dtype=torch.float32)
  -> 161002 "when non-quant ... y dtype should be FLOAT16"
```

**That is a framework-op interface restriction, not a hardware limit.** The
device's cube accumulates FP16 operands in FP32 natively, and both the DSL and
the TIK layer expose it.

## Decisive evidence: `f16f16f32` is the only float matmul combination

`python/site-packages/te/platform/cce_params.py` (SHA-256
`de52f14f0f6350c8cd9e055f74679e45db0ed787d70f2395ec09b00d1e414d07`),
`"matmul"` table at line 917:

```python
"matmul": {
    ASCEND_310AIC:   ["s8s8s32", "u8s8s32", "f16f16f32"],
    ASCEND_910AIC:   ["s8s8s32", "u8s8s32", "f16f16f32"],
    ASCEND_910BAIC:  ["s8s8s32", "u8s8s32", "f16f16f32"],
    ASCEND_910BVEC:  ["s8s8s32", "u8s8s32", "f16f16f32"],
    ASCEND_910_93AIC:["s8s8s32", "u8s8s32", "f16f16f32"],
    ...
}
```

Reading of this row set:

* `f16f16f32` = **float16 a, float16 b, float32 destination**. This is the *only*
  floating-point matmul combination in the list on any of these parts.
* There is **no `f16f16f16`**. A half-output matmul is expressed as
  `f16f16f32` (FP32 accumulator) followed by an explicit cast, not as a
  half-precision accumulate.
* The entries for `ASCEND_910AIC` **and** `ASCEND_910BAIC` are identical, so this
  is not an A2/A3-only fused capability. The runtime device reports
  `Ascend910B` and the TIK high-level matmul validates through
  `api_check_support("tik.matmul", dtype_str)` built exactly from this table
  (`te/tik/api/cube/matmul.py:465-467`, SHA-256
  `b22bbdd1665e54e4f52fb95fc1bfc9c02ba6781faf615e9a11b0c51e77d37829`).

## FP32 output is an explicitly supported fixpipe destination

`te/tik/api/cube/tik_cube_api.py` (SHA-256
`3056b887e9cb808d3704369710415e3d1d55baa9fe9e7df38409c3b7ca008930`), `fixpipe`
docstring:

```
dst: A tensor of type float16, float32, or int32, for the start element of the
     destination operand.
src: tensor of type float32 or int32, for the start element of the source operand.
```

So the L0C (FP32) → GM write accepts an **FP32 destination**: the accumulator can
leave the cube without an intermediate narrowing. This is the piece that makes a
"FP32 partials through the TP collective" kernel expressible at all.

## Bias is a 910B-gated FP32-in-L1 operand, added in FP32

`te/tik/api/cube/cube_common.py` (SHA-256
`fbd495d187ed43d68894d0819af040fd566b51ba4b377667b6ba7e2ae9a34398`),
`check_params_bias` at line 747:

```python
if bias is not None:
    check_equality((is_910b_soc() or is_310b_610l_soc()) and is_aicore_core(), True,
                   "%s param bias, not support %s." % (api_name, get_soc_name()))
    check_equality(init_1out, True, "if %s with bias, init_l1out should be True.")
    check_type_match(bias, Tensor, ...)
    check_equality(len(bias.shape), 1, ...)                 # 1-D [N]
    check_equality(bias.shape[0], kernel_shape[-2], ...)    # length N
    check_equality(bias.scope, scope_cbuf, ...)             # must be L1
    check_equality(bias.dtype, dst.dtype, ...)              # bias dtype == L0C dtype
```

`is_910b_soc()` (`te/tik/tik_lib/tik_soc_manager.py:109`) is
`soc_name in (ASCEND_910B, ASCEND_910_93)` and `ASCEND_910B = "Ascend910B"`
(`tbe/tik/common/tik_api_map.py:21`) — **exactly the SoC this device reports**
(`soc=Ascend910B` in `artifacts/diagnostics/round64/http/serve.log`).

Because the destination operand here is the FP32 L0C, `bias.dtype == dst.dtype`
means **bias must be float32**, and it is added into the FP32 accumulator. That
is precisely the missing behaviour: a bias add with **no Half intermediate
rounding**.

Note the compile-time subtlety: bias is gated on `is_910b_soc()`, which reads the
*compile* SoC. Earlier kernel work in this repo forced
`set_current_compile_soc_info("Ascend910A")` only to escape a wrong
default (`Ascend310B1`) instruction table. For the **bias** variant the correct
compile SoC is therefore `Ascend910B` — the device's own name — not `Ascend910A`.

## The DSL route exposes the same thing one level up

`tbe/dsl/api.py:1490` (SHA-256
`1b75184a13b767001173179529758f14ae3cc7b90ee66475a3ab50dc891ca4b0`):

```python
def matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a="ND",
           format_b="ND", alpha_num=1.0, beta_num=1.0, dst_dtype="float16",
           tensor_bias=None, quantize_params=None, format_out=None,
           compress_index=None, attrs={}, kernel_name="MatMul"):
    ...
    dst_dtype: output data type, support "float16" "float32", default is "float16"
    tensor_bias :the bias with used to init L0C for tensor c
```

with the output-type decision in `tbe/dsl/compute/gemm_compute.py` (SHA-256
`104ad98d6caffb6c66b6babf98ec74451e7c824604f7d67604272e0429d46089`):

* `_get_output_type` line 1483-1497: for `float16`×`float16` inputs,
  `dst_dtype` must be in `("float16","float32")` and
  `out_dtype = "float32"`; it only degrades to `"float16"` when
  `Intrinsic_mmad`/`f162f32` is unavailable
  (`l0c_support_fp32 = tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32")`,
  line 1481).
* `_get_tensor_c_ub` line 1147-1164: `dst_dtype == "float32"` with
  `l0c_support_fp32` and ND layout keeps the FP32 accumulator **without a cast**
  (`tensor_c_ub = tensor_c_ub_temp`), while `dst_dtype == "float16"` performs a
  single cast.
* bias handling line 2617-2648: the bias is combined **before** that cast, i.e.
  added in `out_dtype` (FP32). An FP16 bias, when `l0c_support_fp32`, is
  explicit-cast to FP32 (`tensor_bias_l0c = ... cast(..., dtype="float32")`).

So the DSL already gives, for free, the exact arithmetic ordering wanted:
FP16 operands → FP32 L0C accumulate → FP32 bias add → **one** FP16 cast.

## What this does not establish

* No kernel here has been built or run. `Intrinsic_mmad`/`f162f32` is a
  **runtime** query against the loaded SoC spec; the static table above is strong
  supporting evidence, not a substitute for the query result on the device.
* The large H3 geometry (M=2043, K=5120, N=128 KV / N=13824 MLP) does not fit one
  L1 tile, so a real deployment needs explicit blocked tiling. The prototype's
  first geometry is deliberately small enough for the high-level TIK matmul to
  tile on its own; the blocked variant is the follow-up, not a claimed result.
* `npu_grouped_matmul`'s `161002` remains a true statement about the *framework
  op*. The custom kernel is a different code path.