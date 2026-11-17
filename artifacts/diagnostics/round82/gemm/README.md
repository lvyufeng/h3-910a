# Round82 GEMM prototype — FP16 operands, FP32 accumulator, 910A

Source-only prototype for the custom kernel route the H3 FP16 work needs.
**Nothing in this directory has been compiled or executed.** No precision, speed,
or H3-improvement claim is made here.

## The finding that motivates it

Round80 established two concrete defects in the native path:

* a biased `Half` Linear is numerically *no-bias Half GEMM, then a separate Half
  bias add* — one extra rounding the CPU FP16 reference does not do;
* TP row-parallel projections round the FP16 partial GEMM output **before** the
  FP32 all-reduce, so the collective cannot recover it.

and that the framework op refuses the obvious fix:

```
npu_grouped_matmul([F16 x], [F16 w], output_dtype=torch.float32)
  -> 161002 "when non-quant ... y dtype should be FLOAT16"
```

[SDK_FINDINGS.md](<SDK_FINDINGS.md>) shows that **161002 is an op-interface
restriction, not a hardware limit.** CANN 9.2.0's own `"matmul"` capability table
lists, for both `ASCEND_910AIC` and `ASCEND_910BAIC`:

```python
["s8s8s32", "u8s8s32", "f16f16f32"]
```

`f16f16f32` = float16 operands into a **float32 destination**, and there is **no
`f16f16f16` entry at all** — FP32 accumulation is the native mode on this part,
and a half output is always an explicit cast. Two further SDK facts complete the
picture: `fixpipe` accepts a **float32 GM destination** from the FP32 L0C source,
and bias is a **910B-gated float32-in-L1** operand that is added **in FP32**
(`bias.dtype == dst.dtype`), which is exactly the rounding the native path loses.

Every claim is quoted with file, line and SHA-256 of the copy that was read;
those copies are under [`sdk/`](<sdk/>) and listed in
[sdk_receipts82.json](<sdk_receipts82.json>).

## What is in the prototype

[gemm_fp16_fp32out82.py](<gemm_fp16_fp32out82.py>) builds one kernel of two
variants through the non-deprecated DSL entry
`tbe.dsl.compute.mmad_compute.matmul` + `tbe.dsl.auto_schedule` + `tbe.dsl.build`:

| variant | dst_dtype | bias | arithmetic | use |
|---|---|---|---|---|
| `fp32_out` | float32 | none | FP32 accumulate → FP32 out, no narrowing | TP partials that must survive the FP32 collective |
| `fp32_bias_fp16_out` | float16 | float32, L1 | FP32 accumulate → **FP32 bias add** → one FP16 cast | the CPU-FP16-style linear |

Geometry is deliberately a **single tile** (`M=128, K=512, N=128`, A+B = 256 KiB
against a 1 MiB L1, L0C 64 KiB), because TIK's high-level matmul requires the
whole A and B resident in L1 and the real H3 shapes cannot be — `kv_proj` alone is
~20 MiB of float16. The H3 shapes are recorded in the module as the target, **not**
as something this kernel builds.

[gemm_launcher82.c](<gemm_launcher82.c>) is the launch shim, reusing the path
already proven on this chip ([KERNEL_RUNS.md](<../design/ascendc_flash_attn/KERNEL_RUNS.md>),
[fa910a_lib.c](<../design/patches/fa910a_lib.c>)):

```
rtDevBinaryRegister -> rtFunctionRegister -> rtKernelLaunch
```

`aclrtBinary*` loads and submits fine and then faults with
`ACL_ERROR_RT_AICORE_EXCEPTION (507015)` on the very same object file, so it is
deliberately absent.

[run_gemm_probe82.py](<run_gemm_probe82.py>) is what the root runs: build →
compile the shim → register → launch twice → compare against a numpy FP32 golden
computed from the *same* FP16 operand values, with a separate
`vs_native_half_bias_ordering` control so the bias-rounding question is answered
directly rather than inferred.

## One correction to earlier kernel practice

Earlier TBE work in this repo forces `set_current_compile_soc_info("Ascend910A")`
to escape a wrong `Ascend310B1` default. That is **wrong for the bias variant**:
the SDK gates bias on `is_910b_soc()`, which reads the *compile* SoC, and
`ASCEND_910B == "Ascend910B"` — the device's own reported name. The prototype
pins `Ascend910B` and asserts it, so a future edit cannot silently drop the bias.

## Status and honest limits

* Not compiled, not executed. The DSL build call, the `kernel_meta` artefact
  names and the `tbe.dsl.build` config keys are read from SDK source, not observed.
* `Intrinsic_mmad`/`f162f32` is a **runtime** query; the static table is strong
  supporting evidence, not the query result. `--capability` answers it on device.
* No H3 geometry, no blocked tiling, no performance number.
* A rounding match on the micro probe does **not** by itself close the Round81
  7.42% CPU-FP16 gap; that is the root's input-isolation experiment.

[manifest82.json](<manifest82.json>) pins every file hash, the SDK line numbers,
and the ordered next steps.