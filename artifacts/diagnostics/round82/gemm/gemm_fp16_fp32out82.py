#!/usr/bin/env python3
"""Minimal FP16-operand -> FP32-accumulator GEMM kernel prototype for 910A.

SOURCE ONLY. This module has never been compiled or executed. It is not a
precision result, a speed result, or an H3 improvement claim.

What it builds, and why this exact shape set
--------------------------------------------
The SDK's TIK `matmul(dst_l0c, a_l1, b_l1, m, k, n, init_l1out, bias)` requires
the WHOLE A and B operands to live in L1 (te/tik/api/cube/matmul.py
`_check_params`/`_check_overflow`), and L1 is 1 MiB on this part
(TBE_BUILD.md: L1_SIZE 1 048 576, L0C_SIZE 262 144). The H3 geometry
(M=2043, K=5120) does not fit, so a real deployment needs explicit blocked
tiling. This prototype therefore targets a geometry that DOES fit one tile, so
that the dtype question is answered before any tiling work is attempted:

    A: [M=128, K=512] float16  = 131 072 B
    B: [N=128, K=512] float16  = 131 072 B   (weight layout, trans_b=True)
    bias: [N=128]     float32  =     512 B   (L1, dtype must equal dst dtype)
    L0C: [M=128, N=128] float32=  65 536 B

Variants
--------
  fp32_out            dst_dtype float32, no bias
                      -> L0C FP32 leaves the cube as FP32. Use for TP partials
                         that must survive the FP32 collective unrounded.

  fp32_bias_fp16_out  dst_dtype float16, tensor_bias float32
                      -> FP32 accumulate, FP32 bias add, then exactly ONE FP16
                         cast. This is the arithmetic the CPU FP16 reference does
                         and the native biased torch Linear does not.

API route
---------
Compute is expressed through the non-deprecated DSL entry
`tbe.dsl.compute.mmad_compute.matmul` (see SDK_FINDINGS.md), scheduled with
`tbe.dsl.auto_schedule` and built with `tbe.dsl.build`. The resulting
kernel_meta/<name>.o + .json are launched by `gemm_launcher82.c`, which uses the
legacy rtDevBinaryRegister/rtFunctionRegister/rtKernelLaunch path already proven
on this chip (artifacts/design/ascendc_flash_attn/KERNEL_RUNS.md; the
aclrtBinary* family faults with 507015 on the same .o).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

# --- pinned contract (kept in sync with manifest82.json) ---------------------
PINNED_CANN_ROOT = "/home/mseco/Ascend/cann-9.2.0"
PINNED_COMPILE_SOC = "Ascend910B"
PINNED_DEVICE_SOC = "Ascend910B"

# Micro geometry that fits a single L1/L0C tile on this part.
MICRO_M, MICRO_K, MICRO_N = 128, 512, 128

# The real H3 row shapes this must eventually serve (documented, not built here).
H3_GEOMETRY = {
    "kv_proj": {"M": 2043, "K": 5120, "N": 128},
    "q_proj": {"M": 2043, "K": 5120, "N": 1024},
    "mlp_up": {"M": 2043, "K": 5120, "N": 3456},
}

VARIANTS = {
    "fp32_out": {"dst_dtype": "float32", "use_bias": False},
    "fp32_bias_fp16_out": {"dst_dtype": "float16", "use_bias": True},
}


class GemmPrototypeError(RuntimeError):
    """Raised instead of guessing when the environment does not match."""


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_compile_soc(soc: str = PINNED_COMPILE_SOC) -> None:
    """Pin the compile SoC before touching any TIK/DSL builder.

    The default on this install resolves to Ascend310B1 (TBE_BUILD.md), which
    selects a wrong instruction table. For the bias variant the SoC must be
    `Ascend910B`, because the SDK gates bias on `is_910b_soc()`, which reads the
    compile SoC name.
    """
    import tbe.common.platform as platform

    platform.set_current_compile_soc_info(soc)
    resolved = platform.get_soc_spec("FULL_SOC_VERSION")
    if resolved != soc:
        raise GemmPrototypeError(f"compile SoC is {resolved!r}, expected {soc!r}")
    return resolved


def capability_report() -> dict:
    """Read the two capability facts this kernel depends on, without building.

    This is a query of the loaded platform layer, so it needs CANN present, but
    it does not launch a kernel.
    """
    import tbe.common.platform as platform

    compile_soc = set_compile_soc()
    l0c_fp32 = platform.intrinsic_check_support("Intrinsic_mmad", "f162f32")
    report = {
        "compile_soc": compile_soc,
        "Intrinsic_mmad_f162f32": bool(l0c_fp32),
        "core_num": platform.get_soc_spec("CORE_NUM"),
        "l1_size": platform.get_soc_spec("L1_SIZE"),
        "l0c_size": platform.get_soc_spec("L0C_SIZE"),
        "note": ("Intrinsic_mmad/f162f32 is a runtime query; it is the same gate "
                 "gemm_compute._get_output_type uses to decide float32 vs float16 "
                 "accumulation"),
    }
    # The runtime/DCMI audit reports 30 usable cores for this original 910B
    # silicon bin (32GiB is memory,not physical core count). Do not substitute
    # the 32-core Ascend910A compiler default for verified runtime capacity.
    if int(report['core_num']) != 30 or int(report['l1_size']) != 1048576:
        raise GemmPrototypeError('compile profile does not match the verified 910A-generation hardware')
    if not l0c_fp32:
        raise GemmPrototypeError(
            "Intrinsic_mmad/f162f32 is not supported here; the FP32-accumulator "
            "route this prototype needs is unavailable and must not be faked")
    return report


def build_kernel(variant="fp32_out", m=MICRO_M, k=MICRO_K, n=MICRO_N, out_dir="kernels"):
    """Install the static operation context required by the standalone SDK tiler."""
    from tbe.common.context.op_context import OpContext
    with OpContext("static"):
        return _build_kernel_in_context(variant, m, k, n, out_dir)


def _build_kernel_in_context(variant: str = "fp32_out", m: int = MICRO_M, k: int = MICRO_K,
                 n: int = MICRO_N, out_dir: str | os.PathLike = "kernels") -> dict:
    """Build one prototype kernel and return its artefacts and metadata."""
    if variant not in VARIANTS:
        raise GemmPrototypeError(f"unknown variant {variant!r}; expected {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    if min(m, k, n) <= 0:
        raise GemmPrototypeError("m, k, n must all be positive")
    # TIK matmul accepts m,n in [1,4096] and k <= 32768/dtype_size.
    if m > 4096 or n > 4096:
        raise GemmPrototypeError("m and n must be <= 4096 for this TIK matmul")
    if k > 16384:
        raise GemmPrototypeError("k must be <= 16384 for float16 operands")

    soc = set_compile_soc()
    import tbe.dsl
    from te import tvm
    from tbe.dsl import matmul as dsl_matmul

    a = tvm.placeholder((m, k), name="a", dtype="float16")
    b = tvm.placeholder((n, k), name="b", dtype="float16")
    bias = tvm.placeholder((n,), name="bias", dtype="float32") if spec["use_bias"] else None

    result = dsl_matmul(
        a, b,
        trans_a=False, trans_b=True,
        format_a="ND", format_b="ND",
        alpha_num=1.0, beta_num=1.0,
        dst_dtype=spec["dst_dtype"],
        tensor_bias=bias,
        quantize_params=None,
        format_out=None,
        compress_index=None,
        attrs={},
        kernel_name=f"gemm_proto_{variant}",
    )

    tensor_list = [a, b] + ([bias] if bias is not None else []) + [result]
    kernel_name = f"gemm_proto_{variant}_m{m}_k{k}_n{n}"
    with tvm.target.cce():
        schedule = tbe.dsl.auto_schedule(result)
    if schedule is None:
        raise GemmPrototypeError('public DSL returned no CCE schedule for shape=%s dtype=%s tag=%s' %
                                 (result.shape, result.dtype, result.op.tag))

    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="gemm82_build_", dir=out_root)).resolve()
    if work.parent != out_root or not work.name.startswith("gemm82_build_"):
        raise GemmPrototypeError("unexpected build workspace")
    old_cwd = os.getcwd()
    try:
        os.chdir(work)
        tbe.dsl.build(schedule, {"name": kernel_name, "tensor_list": tensor_list})
        src_o = work / "kernel_meta" / (kernel_name + ".o")
        src_json = work / "kernel_meta" / (kernel_name + ".json")
        if not src_o.is_file() or not src_json.is_file():
            raise GemmPrototypeError(
                f"build produced no kernel_meta artefacts for {kernel_name}")
        meta = json.loads(src_json.read_text())
        dest = out_root / kernel_name
        if dest.exists():
            raise GemmPrototypeError(f"refusing to overwrite {dest}")
        dest.mkdir()
        shutil.copyfile(src_o, dest / (kernel_name + ".o"))
        shutil.copyfile(src_json, dest / (kernel_name + ".json"))
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(work, ignore_errors=True)

    return {
        "variant": variant,
        "dst_dtype": spec["dst_dtype"],
        "uses_fp32_bias": spec["use_bias"],
        "geometry": {"m": m, "k": k, "n": n},
        "compile_soc": soc,
        "kernel_name": kernel_name,
        "kernel_symbol": meta.get("kernelName"),
        "block_dim": meta.get("blockDim"),
        "core_type": meta.get("coreType"),
        "obj_path": str(dest / (kernel_name + ".o")),
        "obj_sha256": _sha256(dest / (kernel_name + ".o")),
        "json_path": str(dest / (kernel_name + ".json")),
        "kernel_arg_order": _arg_order(spec["use_bias"]),
        "built": True,
        "executed": False,
    }


def _arg_order(use_bias: bool) -> list:
    """Kernel parameter order, which the launcher must match exactly."""
    order = ["a_gm_float16", "b_gm_float16"]
    if use_bias:
        order.append("bias_gm_float32")
    order.append("out_gm_<dst_dtype>")
    return order


def build_plan(variant: str = "fp32_out") -> dict:
    """Pure description of what would be built; needs no CANN, no device."""
    if variant not in VARIANTS:
        raise GemmPrototypeError(f"unknown variant {variant!r}")
    spec = VARIANTS[variant]
    m, k, n = MICRO_M, MICRO_K, MICRO_N
    return {
        "variant": variant,
        "dst_dtype": spec["dst_dtype"],
        "uses_fp32_bias": spec["use_bias"],
        "geometry": {"m": m, "k": k, "n": n},
        "byte_sizes": {
            "a_float16": m * k * 2, "b_float16": n * k * 2,
            "bias_float32": n * 4 if spec["use_bias"] else 0,
            "out_float32": m * n * 4, "out_float16": m * n * 2,
        },
        "compile_soc": PINNED_COMPILE_SOC,
        "api": ("tbe.dsl.compute.mmad_compute.matmul(dst_dtype=..., tensor_bias=...) "
                "+ tbe.dsl.auto_schedule + tbe.dsl.build"),
        "launch": ("gemm_launcher82.c: rtDevBinaryRegister -> rtFunctionRegister -> "
                   "rtKernelLaunch (NOT aclrtBinary*)"),
        "kernel_arg_order": _arg_order(spec["use_bias"]),
        "executed": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="print the build plan only")
    parser.add_argument("--capability", action="store_true", help="query SoC capability")
    parser.add_argument("--build", choices=sorted(VARIANTS), help="build one variant")
    parser.add_argument("--out-dir", default="kernels")
    args = parser.parse_args()

    if args.plan:
        print(json.dumps(build_plan(), indent=2))
        return
    if args.capability:
        print(json.dumps(capability_report(), indent=2))
        return
    if args.build:
        print(json.dumps(build_kernel(args.build, out_dir=args.out_dir), indent=2))
        return
    parser.error("choose one of --plan, --capability, --build VARIANT")


if __name__ == "__main__":
    main()