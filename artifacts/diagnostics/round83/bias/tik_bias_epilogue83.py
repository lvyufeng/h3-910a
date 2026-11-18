#!/usr/bin/env python3
"""Round83: TIK FP16 matmul -> FP32 L0C -> FP32 UB bias add -> single Half cast.

WHY THIS EXISTS
---------------
Round82 proved on this exact device that a direct TIK kernel can compute
FP16 x FP16 with an FP32 L0C accumulator and write **FP32** to GM
(M128/K512/N128, repeated launches bit-identical, 4.081e-7 relative RMS versus
an FP32-accumulate diagnostic).  Round82 also measured, in the framework path,
that a native biased Half Linear equals "no-bias Half GEMM + a Half bias add",
i.e. it pays an extra Half rounding that the CPU FP16 reference does not.

Round82's attempt to use the hardware bias operand instead was rejected by the
SDK *before compilation*:  ``matmul param bias, not support Ascend910``.
The full/runtime SoC label is ``Ascend910B``, but the TIK bias gate uses the
short SoC name and this part is ``Ascend910``.  Forcing an A2 SoC to slip past
that gate is not acceptable.

So this kernel keeps the accumulator path and adds the bias **in FP32 in UB**:

    FP16 A,B  ->  FP32 L0C (t.matmul, bias=None)
              ->  FP32 UB (tensor_mov, no narrowing)
              ->  + bias[n] broadcast in FP32 (vec_add)
              ->  exactly ONE FP32 -> FP16 conversion (h_cast, round_mode none)
              ->  FP16 GM, packed (N1, M, 16)

That is the rounding order of ``torch.nn.functional.linear`` in Half on the
CPU: accumulate in FP32, add the bias value, then narrow once.

SCOPE / NON-CLAIMS
------------------
Source only in this task: **not compiled and not executed here**.  It is a
private micro kernel for M128/K512/N128 -- not the H3 geometry (M=2043,
K=5120, which needs blocked tiling) and not a speed claim.  Matching this micro
control does not by itself prove full-encoder CPU FP16 agreement.

The bias tensor is FP32 but must contain the **exact original Half bias
values** promoted to FP32 by the caller; no FP32 weights/shadows are allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

# --- pinned contract -------------------------------------------------------
PINNED_CANN_ROOT = "/home/mseco/Ascend/cann-9.2.0"
PINNED_COMPILE_SOC = "Ascend910B"   # device reports Ascend910B, 30 usable cores
EXPECTED_CORE_NUM = 30              # live DCMI, all 8 chips (round82 receipt)
EXPECTED_L1_SIZE = 1048576
EXPECTED_L0C_SIZE = 262144

# --- micro geometry (one L1/L0C tile on this part) -------------------------
MICRO_M, MICRO_K, MICRO_N = 128, 512, 128

# The real H3 shapes this must eventually serve (documented, not built here).
H3_GEOMETRY = {
    "kv_proj": {"M": 2043, "K": 5120, "N": 128},
    "q_proj": {"M": 2043, "K": 5120, "N": 1024},
    "mlp_up": {"M": 2043, "K": 5120, "N": 3456},
}

#: Bumped once more by the root to make the latest criterion explicit.
#:
#: All variants share the same 5 GM arguments, in this order, so one launcher
#: ABI serves every variant (the aux buffer is unused by the math; it exists
#: because the proven launcher takes five GM pointers):
#:
#:   a_gm_float16, b_gm_float16, bias_gm_float32, aux_gm_float16, out_gm_<dst>
VARIANTS = {
    # primary: FP32 accumulate -> FP32 bias add -> ONE Half cast
    "fp32_bias_fp16_out": {"dst_dtype": "float16", "add_bias": True, "cast_final": True},
    # diagnostic: keep the biased result in FP32 (proves the bias add is exact)
    "fp32_bias_fp32_out": {"dst_dtype": "float32", "add_bias": True, "cast_final": False},
    # control: no bias, single cast -- must reproduce Round82's unbiased Half
    "fp16_out_unbiased": {"dst_dtype": "float16", "add_bias": False, "cast_final": True},
}

GM_ARG_ORDER = [
    "a_gm_float16",
    "b_gm_float16",
    "bias_gm_float32",
    "aux_gm_float16",
    "out_gm_<dst_dtype>",
]


class BiasKernelError(RuntimeError):
    """Raised instead of guessing when the environment or request is wrong."""


# --- host-side packing (pure python; unit-testable without torch) ----------
def pack_mk_index(i_m: int, i_k: int, m: int, k: int) -> int:
    """Flat index of A[i_m, i_k] inside the (K1, M, 16) packed operand."""
    return (i_k // 16) * (m * 16) + i_m * 16 + (i_k % 16)


def pack_nk_weight_index(i_n: int, i_k: int, n: int, k: int) -> int:
    """Flat index of W[i_n, i_k] inside the (K1, N, 16) packed weight."""
    return (i_k // 16) * (n * 16) + i_n * 16 + (i_k % 16)


def unpack_mn_index(i_m: int, i_n: int, m: int, n: int) -> int:
    """Flat index of C[i_m, i_n] inside the (N1, M, 16) packed output."""
    return (i_n // 16) * (m * 16) + i_m * 16 + (i_n % 16)


def bias_group_slice(i_n: int) -> int:
    """Element offset of the 16-value bias group that column ``i_n`` belongs to."""
    return (i_n // 16) * 16


# --- lazy TIK side ---------------------------------------------------------
def _tik():
    """Import TIK lazily so this module is importable without CANN/torch."""
    from tbe import tik  # noqa: PLC0415  (deliberately deferred)

    return tik


def set_compile_soc(soc: str = PINNED_COMPILE_SOC) -> str:
    """Pin the compile SoC and assert the verified hardware profile.

    The default SoC on this install resolves to a wrong instruction table, and
    Round82's live DCMI audit showed 30 usable cores on all 8 chips (32 GiB is
    memory, not core count), so the compile profile must be the 30-core
    ``Ascend910B`` one.  Nothing here touches the bias gate: this kernel does
    not pass a bias operand to ``matmul`` at all.
    """
    import tbe.common.platform as platform  # noqa: PLC0415

    platform.set_current_compile_soc_info(soc)
    resolved = platform.get_soc_spec("FULL_SOC_VERSION")
    if resolved != soc:
        raise BiasKernelError(f"compile SoC is {resolved!r}, expected {soc!r}")
    core_num = int(platform.get_soc_spec("CORE_NUM"))
    if core_num != EXPECTED_CORE_NUM:
        raise BiasKernelError(
            f"compile profile reports {core_num} cores, this device has "
            f"{EXPECTED_CORE_NUM}; refusing to emit a mismatched kernel")
    return resolved


def _validate(variant: str, m: int, k: int, n: int) -> dict:
    if variant not in VARIANTS:
        raise BiasKernelError(f"unknown variant {variant!r}; expected {sorted(VARIANTS)}")
    if (m, k, n) != (MICRO_M, MICRO_K, MICRO_N):
        raise BiasKernelError(
            f"micro geometry is fixed to {MICRO_M}x{MICRO_K}x{MICRO_N}; got {m}x{k}x{n}. "
            "The H3 shapes need blocked tiling, which does not exist yet")
    if n % 16 or k % 16 or m % 16:
        raise BiasKernelError("m, k, n must be multiples of 16 for this tile")
    return VARIANTS[variant]


def build_plan(variant: str = "fp32_bias_fp16_out") -> dict:
    """Pure description of what would be built; needs no CANN and no device."""
    spec = _validate(variant, MICRO_M, MICRO_K, MICRO_N)
    m, k, n = MICRO_M, MICRO_K, MICRO_N
    n1 = n // 16
    return {
        "variant": variant,
        "dst_dtype": spec["dst_dtype"],
        "adds_fp32_bias": spec["add_bias"],
        "single_final_half_cast": spec["cast_final"],
        "geometry": {"m": m, "k": k, "n": n, "n1": n1},
        "byte_sizes": {
            "a_gm_float16": m * k * 2,
            "b_gm_float16": n * k * 2,
            "bias_gm_float32": n * 4,
            "aux_gm_float16": 16 * 2,
            "out_gm_half": m * n * 2,
            "ub_fp32_accumulator": m * n * 4,
            "ub_half_result": m * n * 2,
        },
        "gm_argument_order": list(GM_ARG_ORDER),
        "gm_argument_count": 5,
        "aux_argument_used_by_math": False,
        "matmul_bias_operand_used": False,
        "compile_soc": PINNED_COMPILE_SOC,
        "expected_core_num": EXPECTED_CORE_NUM,
        "vector_add": {
            "mask": 16,           # 16 FP32 elements = 64 B = 2 blocks
            "repeat_times": m,    # one repeat per output row of the group
            "dst_rep_stride": 2,  # blocks (32 B) -> next row, 16 floats
            "src0_rep_stride": 2,
            "src1_rep_stride": 0,  # same 16 bias values for every row
            "groups": n1,
            "in_place": True,
        },
        "final_cast": {"api": 'h_cast(dst, src, "none")', "round_mode": "none"},
        "max_vector_offset_elements_for_last_group": m * 16,
        "ub_tensor_elements": n1 * m * 16,
        "executed": False,
        "proves_full_encoder_precision": False,
        "proves_speed": False,
    }


def build_kernel(variant: str = "fp32_bias_fp16_out", m: int = MICRO_M,
                 k: int = MICRO_K, n: int = MICRO_N, out_dir="kernels") -> dict:
    """Build one micro kernel and return its artefacts and metadata.

    Lazy: TIK is imported only here, so importing this module for tests or for
    the plan needs no CANN on the machine.
    """
    spec = _validate(variant, m, k, n)
    dst_dtype = spec["dst_dtype"]
    add_bias = spec["add_bias"]
    cast_final = spec["cast_final"]
    n1 = n // 16

    tik = _tik()
    soc = set_compile_soc()

    name = f"tik_bias83_{variant}_m{m}_k{k}_n{n}"
    t = tik.Tik(disable_debug=True)

    # ---- GM parameters: exactly five, in the documented order ----
    a_gm = t.Tensor("float16", (k // 16, m, 16), name="a_gm", scope=tik.scope_gm)
    b_gm = t.Tensor("float16", (k // 16, n, 16), name="b_gm", scope=tik.scope_gm)
    bias_gm = t.Tensor("float32", (n,), name="bias_gm", scope=tik.scope_gm)
    aux_gm = t.Tensor("float16", (16,), name="aux_gm", scope=tik.scope_gm)
    out_gm = t.Tensor(dst_dtype, (n1, m, 16), name="out_gm", scope=tik.scope_gm)

    with t.for_range(0, 1, block_num=1) as _core:
        a_l1 = t.Tensor("float16", (k // 16, m, 16), name="a_l1", scope=tik.scope_cbuf)
        b_l1 = t.Tensor("float16", (k // 16, n, 16), name="b_l1", scope=tik.scope_cbuf)
        acc_l0c = t.Tensor("float32", (n1, m, 16), name="acc_l0c", scope=tik.scope_cc)
        # FP32 UB staging buffer: the accumulator leaves L0C without narrowing.
        c32 = t.Tensor("float32", (n1, m, 16), name="c32_ub", scope=tik.scope_ubuf)
        bias_ub = t.Tensor("float32", (n,), name="bias_ub", scope=tik.scope_ubuf)

        # bursts are in 32 B blocks for data_move, 1024 B blocks for tensor_mov
        t.data_move(a_l1, a_gm, 0, 1, m * k * 2 // 32, 0, 0)
        t.data_move(b_l1, b_gm, 0, 1, n * k * 2 // 32, 0, 0)
        t.data_move(bias_ub, bias_gm, 0, 1, n * 4 // 32, 0, 0)

        # No bias operand: that path is rejected for Ascend910 before compilation.
        t.matmul(acc_l0c, a_l1, b_l1, m, k, n, init_l1out=True, bias=None)

        # L0C FP32 -> UB FP32.  No conversion here: all accumulator bits survive.
        t.tensor_mov(c32, acc_l0c, "m", 1, m * n * 4 // 1024, 0, 0)

        if add_bias:
            # One vec_add per 16-column group: add the same 16 FP32 bias values
            # to every one of the m rows.  mask=16 FP32 elements (2 blocks);
            # rep_stride=2 blocks advances exactly one packed row.
            for group in range(n1):
                t.vec_add(16,
                          c32[group, :, :],           # dst   (in place)
                          c32[group, :, :],           # src0
                          bias_ub[group * 16],        # src1  (16 FP32 bias values)
                          m, 2, 2, 0)

        if cast_final:
            c16 = t.Tensor("float16", (n1, m, 16), name="c16_ub", scope=tik.scope_ubuf)
            # Exactly one FP32 -> FP16 conversion, after the bias add.
            t.h_cast(c16, c32, "none")
            t.data_move(out_gm, c16, 0, 1, m * n * 2 // 32, 0, 0)
        else:
            t.data_move(out_gm, c32, 0, 1, m * n * 4 // 32, 0, 0)

    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tik_bias83_build_", dir=root)).resolve()
    if work.parent != root or not work.name.startswith("tik_bias83_build_"):
        raise BiasKernelError("unexpected build workspace")
    previous = os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=name, inputs=[a_gm, b_gm, bias_gm, aux_gm], outputs=[out_gm])
    finally:
        os.chdir(previous)

    obj = work / "kernel_meta" / (name + ".o")
    meta = work / "kernel_meta" / (name + ".json")
    if not obj.is_file() or not meta.is_file():
        raise BiasKernelError(f"build produced no kernel_meta artefacts for {name}")
    data = json.loads(meta.read_text())
    if data.get("blockDim") != 1:
        raise BiasKernelError(f"unexpected blockDim {data.get('blockDim')!r}, expected 1")

    return {
        "variant": variant,
        "dst_dtype": dst_dtype,
        "uses_fp32_bias": add_bias,
        "single_final_half_cast": cast_final,
        "geometry": {"m": m, "k": k, "n": n, "n1": n1},
        "compile_soc": soc,
        "kernel_name": name,
        "kernel_symbol": data.get("kernelName"),
        "block_dim": 1,
        "core_type": data.get("coreType"),
        "obj_path": str(obj),
        "obj_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
        "json_path": str(meta),
        "gm_argument_order": list(GM_ARG_ORDER),
        "gm_argument_count": 5,
        "unused_aux_inputs": 1,
        "matmul_bias_operand_used": False,
        "input_layout": "K1_M_16",
        "weight_layout": "K1_N_16",
        "output_layout": "N1_M_16",
        "bias_dtype": "float32" if add_bias else None,
        "bias_values_must_be_exact_original_half": add_bias,
        "built": True,
        "executed": False,
        "proves_full_encoder_precision": False,
        "proves_speed": False,
    }