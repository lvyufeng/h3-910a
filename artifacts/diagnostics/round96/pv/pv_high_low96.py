#!/usr/bin/env python3
"""Round96 fused two-Half-probability P@V kernel: one launch, one Half store.

SOURCE ONLY. Nothing in this module has been compiled or executed by its author:
no SSH, no NPU/GPU work, no CANN build, no torch import, no model run, no
installation, no production edit, no commit. Building a kernel says nothing about
its numbers, its speed or its precision. Every runtime receipt belongs to the
root agent that compiles and launches it.

WHY THIS KERNEL EXISTS
----------------------
Round88's accurate vision attention splits the Float32 softmax probability into

    P_hi = Half(P)                      (Half, exact rounding)
    P_lo = Half((P - Float(P_hi)) * 1024)

and then runs TWO separate Round85 batched GEMM launches, each producing its own
FP32 GM tensor, and combines them on the host as
``out = pv_hi + pv_lo * (1/1024)`` before ONE Half cast. That is arithmetically
right but costs, per query chunk: two kernel launches, two FP32 GM outputs, two
host unpack/permute/reshape copies, a Float32 multiply and add, and one Half cast.

This kernel does the same arithmetic in ONE launch: two independent FP32 L0C
accumulators run the full K contraction (P_hi@V and P_lo@V), then the low product
is moved to a Float32 UB tile first, scaled by 1/1024, added to the Float32 high
tile, and the sum is narrowed exactly once to Half and stored. No FP32 GM output
for the products, no host combine.

ARITHMETIC (unchanged from the executed Round88 path)
-----------------------------------------------------
* Operands are the original Half activations: no Float32 weight shadow, no BF16
  gold cast, no Float32 input promotion and no teacher.
* Each product keeps ONE FP32 L0C accumulator live across every K block: no Half
  partial and no FP32-UB partial sum.
* The combine is ``Float(t_hi) + Float(t_lo) * (1/1024)`` in Float32 followed by a
  single ``h_cast('none')``. 1/1024 is a power of two, so the scale is exact for
  every normal value; the add is one IEEE Float32 addition.
* The only difference from Round88 is *where* those two Float32 operations run
  (vector unit instead of host) and that they happen before any narrowing.

LAYOUTS (identical to the executed Round85 convention)
------------------------------------------------------
* hi A GM: ``(H, K1, Mp, 16)`` float16, ``(h,k1,m,k0) = P_hi[h, m, k1*16+k0]``
* lo A GM: same shape/dtype for ``P_lo``
* V  B GM: ``(H, K1, N, 16)``  float16, ``(h,k1,n,k0) = V[h, n, k1*16+k0]``
  (i.e. the host passes ``V`` transposed to ``(H, N, K)`` and packed as B)
* out C GM: ``(H, N1, Mp, 16)`` float16, ``(h,n1,m,n0) = out[h, m, n1*16+n0]``

``Mp = ceil(M/128)*128``; rows ``M..Mp-1`` of both A operands are zero, so the
padded rows produce exact zero output rows and the host crops them. **Every head
owns a distinct V matrix and distinct A matrices**; the head stride is part of
every GM offset, so no head can read another head's operands.

GM ARGUMENT ABI (5 pointers, the proven fa910a five-GM launcher)
--------------------------------------------------------------
``[hi_Half, lo_Half, V_Half, aux_Float32_16_unused, out_Half]``

``aux`` is an arithmetic placeholder that only exists because the verified
launcher always passes five GM pointers; the kernel never reads it. It is
declared as a float32 GM tensor and passed explicitly to ``BuildCCE`` so
automatic operand pruning cannot drop it.

TARGET GEOMETRY (real H3 vision attention, per query chunk)
-----------------------------------------------------------
``H=16`` heads, ``K=4096`` (true 4032 keys + 64 zero padding), ``N=128``
(true head_dim 72 + 56 zero padding), ``M=512`` full chunk and ``M=448`` tail
(both pad to ``Mp=512``). Tasks: ``16 * 4 * 1 = 64`` over ``30`` physical cores.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

# --- pinned device contract (identical to the executed Round83/85 kernels) ---
PINNED_COMPILE_SOC = "Ascend910B"
PINNED_RUNTIME_SOC = "Ascend910B"
PINNED_CORE_NUM = 30
PINNED_L1_SIZE = 1048576
PINNED_L0C_SIZE = 262144
PINNED_MAX_CORES = 30
PINNED_UB_SIZE = 196608        # conservative floor: 192 KiB
#: Two Float32 tiles (2 x 64 KiB) + one Half tile (32 KiB) for the real shape.
PINNED_UB_MIN_BYTES = 163840

#: Round85 batched GEMM whose proven conventions this file reuses (read-only).
PINNED_R85_SOURCE = "batch_gemm85.py"
PINNED_R85_SHA256 = "546cd211ddb3938c3d8dac9d5207a79646c1458a3887df7e8c8562426069854a"
REFERENCE_RUNTIME_MODULE = "runtime5gm83"
REFERENCE_RUNTIME_MODULE_SHA256 = (
    "c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6")
REFERENCE_RUNTIME_LIBRARY_SHA256 = (
    "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043")

DEFAULT_MB, DEFAULT_NB, DEFAULT_KB = 128, 128, 128

#: ``P = P_hi + P_lo / LOW_SCALE``; 1024 is a power of two, so the downscale is
#: exact for normal Float32 values and matches the executed Round88 host combine.
LOW_SCALE = 1024.0
LOW_DOWNSCALE = 1.0 / LOW_SCALE

#: Proven Float32 vector shape (executed Round90 erf kernel): 64 elements per
#: repeat, 8 x 32 B = 256 B = 64 floats between consecutive repeats.
FP32_MASK = 64
FP32_REP_STRIDE = 8
FP32_REP_ELEMS = FP32_MASK

#: Real vision P@V geometries: (H, M, K, N). K/N are the padded tile multiples.
TARGET_SHAPES = {
    "pv_full_chunk": (16, 512, 4096, 128),
    "pv_tail_chunk": (16, 448, 4096, 128),
}

#: Smallest geometry that proves two-product K accumulation and per-head V
#: distinctness without touching the model (root-side smoke compile only).
SUGGESTED_SMOKE_GEOMETRY = (2, 128, 256, 128)

KERNEL_ARG_ORDER = ["hi_gm_float16", "lo_gm_float16", "v_gm_float16",
                    "aux_gm_float32", "out_gm_float16"]


class FusedPvError(RuntimeError):
    """Raised instead of guessing when a shape or the environment is unexpected."""


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Pure-python tiling plan (stdlib only, unit-testable without TIK or torch)
# ---------------------------------------------------------------------------


def plan_tiling(batch: int, m: int, k: int, n: int, mb: int = DEFAULT_MB,
                nb: int = DEFAULT_NB, kb: int = DEFAULT_KB) -> dict:
    """Return every number the kernel, the host packing and the tests must agree on."""
    for name, value in (("batch", batch), ("m", m), ("k", k), ("n", n),
                        ("mb", mb), ("nb", nb), ("kb", kb)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise FusedPvError(f"{name} must be a python int, got {type(value).__name__}")
        if value <= 0:
            raise FusedPvError(f"{name} must be positive, got {value}")
    for name, value, unit in (("mb", mb, 16), ("nb", nb, 16), ("kb", kb, 16)):
        if value % unit:
            raise FusedPvError(f"{name}={value} must be a multiple of {unit}")
    if k % kb:
        raise FusedPvError(f"k={k} is not a multiple of the K tile kb={kb}")
    if n % nb:
        raise FusedPvError(f"n={n} is not a multiple of the N tile nb={nb}")
    group_elems = mb * 16
    if group_elems % FP32_MASK:
        raise FusedPvError("one N-group row block must be a whole number of Float32 repeats")

    m_pad = _ceil_to(m, mb)
    k1, n1 = k // 16, n // 16
    m_tiles, n_tiles = m_pad // mb, n // nb
    per_head_tasks = m_tiles * n_tiles
    tasks = batch * per_head_tasks
    cores = min(PINNED_MAX_CORES, tasks)
    steps = (tasks + cores - 1) // cores
    k_tiles = k // kb
    kb1 = kb // 16

    # Per-head dense GM strides in elements. ``hi`` and ``lo`` share the A layout.
    head_stride = {"hi": k1 * m_pad * 16, "lo": k1 * m_pad * 16,
                   "b": k1 * n * 16, "out": n1 * m_pad * 16}
    gm_shapes = {
        "hi": [batch, k1, m_pad, 16],
        "lo": [batch, k1, m_pad, 16],
        "b": [batch, k1, n, 16],
        "out": [batch, n1, m_pad, 16],
        "aux": [16],
        "aux_dtype": "float32",
    }
    gm_elems = {name: batch * head_stride[name] for name in ("hi", "lo", "b", "out")}
    gm_bytes = {"hi": gm_elems["hi"] * 2, "lo": gm_elems["lo"] * 2,
                "b": gm_elems["b"] * 2, "out": gm_elems["out"] * 2,
                "aux": 16 * 4}

    # DMA descriptions are per-head and identical to the executed Round83/85
    # convention: 32-byte units. A full K0=16 Half row is exactly one unit; the
    # Half output row is also one unit (unlike the FP32 output's two units).
    dma = {
        "a_load": {"nburst": kb1, "burst": mb, "src_stride": m_pad - mb, "dst_stride": 0},
        "b_load": {"nburst": kb1, "burst": nb, "src_stride": n - nb, "dst_stride": 0},
        "l0c_to_ub": {"block_mode": "m", "nburst": 1,
                      "burst_len_kb": mb * nb * 4 // 1024},
        "out_store": {"nburst": nb // 16, "burst": mb, "src_stride": 0,
                      "dst_stride": m_pad - mb},
    }
    float_tile = (nb // 16) * mb * 16 * 4
    half_tile = (nb // 16) * mb * 16 * 2
    buffers = {
        "l1_a_hi_bytes": kb1 * mb * 16 * 2,
        "l1_a_lo_bytes": kb1 * mb * 16 * 2,
        "l1_b_bytes": kb1 * nb * 16 * 2,
        "l1_total_bytes": 2 * kb1 * mb * 16 * 2 + kb1 * nb * 16 * 2,
        "l0c_high_bytes": float_tile,
        "l0c_low_bytes": float_tile,
        "l0c_total_bytes": 2 * float_tile,
        "ub_high_bytes": float_tile,
        "ub_low_bytes": float_tile,
        "ub_half_bytes": half_tile,
        "ub_total_bytes": 2 * float_tile + half_tile,
    }
    vector = {
        "mask": FP32_MASK,
        "rep_stride": FP32_REP_STRIDE,
        "elements_per_repeat": FP32_REP_ELEMS,
        "bytes_per_repeat": FP32_REP_ELEMS * 4,
        "chunks_per_tile": nb // 16,
        "elements_per_chunk": group_elems,
        "repeats_per_chunk": group_elems // FP32_MASK,
        "max_repeat_immediate": 255,
        "chunk_split_reason": "the Float32 tile is 16384 elements; one call would need 256 "
                              "repeats, above the 8-bit repeat immediate, so each static "
                              "chunk stays at 32 repeats",
        "float_vector_calls": 2 * (nb // 16),
        "h_cast_calls": 1,
        "scale_op": "vec_muls(low chunk, LOW_DOWNSCALE) in place",
        "add_op": "vec_add(high chunk, high chunk, low chunk) in place, Float32",
        "order": ["tensor_mov_low_L0C_to_UB", "tensor_mov_high_L0C_to_UB",
                  "vec_muls_low_downscale", "vec_add_high_plus_scaled_low",
                  "h_cast_once_none", "data_move_half_store"],
        "scale_is_power_of_two": True,
    }

    return {
        "logical": {"batch": batch, "m": m, "k": k, "n": n},
        "tiles": {"mb": mb, "nb": nb, "kb": kb},
        "padded": {"m_pad": m_pad, "m_pad_rows": m_pad - m, "k1": k1, "n1": n1},
        "grid": {"m_tiles": m_tiles, "n_tiles": n_tiles,
                 "per_head_tasks": per_head_tasks, "tasks": tasks,
                 "cores": cores, "steps": steps},
        "head_stride_elems": head_stride,
        "gm_shapes": gm_shapes,
        "gm_elems": gm_elems,
        "gm_bytes": gm_bytes,
        "dma": dma,
        "buffers": buffers,
        "vector": vector,
        "low_scale": LOW_SCALE,
        "low_downscale": LOW_DOWNSCALE,
        "accumulation": {
            "k_tiles": k_tiles,
            "products": ["high", "low"],
            "independent_l0c_per_product": True,
            "single_l0c_across_all_k_blocks": True,
            "init_first_k_block_only": True,
            "half_partial": False,
            "fp32_ub_partial_sum": False,
            "mid_product_gm_output": False,
            "host_combine_required": False,
            "final_output_dtype": "float16",
            "one_half_cast": True,
        },
        "batched": {
            "per_head_distinct_b": True,
            "per_head_distinct_hi_and_lo_a": True,
            "head_offset_applied_to_all_gms": True,
            "task_flatten": "head*(m_tiles*n_tiles) + m_tile*n_tiles + n_tile",
            "b_l1_reused_by_both_products": True,
        },
        "kernel_arg_order": KERNEL_ARG_ORDER,
        "limits": {"l1_size": PINNED_L1_SIZE, "l0c_size": PINNED_L0C_SIZE,
                   "ub_size": PINNED_UB_SIZE, "max_cores": PINNED_MAX_CORES},
    }


def _check_budget(plan: dict) -> None:
    buf = plan["buffers"]
    if buf["l1_total_bytes"] > PINNED_L1_SIZE:
        raise FusedPvError(f"L1 need {buf['l1_total_bytes']} B exceeds {PINNED_L1_SIZE} B")
    if buf["l0c_total_bytes"] > PINNED_L0C_SIZE:
        raise FusedPvError(f"L0C need {buf['l0c_total_bytes']} B exceeds {PINNED_L0C_SIZE} B")
    if buf["ub_total_bytes"] > PINNED_UB_SIZE:
        raise FusedPvError(f"UB need {buf['ub_total_bytes']} B exceeds {PINNED_UB_SIZE} B")
    if plan["vector"]["repeats_per_chunk"] > plan["vector"]["max_repeat_immediate"]:
        raise FusedPvError("vector repeat count exceeds the 8-bit repeat immediate")


def flatten_task(task: int, batch: int, n_tiles: int, per_head_tasks: int) -> tuple:
    """``task -> (head, m_tile, n_tile)`` for the round-robin flattened grid."""
    if not isinstance(task, int) or isinstance(task, bool):
        raise FusedPvError(f"task must be a python int, got {type(task).__name__}")
    total = batch * per_head_tasks
    if not 0 <= task < total:
        raise FusedPvError(f"task {task} outside [0, {total})")
    head, local = divmod(task, per_head_tasks)
    m_tile, n_tile = divmod(local, n_tiles)
    return head, m_tile, n_tile


def gm_offsets(plan: dict, head: int, m_tile: int, n_tile: int, kt: int) -> dict:
    """Absolute GM element offsets for one (head, tile, K block); head stride included."""
    gr, pd, tl = plan["grid"], plan["padded"], plan["tiles"]
    if not 0 <= head < plan["logical"]["batch"]:
        raise FusedPvError(f"head {head} outside batch {plan['logical']['batch']}")
    if not 0 <= m_tile < gr["m_tiles"]:
        raise FusedPvError(f"m_tile {m_tile} outside {gr['m_tiles']}")
    if not 0 <= n_tile < gr["n_tiles"]:
        raise FusedPvError(f"n_tile {n_tile} outside {gr['n_tiles']}")
    if not 0 <= kt < plan["accumulation"]["k_tiles"]:
        raise FusedPvError(f"kt {kt} outside K tiles")
    mb, nb, kb1 = tl["mb"], tl["nb"], tl["kb"] // 16
    m_pad, n, n1 = pd["m_pad"], plan["logical"]["n"], pd["n1"]
    base_m, base_n = m_tile * mb, n_tile * nb
    hs = plan["head_stride_elems"]
    a_off = (kt * kb1 * m_pad + base_m) * 16
    return {
        "hi": head * hs["hi"] + a_off,
        "lo": head * hs["lo"] + a_off,
        "b": head * hs["b"] + (kt * kb1 * n + base_n) * 16,
        "out": head * hs["out"] + ((base_n // 16) * m_pad + base_m) * 16,
        "out_elements": (nb // 16) * mb * 16,
    }


# ---------------------------------------------------------------------------
# Environment / provenance guards (lazy: importing needs no CANN and no torch)
# ---------------------------------------------------------------------------


def capability_report(ub_bytes_needed=None) -> dict:
    """Query the facts this kernel depends on, without building anything."""
    import tbe.common.platform as platform

    platform.set_current_compile_soc_info(PINNED_COMPILE_SOC)
    resolved = platform.get_soc_spec("FULL_SOC_VERSION")
    if resolved != PINNED_COMPILE_SOC:
        raise FusedPvError(f"compile SoC resolved to {resolved!r}, expected "
                           f"{PINNED_COMPILE_SOC!r}; refusing to emit a different ISA")
    report = {
        "compile_soc": resolved,
        "runtime_expected_soc": PINNED_RUNTIME_SOC,
        "Intrinsic_mmad_f162f32": bool(
            platform.intrinsic_check_support("Intrinsic_mmad", "f162f32")),
        "core_num": int(platform.get_soc_spec("CORE_NUM")),
        "l1_size": int(platform.get_soc_spec("L1_SIZE")),
        "l0c_size": int(platform.get_soc_spec("L0C_SIZE")),
    }
    try:
        report["ub_size"] = int(platform.get_soc_spec("UB_SIZE"))
    except Exception:  # noqa: BLE001 - older SDKs may not expose UB_SIZE
        report["ub_size"] = None
    if not report["Intrinsic_mmad_f162f32"]:
        raise FusedPvError("Intrinsic_mmad/f162f32 unsupported: the FP32 accumulator this "
                           "kernel needs is unavailable and must not be faked")
    if report["core_num"] != PINNED_CORE_NUM:
        raise FusedPvError(
            f"compiler CORE_NUM is {report['core_num']}, expected {PINNED_CORE_NUM}")
    if report["l1_size"] != PINNED_L1_SIZE or report["l0c_size"] != PINNED_L0C_SIZE:
        raise FusedPvError("L1/L0C sizes do not match the pinned 910A-generation part")
    need = PINNED_UB_MIN_BYTES if ub_bytes_needed is None else int(ub_bytes_needed)
    if report["ub_size"] is not None and report["ub_size"] < need:
        raise FusedPvError(
            f"UB_SIZE {report['ub_size']} B is too small for the {need} B this kernel needs")
    return report


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reference_sources(r85_path=None, runtime_module_path=None, strict=True) -> dict:
    """Verify the Round85 conventions and the five-GM bridge this file targets."""
    status = {}
    here = Path(__file__).resolve().parent
    if r85_path is None:
        for candidate in (here / PINNED_R85_SOURCE, here.parent.parent / "round85"
                          / "batch" / PINNED_R85_SOURCE):
            if candidate.is_file():
                r85_path = candidate
                break
    if r85_path is not None and Path(r85_path).is_file():
        actual = _sha256_file(r85_path)
        status["reference_r85_source"] = str(r85_path)
        status["reference_r85_source_sha256"] = actual
        if actual != PINNED_R85_SHA256:
            raise FusedPvError(f"{PINNED_R85_SOURCE} changed: {actual} != "
                               f"{PINNED_R85_SHA256}")
    elif strict:
        raise FusedPvError(f"cannot locate {PINNED_R85_SOURCE} to verify its pin")

    if runtime_module_path is None:
        for candidate in (here / "runtime5gm83.py",):
            if candidate.is_file():
                runtime_module_path = candidate
                break
    if runtime_module_path is not None and Path(runtime_module_path).is_file():
        actual = _sha256_file(runtime_module_path)
        status["runtime_module"] = str(runtime_module_path)
        status["runtime_module_sha256"] = actual
        if actual != REFERENCE_RUNTIME_MODULE_SHA256:
            raise FusedPvError(f"{REFERENCE_RUNTIME_MODULE} changed: {actual} != "
                               f"{REFERENCE_RUNTIME_MODULE_SHA256}")
    elif strict:
        status["runtime_module_missing"] = True
    status["runtime_library_sha256"] = REFERENCE_RUNTIME_LIBRARY_SHA256
    status["proven_abi"] = "fiveGMpointers"
    return status


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------


def kernel_name_for(plan: dict) -> str:
    t, g, lg = plan["tiles"], plan["grid"], plan["logical"]
    return (f"pv_hi_lo96_h{lg['batch']}_m{lg['m']}_k{lg['k']}_n{lg['n']}"
            f"_mb{t['mb']}_nb{t['nb']}_kb{t['kb']}_c{g['cores']}")


def build_kernel(batch: int, m: int, k: int, n: int, out_dir=None,
                 mb: int = DEFAULT_MB, nb: int = DEFAULT_NB,
                 kb: int = DEFAULT_KB) -> dict:
    """Build (or reuse) one fused-PV kernel and describe it; never overwrite."""
    plan = plan_tiling(batch, m, k, n, mb=mb, nb=nb, kb=kb)
    _check_budget(plan)

    from tbe import tik

    capability = capability_report(ub_bytes_needed=plan["buffers"]["ub_total_bytes"])
    name = kernel_name_for(plan)
    out_root = Path(out_dir).resolve() if out_dir is not None else \
        Path.cwd() / "pv_kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec96.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability)

    with (out_root / ".compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan,
                                    capability)

        work = Path(tempfile.mkdtemp(prefix="pv96_build_", dir=out_root)).resolve()
        if work.parent != out_root:
            raise FusedPvError("unexpected build workspace")
        t = tik.Tik(disable_debug=True)
        gm = _emit_kernel(t, plan, tik)
        previous = os.getcwd()
        try:
            os.chdir(work)
            # All five GM operands are explicit: ``aux`` is an unused placeholder
            # that keeps the proven five-pointer ABI, and auto-collection would
            # prune it if it were omitted from the inputs list.
            t.BuildCCE(kernel_name=name,
                       inputs=[gm["hi"], gm["lo"], gm["v"], gm["aux"]],
                       outputs=[gm["out"]])
        finally:
            os.chdir(previous)
        built_obj = work / "kernel_meta" / (name + ".o")
        built_meta = work / "kernel_meta" / (name + ".json")
        if not built_obj.is_file() or not built_meta.is_file():
            raise FusedPvError(f"BuildCCE produced no kernel_meta artefacts for {name}")
        if dest.exists():
            raise FusedPvError(f"refusing to overwrite existing kernel directory {dest}")
        dest.mkdir(parents=True)
        import shutil

        shutil.copyfile(built_obj, obj)
        shutil.copyfile(built_meta, meta_path)

    return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                            compiled_now=True)


def _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                     compiled_now=False) -> dict:
    meta = json.loads(meta_path.read_text())
    if int(meta.get("blockDim", -1)) != plan["grid"]["cores"]:
        raise FusedPvError(
            f"cached kernel {name} has blockDim {meta.get('blockDim')} but the plan needs "
            f"{plan['grid']['cores']} cores; refusing to reuse a mismatched binary")
    spec = {
        "round": 96,
        "scope": ("fused two-Half-probability P@V: two independent FP32 L0C accumulations "
                  "over the full K axis, Float32 UB combine, one Half cast, one launch"),
        "executed": False,
        "compiled_now": compiled_now,
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "kernel_name": name,
        "kernel_symbol": meta["kernelName"],
        "block_dim": int(meta["blockDim"]),
        "core_type": meta.get("coreType"),
        "obj_path": str(obj),
        "obj_sha256": _sha256_file(obj),
        "json_path": str(meta_path),
        "source_sha256": _sha256_file(Path(__file__)),
        "reference_r85_source_sha256": PINNED_R85_SHA256,
        "reference_runtime_module_sha256": REFERENCE_RUNTIME_MODULE_SHA256,
        "reference_runtime_library_sha256": REFERENCE_RUNTIME_LIBRARY_SHA256,
        "dtype": "float16_in_float16_out_two_float32_accumulators",
        "dst_dtype": "float16",
        "input_dtype": "float16",
        "accumulator_dtype": "float32",
        "output_dtype": "float16",
        "kernel_arg_order": plan["kernel_arg_order"],
        "GM_argument_count": 5,
        "fused_products": ["high", "low"],
        "low_scale_1024": LOW_SCALE,
        "low_downscale": LOW_DOWNSCALE,
        "single_final_half_cast": True,
        "one_kernel_launch_per_tile": True,
        "no_mid_product_fp32_gm": True,
        "no_host_combine": True,
        "uses_fp32_bias": False,
        "no_bias_operand": True,
        "no_fp32_weight_shadow": True,
        "no_half_partial": True,
        "logical_shape": plan["logical"],
        "batch": plan["logical"]["batch"],
        "padded_shape": {"m_pad": plan["padded"]["m_pad"], "k1": plan["padded"]["k1"],
                         "n1": plan["padded"]["n1"]},
        "tiles": plan["tiles"],
        "grid": plan["grid"],
        "head_stride_elems": plan["head_stride_elems"],
        "gm_shapes": {key: value for key, value in plan["gm_shapes"].items()},
        "gm_elems": plan["gm_elems"],
        "gm_bytes": plan["gm_bytes"],
        "dma": plan["dma"],
        "buffers": plan["buffers"],
        "vector_plan": plan["vector"],
        "accumulation": plan["accumulation"],
        "batched": plan["batched"],
        "capability": capability,
        "suggested_smoke_geometry": list(SUGGESTED_SMOKE_GEOMETRY),
        "host_packing_required": ["pack_a(hi)", "pack_a(lo)", "pack_w(V^T)"],
        "host_unpacking_required": ["unpack_out_half"],
        "aux_operand": {"role": "unused five-GM ABI placeholder", "dtype": "float32",
                        "elements": 16, "read_by_kernel": False},
        "host_padding": {
            "m_pad_rows": plan["padded"]["m_pad_rows"],
            "a_zero_m_padding": True,
            "out_pad_rows_cropped_by_host": True,
            "key_padding": "host zero-fills padded key rows; the kernel has no mask",
            "head_dim_padding": "host zero-fills columns 72..127 for both operands",
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec


def _emit_kernel(t, plan: dict, tik) -> dict:
    """Emit the TIK program. Every offset and stride here is derived from ``plan``.

    Review notes:

    * Two independent FP32 L0C accumulators (``c_hi``, ``c_lo``), each initialised
      on the first K block and accumulated across every K block: no Half partial
      and no FP32-UB partial sum.
    * The V L1 tile is loaded once per K block and read by both matmuls.
    * ``tensor_mov`` L0C->UB is issued for the LOW product first, then the HIGH
      product, matching the reviewed combine order; the Float32 scale and add are
      vector ops on the UB tiles, and only one ``h_cast('none')`` narrows.
    * Only the outer tile-step loop is dynamic; the K loop is a static Python loop
      because ``init_l1out`` must be a Python bool (the SDK type-checks it).
    * Half output rows are 32 B, so the store stride is ``m_pad - mb`` rows with no
      FP32 factor; padded M rows are cropped by the host.
    """
    lg, tl, pd, gr, dma = (plan["logical"], plan["tiles"], plan["padded"],
                           plan["grid"], plan["dma"])
    vec = plan["vector"]
    batch, m_pad, n, k1, n1 = lg["batch"], pd["m_pad"], lg["n"], pd["k1"], pd["n1"]
    mb, nb, kb = tl["mb"], tl["nb"], tl["kb"]
    kb1 = kb // 16
    k_tiles = plan["accumulation"]["k_tiles"]
    cores, steps, tasks = gr["cores"], gr["steps"], gr["tasks"]
    n_tiles, per_head_tasks = gr["n_tiles"], gr["per_head_tasks"]
    hs = plan["head_stride_elems"]
    if per_head_tasks != gr["m_tiles"] * n_tiles:
        raise FusedPvError("per-head task count disagrees with the tile grid")

    # Declared exactly like the executed Round85 batch kernel (full batch extent),
    # with the per-head stride applied explicitly in every offset below. A flat
    # scalar index into a multi-dim GM tensor is the proven address form here.
    hi_gm = t.Tensor("float16", (batch, k1, m_pad, 16), name="hi_gm", scope=tik.scope_gm)
    lo_gm = t.Tensor("float16", (batch, k1, m_pad, 16), name="lo_gm", scope=tik.scope_gm)
    v_gm = t.Tensor("float16", (batch, k1, n, 16), name="v_gm", scope=tik.scope_gm)
    aux_gm = t.Tensor("float32", (16,), name="aux_gm", scope=tik.scope_gm)
    out_gm = t.Tensor("float16", (batch, n1, m_pad, 16), name="out_gm", scope=tik.scope_gm)

    with t.for_range(0, cores, block_num=cores) as core:
        a_hi_l1 = t.Tensor("float16", (kb1, mb, 16), name="a_hi_l1", scope=tik.scope_cbuf)
        a_lo_l1 = t.Tensor("float16", (kb1, mb, 16), name="a_lo_l1", scope=tik.scope_cbuf)
        b_l1 = t.Tensor("float16", (kb1, nb, 16), name="b_l1", scope=tik.scope_cbuf)
        c_hi = t.Tensor("float32", (nb // 16, mb, 16), name="c_hi_l0c", scope=tik.scope_cc)
        c_lo = t.Tensor("float32", (nb // 16, mb, 16), name="c_lo_l0c", scope=tik.scope_cc)
        u_hi = t.Tensor("float32", (nb // 16, mb, 16), name="u_hi_ub", scope=tik.scope_ubuf)
        u_lo = t.Tensor("float32", (nb // 16, mb, 16), name="u_lo_ub", scope=tik.scope_ubuf)
        u_h = t.Tensor("float16", (nb // 16, mb, 16), name="u_h_ub", scope=tik.scope_ubuf)

        with t.for_range(0, steps) as step:
            task = core + step * cores
            with t.if_scope(task < tasks):
                head = task // per_head_tasks
                local = task - head * per_head_tasks
                mt_idx = local // n_tiles
                nt_idx = local - mt_idx * n_tiles
                base_m = mt_idx * mb
                base_n = nt_idx * nb
                head_hi = head * hs["hi"]
                head_lo = head * hs["lo"]
                head_b = head * hs["b"]
                head_out = head * hs["out"]
                for kt in range(k_tiles):
                    a_off = (kt * kb1 * m_pad + base_m) * 16
                    b_off = (kt * kb1 * n + base_n) * 16
                    with t.new_stmt_scope(disable_sync=False):
                        t.data_move(a_hi_l1, hi_gm[head_hi + a_off], 0,
                                    dma["a_load"]["nburst"], dma["a_load"]["burst"],
                                    dma["a_load"]["src_stride"],
                                    dma["a_load"]["dst_stride"])
                        t.data_move(a_lo_l1, lo_gm[head_lo + a_off], 0,
                                    dma["a_load"]["nburst"], dma["a_load"]["burst"],
                                    dma["a_load"]["src_stride"],
                                    dma["a_load"]["dst_stride"])
                        t.data_move(b_l1, v_gm[head_b + b_off], 0,
                                    dma["b_load"]["nburst"], dma["b_load"]["burst"],
                                    dma["b_load"]["src_stride"],
                                    dma["b_load"]["dst_stride"])
                        t.matmul(c_hi, a_hi_l1, b_l1, mb, kb, nb, init_l1out=(kt == 0))
                        t.matmul(c_lo, a_lo_l1, b_l1, mb, kb, nb, init_l1out=(kt == 0))
                # Low product first, then high product (reviewed combine order).
                with t.new_stmt_scope(disable_sync=False):
                    t.tensor_mov(u_lo, c_lo, dma["l0c_to_ub"]["block_mode"],
                                 dma["l0c_to_ub"]["nburst"],
                                 dma["l0c_to_ub"]["burst_len_kb"], 0, 0)
                with t.new_stmt_scope(disable_sync=False):
                    t.tensor_mov(u_hi, c_hi, dma["l0c_to_ub"]["block_mode"],
                                 dma["l0c_to_ub"]["nburst"],
                                 dma["l0c_to_ub"]["burst_len_kb"], 0, 0)
                with t.new_stmt_scope(disable_sync=False):
                    # Every Float32 scale first, then every Float32 add, then the
                    # single narrowing cast: the reviewed combine order. One static
                    # chunk per 16-column group keeps the repeat count at 32, well
                    # under the 8-bit repeat immediate (a whole-tile call would need
                    # 256 repeats and must not be emitted).
                    for group in range(nb // 16):
                        t.vec_muls(vec["mask"], u_lo[group, :, :], u_lo[group, :, :],
                                   plan["low_downscale"], vec["repeats_per_chunk"],
                                   vec["rep_stride"], vec["rep_stride"])
                    for group in range(nb // 16):
                        t.vec_add(vec["mask"], u_hi[group, :, :], u_hi[group, :, :],
                                  u_lo[group, :, :], vec["repeats_per_chunk"],
                                  vec["rep_stride"], vec["rep_stride"],
                                  vec["rep_stride"])
                    t.h_cast(u_h, u_hi, "none")
                    c_off = head_out + ((base_n // 16) * m_pad + base_m) * 16
                    t.data_move(out_gm[c_off], u_h, 0,
                                dma["out_store"]["nburst"], dma["out_store"]["burst"],
                                dma["out_store"]["src_stride"],
                                dma["out_store"]["dst_stride"])
    return {"hi": hi_gm, "lo": lo_gm, "v": v_gm, "aux": aux_gm, "out": out_gm}


# ---------------------------------------------------------------------------
# Host packing helpers (lazy torch; never executed by this module's own tests)
# ---------------------------------------------------------------------------


def _shape_of(value, name: str, ndim: int) -> tuple:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise FusedPvError(f"{name} expects a tensor-like operand with .shape, got "
                           f"{type(value).__name__}")
    dims = tuple(int(v) for v in shape)
    if len(dims) != ndim:
        raise FusedPvError(f"{name} expects {ndim}-D input, got {dims}")
    return dims


def pack_a(a):
    """``(H, M, K)`` Half -> ``(H, K1, Mp, 16)`` Half with zero padded M rows.

    Element order and zero M padding only: no arithmetic, no dtype change, no
    Float32 shadow. Head ``h``, row ``m``, column ``k = k1*16+k0`` lands at
    ``[h, k1, m, k0]``.
    """
    h, m, k = _shape_of(a, "pack_a (H,M,K)", 3)
    import torch

    if a.dtype != torch.float16:
        raise FusedPvError(f"pack_a expects original float16 values, got {a.dtype}")
    if k % 16:
        raise FusedPvError(f"K={k} must be a multiple of 16")
    m_pad = _ceil_to(m, DEFAULT_MB)
    packed = a.detach().to(torch.float16).contiguous()
    out = torch.zeros((h, k // 16, m_pad, 16), dtype=torch.float16, device=packed.device)
    out[:, :, :m, :] = packed.reshape(h, m, k // 16, 16).permute(0, 2, 1, 3)
    return out.contiguous()


def pack_w(weight):
    """``(H, N, K)`` Half -> ``(H, K1, N, 16)`` Half, values untouched.

    For P@V the host passes ``V`` transposed to ``(H, N, K)`` where ``N`` is the
    padded head dimension and ``K`` the padded key axis.
    """
    h, n, k = _shape_of(weight, "pack_w (H,N,K)", 3)
    import torch

    if weight.dtype != torch.float16:
        raise FusedPvError(f"pack_w expects original float16 values, got {weight.dtype}")
    if k % 16:
        raise FusedPvError(f"K={k} must be a multiple of 16")
    packed = weight.detach().to(torch.float16).contiguous()
    return packed.reshape(h, n, k // 16, 16).permute(0, 2, 1, 3).contiguous()


def unpack_out_half(c, m: int, n: int = None):
    """``(H, N1, Mp, 16)`` Half -> ``(H, M, N)`` Half; crops padded rows.

    The Half output row is exactly one 16-element group, so the trailing axis is
    simply the last 16 columns of the ``(n1, mp, 16)`` layout.
    """
    shape = _shape_of(c, "unpack_out_half (H,N1,Mp,16)", 4)
    if shape[3] != 16:
        raise FusedPvError(f"unpack_out_half expects a trailing 16 axis, got {shape}")
    import torch

    if c.dtype != torch.float16:
        raise FusedPvError(f"unpack_out_half expects the Half kernel output, got {c.dtype}")
    h, n1, m_pad = shape[:3]
    if not 0 < m <= m_pad:
        raise FusedPvError(f"m={m} outside padded rows 1..{m_pad}")
    out = c[:, :, :m, :].permute(0, 2, 1, 3).reshape(h, m, n1 * 16).contiguous()
    if n is not None and n != n1 * 16:
        raise FusedPvError(f"n={n} disagrees with the {n1 * 16} stored columns")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="print tiling plans only")
    parser.add_argument("--capability", action="store_true")
    parser.add_argument("--build", nargs=4, type=int, metavar=("H", "M", "K", "N"))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({name: plan_tiling(*shape) for name, shape in TARGET_SHAPES.items()},
                         indent=2, sort_keys=True))
        return
    if args.capability:
        print(json.dumps(capability_report(), indent=2, sort_keys=True))
        return
    if args.build:
        print(json.dumps(build_kernel(*args.build, out_dir=args.out_dir), indent=2,
                         sort_keys=True))
        return
    parser.error("choose one of --plan, --capability, --build H M K N")


if __name__ == "__main__":
    main()