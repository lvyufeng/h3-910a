#!/usr/bin/env python3
"""Round87 vision GEMM: Half x Half -> FP32 L0C -> FP32 UB bias -> ONE Half GM.

SOURCE ONLY. Nothing here has been compiled, launched or executed by its author:
no SSH, no NPU/GPU work, no model run, no CANN build, no installation, no
production edit and no commit. Building a kernel says nothing about its numbers
or its speed; every runtime receipt belongs to the root agent that deploys it.

WHY THIS KERNEL EXISTS
----------------------
The H3 vision tower's linears are *biased* Half linears. On this 910A-generation
part the native biased Half linear rounds the GEMM accumulator to Half and then
rounds the bias add again, so it pays one extra output rounding that the CPU
FP16 reference does not. The Round83 portable FP32-UB bias epilogue was actually
executed and reduced the same-operand CPU-Half error by ~104x at a micro size and
~26x at the H3 KV size, without any hardware bias operand (TIK's ``matmul(bias=)``
is rejected on short-SoC Ascend910) and without FP32 weight shadows.

This module reuses that verified arithmetic with the Round83 blocked structure
(one FP32 L0C accumulator live across *all* K blocks; no Half partial and no
FP32-UB partial sum) and fixes the vision geometry:

    GM0 a_gm   : float16 (K1, M_pad, 16)      activation, zero padded M rows
    GM1 b_gm   : float16 (K1, N_pad, 16)      weight, values untouched
    GM2 bias_gm: float32 (N_pad,)             exact original Half bias, zero added N
    GM3 aux_gm : float16 (16,)                unused placeholder keeping the 5-GM ABI
    GM4 c_gm   : float16 (N1, M_pad, 16)      ONE Half cast, after the FP32 bias add

The 5-GM pointer order matches the already-proven launcher ABI; the two unused
Half aux placeholders of Round83 are collapsed into a single one because this
kernel needs a real FP32 bias operand instead of the second aux.

HOST PADDING CONTRACT (the adapter owns it; the kernel never pads anything)
--------------------------------------------------------------------------
* ``M``  : padded up to a multiple of 128 by *zero* A rows. Those padded output
           rows are **the bias, not zero** — the host must crop them.
* ``K``  : padded up to a multiple of 128 with *zero* A columns and *zero* W K
           rows. The real checkpoint MLP width is 4304 -> 4352.
* ``N``  : padded up to a multiple of 128 with *zero* W trailing rows and *zero*
           bias entries. The real checkpoint MLP width is 4304 -> 4352.

WHY THE STORE STRIDES ARE NOT THE FP32 ONES
-------------------------------------------
``data_move`` burst lengths and strides are counted in 32-byte units. One
``K0 = 16`` Half row is exactly 32 B, so a Half C row is 1 unit and the per-group
gap is ``M_pad - Mb``. The Round83 FP32-output kernel stores 64 B rows and
therefore uses ``2 * (M_pad - Mb)``; reusing that factor here would corrupt the
Half output. The Round83 bias epilogue already got this right and its actual
executed result is the evidence for this file's choice.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

# --- pinned hardware contract (identical to the actually executed R83 kernel) --
PINNED_COMPILE_SOC = "Ascend910B"
PINNED_RUNTIME_SOC = "Ascend910B"
#: Live DCMI reported 30 usable cores on all 8 original 910A-generation chips.
#: 32 GiB of HBM per chip is not 32 cube cores; 32 is the older compiler default.
PINNED_CORE_NUM = 30
PINNED_L1_SIZE = 1048576
PINNED_L0C_SIZE = 262144
PINNED_UB_SAFE_BYTES = 196608
PINNED_MAX_CORES = 30

DEFAULT_MB, DEFAULT_NB, DEFAULT_KB = 128, 128, 128

# --- immutable reference sources this file's arithmetic was copied from --------
#: Round83 blocked FP32 accumulator kernel: the reviewed plan/DMA/capability source.
REFERENCE_BASE_SHA256 = "78896adadaf47801dd92d69bdfce394d2628f7885be643dd568647c8b4c8c2ee"
#: Round83 blocked bias epilogue (portable FP32-UB bias + one Half cast) --
#: exactly the epilogue this module reuses.
REFERENCE_BIAS_EPILOGUE_SHA256 = "2f6a4470f9abc5f43028f69ddd6e012c29c2a7907bb0f6527667063bf81c8306"
#: Round83 bias micro-kernel that was ACTUALLY executed with these TIK calls.
REFERENCE_EXECUTED_BIAS_MICRO_SHA256 = (
    "124ba029a518e26b9b79051275a83144e702ebf016defe64085d8b3fa05ee03a")
#: Proven launcher binary and the bridge module the root uses unchanged.
REFERENCE_RUNTIME_LIBRARY_SHA256 = (
    "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043")
REFERENCE_RUNTIME_MODULE_SHA256 = (
    "c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6")

#: Keyed by plain file name so a staged copy in this directory and the committed
#: Round83 copy are recognised as the same reference source.
REFERENCE_SOURCES = {
    "blocked_fp32_gemm83.py": REFERENCE_BASE_SHA256,
    "blocked_bias_epilogue83.py": REFERENCE_BIAS_EPILOGUE_SHA256,
    "tik_bias_epilogue83.py": REFERENCE_EXECUTED_BIAS_MICRO_SHA256,
    "runtime5gm83.py": REFERENCE_RUNTIME_MODULE_SHA256,
}

ROOT = Path("/tmp/h3_round87_vision01")
DEFAULT_OUT_DIR = ROOT / "kernels"

#: Real vision geometry. ``k``/``n`` here are the *already padded* (multiple of
#: 128) operands the kernel computes; ``k_logical``/``n_logical`` are the model's
#: real widths and are never passed to the kernel/launcher.
VISION_CASES = (
    {"name": "qkv", "m": 8064, "k": 1152, "n": 3456, "k_logical": 1152, "n_logical": 3456},
    {"name": "proj", "m": 8064, "k": 1152, "n": 1152, "k_logical": 1152, "n_logical": 1152},
    {"name": "mlp1", "m": 8064, "k": 1152, "n": 4352, "k_logical": 1152, "n_logical": 4304},
    {"name": "mlp2", "m": 8064, "k": 4352, "n": 1152, "k_logical": 4304, "n_logical": 1152},
    {"name": "merger1", "m": 2016, "k": 4608, "n": 4608, "k_logical": 4608, "n_logical": 4608},
    {"name": "merger2", "m": 2016, "k": 4608, "n": 5120, "k_logical": 4608, "n_logical": 5120},
)

#: Real checkpoint vision widths this module's padding contract is derived from.
VISION_INTERMEDIATE_4304 = 4304
VISION_INTERMEDIATE_PADDED_4352 = 4352

SUGGESTED_SMOKE_GEOMETRY = (128, 256, 128)

GM_ARG_ORDER = (
    "a_gm_float16",
    "b_gm_float16",
    "bias_gm_float32",
    "aux_gm_float16",
    "c_gm_float16",
)


class VisionBiasError(RuntimeError):
    """Raised instead of guessing, silently falling back or overwriting output."""


# ---------------------------------------------------------------------------
# Pure-python helpers (stdlib only: unit-testable without TIK, CANN or torch)
# ---------------------------------------------------------------------------


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _require_int(name: str, value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VisionBiasError(f"{name} must be a python int, got {type(value).__name__}")
    if value <= 0:
        raise VisionBiasError(f"{name} must be positive, got {value}")
    return value


def pad_m(m: int, mb: int = DEFAULT_MB) -> int:
    return _ceil_to(_require_int("m", m), mb)


def pad_k(k: int, kb: int = DEFAULT_KB) -> int:
    return _ceil_to(_require_int("k", k), kb)


def pad_n(n: int, nb: int = DEFAULT_NB) -> int:
    return _ceil_to(_require_int("n", n), nb)


def plan_tiling(m: int, k: int, n: int, mb: int = DEFAULT_MB, nb: int = DEFAULT_NB,
                kb: int = DEFAULT_KB) -> dict:
    """Every number the kernel emission and the host packing must agree on.

    Deliberately torch-free and TIK-free so the DMA stride arithmetic can be
    reviewed on a machine with neither installed.
    """
    for name, value, unit in (("mb", mb, 16), ("nb", nb, 16), ("kb", kb, 16)):
        _require_int(name, value)
        if value % unit:
            raise VisionBiasError(f"{name}={value} must be a multiple of {unit}")
    m = _require_int("m", m)
    k = _require_int("k", k)
    n = _require_int("n", n)
    if k % kb:
        raise VisionBiasError(
            f"k={k} is not a multiple of the K tile kb={kb}; the host must zero-pad K "
            f"(vision MLP 4304 -> {pad_k(4304, kb)})")
    if n % nb:
        raise VisionBiasError(
            f"n={n} is not a multiple of the N tile nb={nb}; the host must zero-pad N "
            f"(vision MLP 4304 -> {pad_n(4304, nb)})")

    m_pad = _ceil_to(m, mb)
    k1, n1 = k // 16, n // 16
    m_tiles, n_tiles = m_pad // mb, n // nb
    tasks = m_tiles * n_tiles
    cores = min(PINNED_MAX_CORES, tasks)
    steps = (tasks + cores - 1) // cores
    k_tiles = k // kb
    kb1 = kb // 16

    gm_shapes = {
        "a": [k1, m_pad, 16],
        "b": [k1, n, 16],
        "bias": [n],
        "aux": [16],
        "c": [n1, m_pad, 16],
    }
    gm_elems = {
        "a": k1 * m_pad * 16,
        "b": k1 * n * 16,
        "bias": n,
        "c": n1 * m_pad * 16,
    }
    gm_bytes = {
        "a": gm_elems["a"] * 2,
        "b": gm_elems["b"] * 2,
        "bias": gm_elems["bias"] * 4,
        "c": gm_elems["c"] * 2,
    }

    # DMA descriptions. data_move burst/strides are in 32-byte units; tensor_mov
    # burst_len is in 1024-byte units. A full K0=16 Half row is exactly 32 B, so
    # a burst of `mb` rows is `mb` units and the A gap between K1 planes is
    # `m_pad - mb`. The C store is HALF (1 unit per row), hence `m_pad - mb` and
    # NOT the FP32-output kernel's `2 * (m_pad - mb)`.
    dma = {
        "a_load": {"nburst": kb1, "burst": mb, "src_stride": m_pad - mb, "dst_stride": 0},
        "b_load": {"nburst": kb1, "burst": nb, "src_stride": n - nb, "dst_stride": 0},
        "bias_load": {"nburst": 1, "burst": nb * 4 // 32, "src_stride": 0, "dst_stride": 0},
        "l0c_to_ub": {"block_mode": "m", "nburst": 1,
                      "burst_len_kb": mb * nb * 4 // 1024},
        "c_store": {"nburst": nb // 16, "burst": mb, "src_stride": 0,
                    "dst_stride": m_pad - mb, "units_bytes": 32,
                    "half_row_units": 1, "fp32_output_stride_would_be": 2 * (m_pad - mb)},
    }

    buffers = {
        "l1_a_bytes": kb1 * mb * 16 * 2,
        "l1_b_bytes": kb1 * nb * 16 * 2,
        "l0c_bytes": (nb // 16) * mb * 16 * 4,
        "ub_c32_bytes": (nb // 16) * mb * 16 * 4,
        "ub_c16_bytes": (nb // 16) * mb * 16 * 2,
        "ub_bias_bytes": nb * 4,
        "ub_bytes": (nb // 16) * mb * 16 * 4 + (nb // 16) * mb * 16 * 2 + nb * 4,
    }

    return {
        "logical": {"m": m, "k": k, "n": n},
        "tiles": {"mb": mb, "nb": nb, "kb": kb},
        "padded": {"m_pad": m_pad, "m_pad_rows": m_pad - m, "k1": k1, "n1": n1},
        "grid": {"m_tiles": m_tiles, "n_tiles": n_tiles, "tasks": tasks,
                 "cores": cores, "steps": steps,
                 "all_physical_cores_used": cores == PINNED_MAX_CORES,
                 "tasks_per_core_upper_bound": steps},
        "gm_shapes": gm_shapes,
        "gm_elems": gm_elems,
        "gm_bytes": gm_bytes,
        "dma": dma,
        "buffers": buffers,
        "accumulation": {
            "k_tiles": k_tiles,
            "k_blocks_per_tile": kb,
            "single_l0c_across_all_k_blocks": True,
            "init_first_k_block_only": True,
            "half_partial": False,
            "fp32_ub_partial_sum": False,
            "final_output_dtype": "float16",
            "single_final_half_cast": True,
            "bias_added_in_fp32": True,
        },
        "epilogue": {
            "order": ["tensor_mov_L0C_to_UB_fp32", "vec_add_fp32_bias", "h_cast_once", "data_move_half"],
            "bias_ub_dtype": "float32",
            "tile_bias_groups": nb // 16,
            "vec_add_mask_fp32_elements": 16,
            "vec_add_rep_rows": mb,
            "vec_add_rep_strides": [2, 2],
            "h_cast_count_per_tile": 1,
            "h_cast_api": "h_cast(dst, src, 'none')",
            "hardware_bias_operand_used": False,
            "padded_output_rows_are_bias_not_zero": True,
        },
        "host_padding_contract": {
            "m": "zero A rows up to m_pad; padded output rows equal the bias and "
                 "must be cropped before model use",
            "k": "zero A columns and zero W K-rows up to a multiple of 128 "
                 f"(vision MLP {VISION_INTERMEDIATE_4304} -> "
                 f"{VISION_INTERMEDIATE_PADDED_4352})",
            "n": "zero W trailing rows and zero bias entries up to a multiple of 128 "
                 f"(vision MLP {VISION_INTERMEDIATE_4304} -> "
                 f"{VISION_INTERMEDIATE_PADDED_4352}); crop output columns",
            "mlp_logical_intermediate": VISION_INTERMEDIATE_4304,
            "mlp_padded_intermediate": VISION_INTERMEDIATE_PADDED_4352,
            "bias_values": "exact original Half bias promoted to Float32; zero for "
                           "added N entries",
        },
        "kernel_arg_order": list(GM_ARG_ORDER),
        "limits": {"l1_size": PINNED_L1_SIZE, "l0c_size": PINNED_L0C_SIZE,
                   "ub_safe_bytes": PINNED_UB_SAFE_BYTES, "max_cores": PINNED_MAX_CORES},
    }


def _check_budget(plan: dict) -> None:
    buf = plan["buffers"]
    l1_need = buf["l1_a_bytes"] + buf["l1_b_bytes"]
    if l1_need > PINNED_L1_SIZE:
        raise VisionBiasError(f"L1 need {l1_need} B exceeds {PINNED_L1_SIZE} B")
    if buf["l0c_bytes"] > PINNED_L0C_SIZE:
        raise VisionBiasError(f"L0C need {buf['l0c_bytes']} B exceeds {PINNED_L0C_SIZE} B")
    if buf["ub_bytes"] > PINNED_UB_SAFE_BYTES:
        raise VisionBiasError(
            f"UB need {buf['ub_bytes']} B exceeds the conservative {PINNED_UB_SAFE_BYTES} B bound")


def kernel_name_for(plan: dict) -> str:
    lg, tl, gr = plan["logical"], plan["tiles"], plan["grid"]
    return (f"vision_bias87_m{lg['m']}_k{lg['k']}_n{lg['n']}"
            f"_mb{tl['mb']}_nb{tl['nb']}_kb{tl['kb']}_c{gr['cores']}")


# --- pure-python index maps for the three GM operands ------------------------
# These are the exact element maps the torch packing helpers below implement with
# ``reshape(...).transpose(0, 1)`` / ``transpose(0, 1).reshape(...)``. They are
# kept torch-free so the layout is verifiable on a machine without torch, and the
# host adapter can cross-check any packing it decides to batch/fuse later.


def pack_a_index(m_index: int, k_index: int) -> tuple:
    """``A[m, k]`` -> ``packed_a[k1, m, k0]`` with ``k = k1*16 + k0``."""
    k1, k0 = divmod(int(k_index), 16)
    return (k1, int(m_index), k0)


def pack_w_index(n_index: int, k_index: int) -> tuple:
    """``W[n, k]`` -> ``packed_w[k1, n, k0]`` with ``k = k1*16 + k0``."""
    k1, k0 = divmod(int(k_index), 16)
    return (k1, int(n_index), k0)


def c_index(m_index: int, n_index: int) -> tuple:
    """Kernel output ``C[n1, m, n0]`` for logical ``(m, n)`` with ``n = n1*16 + n0``."""
    n1, n0 = divmod(int(n_index), 16)
    return (n1, int(m_index), n0)


def host_layouts() -> dict:
    """Plain description of the three GM element maps plus their inverse order."""
    return {
        "a": {"logical": "(M, K) float16", "gm": "(K1, M_pad, 16) float16",
              "map": "A[m, k] -> [k1, m, k0] with k = k1*16 + k0",
              "m_padding": "rows m..M_pad-1 are zero"},
        "b": {"logical": "(N, K) float16", "gm": "(K1, N, 16) float16",
              "map": "W[n, k] -> [k1, n, k0] with k = k1*16 + k0",
              "values": "untouched original Half"},
        "bias": {"logical": "(N,) original Half", "gm": "(N,) float32",
                 "map": "exact Half -> Float32 promotion; appended N entries are zero"},
        "c": {"logical": "(M, N) float16", "gm": "(N1, M_pad, 16) float16",
              "map": "C[m, n] <- [n1, m, n0] with n = n1*16 + n0",
              "host_crop": "drop rows m..M_pad-1 (they equal the bias) and columns "
                           "n_logical..N-1 (added zero-N)"},
    }


def planned_geometries() -> list:
    """The six real vision geometries with their resolved tiling numbers."""
    rows = []
    for case in VISION_CASES:
        plan = plan_tiling(case["m"], case["k"], case["n"])
        row = dict(case)
        row.update({
            "m_pad": plan["padded"]["m_pad"],
            "m_pad_rows": plan["padded"]["m_pad_rows"],
            "k_padded": pad_k(case["k_logical"]),
            "n_padded": pad_n(case["n_logical"]),
            "k_tiles": plan["accumulation"]["k_tiles"],
            "tasks": plan["grid"]["tasks"],
            "cores": plan["grid"]["cores"],
            "steps": plan["grid"]["steps"],
            "kernel_name": kernel_name_for(plan),
        })
        if row["k_padded"] != case["k"] or row["n_padded"] != case["n"]:
            raise VisionBiasError(f"case {case['name']} declares unpadded kernel operands")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reference-source verification (the R83 sources this file was derived from)
# ---------------------------------------------------------------------------


def _reference_candidates() -> list:
    here = Path(__file__).resolve().parent
    return [
        here / "blocked_fp32_gemm83.py",
        here / "blocked_bias_epilogue83.py",
        here / "tik_bias_epilogue83.py",
        here / "runtime5gm83.py",
        here.parents[1] / "round83" / "blocked" / "blocked_fp32_gemm83.py",
        here.parents[1] / "round83" / "blocked_bias_epilogue83.py",
        here.parents[1] / "round83" / "bias" / "tik_bias_epilogue83.py",
        here.parents[1] / "round83" / "runtime5gm83.py",
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_status() -> dict:
    """Hash every locatable R83 reference source; never raises for a missing file.

    A present source whose bytes differ from its pin is a hard error: the
    arithmetic this module copies would no longer match the reviewed evidence.
    """
    by_name = {}
    for candidate in _reference_candidates():
        if not candidate.is_file():
            continue
        name = candidate.name
        digest = _sha256_file(candidate)
        expected = REFERENCE_SOURCES.get(name)
        entry = {"path": str(candidate), "sha256": digest,
                 "expected_sha256": expected, "matches_pin": digest == expected}
        previous = by_name.get(name)
        if previous is not None and previous["sha256"] != digest:
            raise VisionBiasError(
                f"two different copies of reference source {name} disagree: "
                f"{previous['path']} vs {candidate}")
        by_name[name] = entry
    missing = sorted(name for name in REFERENCE_SOURCES if name not in by_name)
    return {
        "sources": by_name,
        "verified": sorted(name for name in by_name
                           if by_name[name]["matches_pin"]),
        "missing": missing,
        "all_pinned_sources_present_and_matching": not missing and all(
            entry["matches_pin"] for entry in by_name.values()),
        "no_reference_source_edited_by_this_round": True,
    }


def check_reference_sources(strict: bool = True) -> dict:
    """Fail closed if a present R83 reference source was modified."""
    status = reference_status()
    for name, entry in status["sources"].items():
        if not entry["matches_pin"] and strict:
            raise VisionBiasError(
                f"reference source {name} changed: {entry['sha256']} != "
                f"{entry['expected_sha256']}; this Round87 kernel's arithmetic assumption no "
                f"longer matches the reviewed evidence at {entry['path']}")
    return status


# ---------------------------------------------------------------------------
# Environment guards (lazy: importing this module needs no CANN and no torch)
# ---------------------------------------------------------------------------


def capability_report() -> dict:
    """Query the facts this kernel depends on, without building anything."""
    import tbe.common.platform as platform

    platform.set_current_compile_soc_info(PINNED_COMPILE_SOC)
    resolved = platform.get_soc_spec("FULL_SOC_VERSION")
    if resolved != PINNED_COMPILE_SOC:
        raise VisionBiasError(
            f"compile SoC resolved to {resolved!r}, expected {PINNED_COMPILE_SOC!r}; refusing to "
            "emit a different ISA")
    report = {
        "compile_soc": resolved,
        "runtime_expected_soc": PINNED_RUNTIME_SOC,
        "runtime_soc_label_note": "the runtime reports the short SoC name Ascend910 for this "
                                  "original 910A-generation part; 30 usable cube cores",
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
        raise VisionBiasError(
            "Intrinsic_mmad/f162f32 unsupported: the FP32 accumulator route this kernel needs "
            "is unavailable and must not be faked")
    if report["core_num"] != PINNED_CORE_NUM:
        raise VisionBiasError(
            f"compiler CORE_NUM is {report['core_num']}, expected {PINNED_CORE_NUM}. The live "
            "DCMI audit reports 30 usable cores on all 8 original 910B chips; 32 is the older "
            "compiler default and 32 GiB of memory is not 32 cores.")
    if report["l1_size"] != PINNED_L1_SIZE or report["l0c_size"] != PINNED_L0C_SIZE:
        raise VisionBiasError("L1/L0C sizes do not match the pinned 910A-generation part")
    if report["ub_size"] is not None and report["ub_size"] < PINNED_UB_SAFE_BYTES:
        raise VisionBiasError(
            f"UB_SIZE {report['ub_size']} is smaller than the conservative "
            f"{PINNED_UB_SAFE_BYTES} B bound this kernel relies on")
    return report


# ---------------------------------------------------------------------------
# Kernel emission
# ---------------------------------------------------------------------------


def _emit_kernel(t, plan: dict, tik) -> dict:
    """Emit the TIK program. Every offset here is derived from ``plan``.

    Review notes:

    * ``init_l1out`` must be a Python ``bool`` (the SDK type-checks it), so the K
      loop is a *static* Python loop and only ``kt == 0`` initialises L0C; every
      later K block accumulates into the same single FP32 L0C tensor. There is no
      Half partial and no FP32-UB partial sum.
    * The outer tile step is the only dynamic loop; the head/tile index is folded
      into ``task`` so the emitted IR does not grow with the problem size.
    * Each K block is wrapped in ``new_stmt_scope(disable_sync=False)``, the sync
      boundary that orders the L1 reload (WAR) against the previous cube read.
    * The epilogue is exactly the executed Round83 order: FP32 L0C -> FP32 UB
      (no narrowing), FP32 bias add, ONE ``h_cast('none')``, then the Half store.
      ``matmul`` is called with ``bias=None``: TIK's hardware bias argument is
      rejected on short-SoC Ascend910 and must never be faked.
    * Padded A rows are zero on the host, so those output rows are exactly the
      bias; the host crops them. No other OOB access exists because ``M_pad`` is
      a multiple of ``mb`` and ``N``/``K`` are tile multiples.
    """
    lg, tl, pd, gr, dma = (plan["logical"], plan["tiles"], plan["padded"],
                           plan["grid"], plan["dma"])
    n, k1, n1 = lg["n"], pd["k1"], pd["n1"]
    m_pad = pd["m_pad"]
    mb, nb, kb = tl["mb"], tl["nb"], tl["kb"]
    kb1 = kb // 16
    k_tiles = plan["accumulation"]["k_tiles"]
    cores, steps, tasks, n_tiles = gr["cores"], gr["steps"], gr["tasks"], gr["n_tiles"]

    ag = t.Tensor("float16", (k1, m_pad, 16), name="a_gm", scope=tik.scope_gm)
    bg = t.Tensor("float16", (k1, n, 16), name="b_gm", scope=tik.scope_gm)
    biasg = t.Tensor("float32", (n,), name="bias_gm", scope=tik.scope_gm)
    aux = t.Tensor("float16", (16,), name="aux_gm", scope=tik.scope_gm)
    cg = t.Tensor("float16", (n1, m_pad, 16), name="c_gm", scope=tik.scope_gm)

    with t.for_range(0, cores, block_num=cores) as core:
        al = t.Tensor("float16", (kb1, mb, 16), name="a_l1", scope=tik.scope_cbuf)
        bl = t.Tensor("float16", (kb1, nb, 16), name="b_l1", scope=tik.scope_cbuf)
        # "local.L0C": the SDK requires dst.scope == scope_cbuf_out for matmul.
        c0 = t.Tensor("float32", (nb // 16, mb, 16), name="c_l0c", scope=tik.scope_cc)
        c32 = t.Tensor("float32", (nb // 16, mb, 16), name="c32_ub", scope=tik.scope_ubuf)
        c16 = t.Tensor("float16", (nb // 16, mb, 16), name="c16_ub", scope=tik.scope_ubuf)
        biasub = t.Tensor("float32", (nb,), name="bias_ub", scope=tik.scope_ubuf)

        with t.for_range(0, steps) as step:
            task = core + step * cores
            with t.if_scope(task < tasks):
                base_m = (task // n_tiles) * mb
                base_n = (task % n_tiles) * nb
                for kt in range(k_tiles):
                    a_off = (kt * kb1 * m_pad + base_m) * 16
                    b_off = (kt * kb1 * n + base_n) * 16
                    with t.new_stmt_scope(disable_sync=False):
                        t.data_move(al, ag[a_off], 0, dma["a_load"]["nburst"],
                                    dma["a_load"]["burst"], dma["a_load"]["src_stride"],
                                    dma["a_load"]["dst_stride"])
                        t.data_move(bl, bg[b_off], 0, dma["b_load"]["nburst"],
                                    dma["b_load"]["burst"], dma["b_load"]["src_stride"],
                                    dma["b_load"]["dst_stride"])
                        # No hardware bias operand: it is rejected on this SoC and
                        # the portable FP32 UB epilogue below is the verified route.
                        t.matmul(c0, al, bl, mb, kb, nb, init_l1out=(kt == 0), bias=None)
                c_off = ((base_n // 16) * m_pad + base_m) * 16
                with t.new_stmt_scope(disable_sync=False):
                    # FP32 L0C -> FP32 UB: the accumulator leaves without narrowing.
                    t.tensor_mov(c32, c0, dma["l0c_to_ub"]["block_mode"],
                                 dma["l0c_to_ub"]["nburst"], dma["l0c_to_ub"]["burst_len_kb"],
                                 0, 0)
                    t.data_move(biasub, biasg[base_n], 0, dma["bias_load"]["nburst"],
                                dma["bias_load"]["burst"], dma["bias_load"]["src_stride"],
                                dma["bias_load"]["dst_stride"])
                    # One vec_add per 16-column group: the same 16 FP32 bias values
                    # are broadcast onto all mb rows (in place, FP32 throughout).
                    for group in range(nb // 16):
                        t.vec_add(16, c32[group, :, :], c32[group, :, :],
                                  biasub[group * 16], mb, 2, 2, 0)
                    # Exactly ONE FP32 -> FP16 conversion, after the bias add.
                    t.h_cast(c16, c32, "none")
                    # Half C row16 is 32 B = 1 unit, so the gap is m_pad - mb and
                    # NOT the FP32-output kernel's 2 * (m_pad - mb).
                    t.data_move(cg[c_off], c16, 0, dma["c_store"]["nburst"],
                                dma["c_store"]["burst"], dma["c_store"]["src_stride"],
                                dma["c_store"]["dst_stride"])
    return {"a": ag, "b": bg, "bias": biasg, "aux": aux, "c": cg}


# ---------------------------------------------------------------------------
# Build (cached, locked, never overwriting a previous output directory)
# ---------------------------------------------------------------------------


def build_kernel(m: int, k: int, n: int, out_dir=None) -> dict:
    """Build (or reuse) one vision bias kernel and describe it.

    ``k`` and ``n`` must already be padded to multiples of 128 by the caller.
    Returns metadata with the Round83 launcher-compatible keys; ``executed`` is
    always ``False`` because building says nothing about numerics or speed.
    """
    plan = plan_tiling(m, k, n)
    _check_budget(plan)

    from tbe import tik

    capability = capability_report()
    if capability["ub_size"] is not None and plan["buffers"]["ub_bytes"] > capability["ub_size"]:
        raise VisionBiasError("UB tile does not fit the reported UB size")

    references = check_reference_sources(strict=True)
    name = kernel_name_for(plan)
    out_root = Path(out_dir).resolve() if out_dir is not None else DEFAULT_OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec87.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                                references, compiled_now=False)

    with (out_root / ".vision87.compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                                    references, compiled_now=False)

        work = Path(tempfile.mkdtemp(prefix="vision_bias87_build_", dir=out_root)).resolve()
        if work.parent != out_root or not work.name.startswith("vision_bias87_build_"):
            raise VisionBiasError("unexpected build workspace")
        t = tik.Tik(disable_debug=True)
        gm = _emit_kernel(t, plan, tik)
        previous = os.getcwd()
        try:
            os.chdir(work)
            # inputs/outputs are passed explicitly: the unused Half aux operand
            # exists only to keep the proven 5-GM-pointer ABI, and automatic
            # collection would prune it.
            t.BuildCCE(kernel_name=name,
                       inputs=[gm["a"], gm["b"], gm["bias"], gm["aux"]],
                       outputs=[gm["c"]])
        finally:
            os.chdir(previous)
        built_obj = work / "kernel_meta" / (name + ".o")
        built_meta = work / "kernel_meta" / (name + ".json")
        if not built_obj.is_file() or not built_meta.is_file():
            # Retain the exact build workspace for failure diagnosis; never delete
            # or move generated paths based only on a string prefix.
            raise VisionBiasError(
                f"BuildCCE produced no kernel_meta artefacts for {name}; the build workspace "
                f"{work} is retained for diagnosis")
        if dest.exists():
            raise VisionBiasError(f"refusing to overwrite existing kernel directory {dest}")
        dest.mkdir(parents=True)
        import shutil

        shutil.copyfile(built_obj, obj)
        shutil.copyfile(built_meta, meta_path)

    return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                            references, compiled_now=True)


def _write_spec_once(spec_path: Path, spec: dict) -> bool:
    """Write the informational spec file only when it is absent or disagrees."""
    if spec_path.is_file():
        try:
            existing = json.loads(spec_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            existing = None
        if isinstance(existing, dict) and \
                existing.get("obj_sha256") == spec["obj_sha256"] and \
                existing.get("kernel_name") == spec["kernel_name"]:
            return False
        # A stale/torn informational file inside our own destination directory is
        # rewritten atomically; the kernel object itself is never touched here.
        intended = Path(spec['obj_path']).resolve(strict=True).parent / 'spec87.json'
        resolved_spec = spec_path.resolve(strict=False)
        if resolved_spec != intended:
            raise VisionBiasError('unexpected resolved absolute spec repair target')
        tmp = resolved_spec.with_suffix(".json.tmp87")
        if tmp.resolve(strict=False).parent != intended.parent:
            raise VisionBiasError("unexpected temporary spec path")
        tmp.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        # Both absolute source and destination are verified before the move.
        os.replace(tmp, resolved_spec)
        return True
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return True


def _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability, references,
                     compiled_now=False) -> dict:
    meta = json.loads(meta_path.read_text())
    if int(meta.get("blockDim", -1)) != plan["grid"]["cores"]:
        raise VisionBiasError(
            f"cached kernel {name} has blockDim {meta.get('blockDim')} but the plan needs "
            f"{plan['grid']['cores']} cores; refusing to reuse a mismatched binary")
    obj_sha = _sha256_file(obj)
    source_sha = _sha256_file(Path(__file__))
    spec = {
        "round": 87,
        "scope": "vision blocked FP16xFP16 -> FP32 L0C -> FP32 UB bias -> ONE Half GM; "
                 "no hardware bias operand; no Half partial; no FP32 weight shadow",
        "executed": False,
        "compiled_now": compiled_now,
        "built": True,
        "built_unix_epoch": int(time.time()),
        "cache_source_epoch": {"kernel_source_sha256": source_sha,
                               "reference_base_sha256": REFERENCE_BASE_SHA256},
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "kernel_name": name,
        "kernel_dir": str(dest),
        "kernel_symbol": meta["kernelName"],
        "block_dim": int(meta["blockDim"]),
        "core_type": meta.get("coreType"),
        "obj_path": str(obj),
        "obj_sha256": obj_sha,
        "json_path": str(meta_path),
        "source_sha256": source_sha,
        "reference_base_source_sha256": REFERENCE_BASE_SHA256,
        "reference_bias_epilogue_source_sha256": REFERENCE_BIAS_EPILOGUE_SHA256,
        "reference_executed_bias_micro_sha256": REFERENCE_EXECUTED_BIAS_MICRO_SHA256,
        "reference_runtime_library_sha256": REFERENCE_RUNTIME_LIBRARY_SHA256,
        "reference_runtime_module_sha256": REFERENCE_RUNTIME_MODULE_SHA256,
        "reference_sources": references,
        "dtype": "float16_in_float16_out_with_fp32_bias",
        "dst_dtype": "float16",
        "input_dtype": "float16",
        "weight_dtype": "float16",
        "bias_dtype": "float32",
        "accumulator_dtype": "float32",
        "output_dtype": "float16",
        "uses_fp32_bias": True,
        "single_final_half_cast": True,
        "matmul_bias_operand_used": False,
        "hardware_bias_operand_used": False,
        "no_fp32_weight_shadow": True,
        "no_bfloat16_teacher": True,
        "bias_values_must_be_exact_original_half": True,
        "padded_output_rows_are_bias_not_zero": True,
        "GM_argument_count": 5,
        "kernel_arg_order": plan["kernel_arg_order"],
        "logical_shape": plan["logical"],
        "padded_shape": {"m_pad": plan["padded"]["m_pad"],
                         "m_pad_rows": plan["padded"]["m_pad_rows"],
                         "k1": plan["padded"]["k1"], "n1": plan["padded"]["n1"]},
        "tiles": plan["tiles"],
        "grid": plan["grid"],
        "gm_shapes": plan["gm_shapes"],
        "gm_elems": plan["gm_elems"],
        "gm_bytes": plan["gm_bytes"],
        "dma": plan["dma"],
        "buffers": plan["buffers"],
        "accumulation": plan["accumulation"],
        "epilogue": plan["epilogue"],
        "host_padding_contract": plan["host_padding_contract"],
        "host_packing_required": ["pack_a", "pack_w", "pack_bias"],
        "host_unpacking_required": ["unpack_c"],
        "capability": capability,
        "suggested_smoke_geometry": list(SUGGESTED_SMOKE_GEOMETRY),
    }
    _write_spec_once(spec_path, spec)
    return spec


# ---------------------------------------------------------------------------
# Host packing helpers (lazy torch; never executed by this module's own tests)
# ---------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def pack_a(a):
    """``(M, K)`` float16 ND -> ``(K1, Mp, 16)`` float16 with zero padded M rows.

    Only element order and zero M padding: no arithmetic, no dtype change, no
    FP32 shadow. Row ``m`` and column ``k = k1*16 + k0`` land at ``[k1, m, k0]``.
    The caller must already have zero padded the K columns.
    """
    torch = _torch()
    if a.dim() != 2:
        raise VisionBiasError(f"pack_a expects a 2-D (M,K) tensor, got {tuple(a.shape)}")
    if a.dtype != torch.float16:
        raise VisionBiasError(f"pack_a expects original float16 values, got {a.dtype}")
    m, k = int(a.shape[0]), int(a.shape[1])
    if k % 16:
        raise VisionBiasError(f"K={k} must be a multiple of 16")
    m_pad = pad_m(m)
    packed = a.detach().to(torch.float16).contiguous()
    out = torch.zeros((k // 16, m_pad, 16), dtype=torch.float16, device=packed.device)
    out[:, :m, :] = packed.reshape(m, k // 16, 16).transpose(0, 1)
    return out.contiguous()


def pack_w(weight):
    """``(N, K)`` float16 ND -> ``(K1, N, 16)`` float16, values untouched."""
    torch = _torch()
    if weight.dim() != 2:
        raise VisionBiasError(f"pack_w expects a 2-D (N,K) tensor, got {tuple(weight.shape)}")
    if weight.dtype != torch.float16:
        raise VisionBiasError(f"pack_w expects original float16 values, got {weight.dtype}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if k % 16:
        raise VisionBiasError(f"K={k} must be a multiple of 16")
    packed = weight.detach().to(torch.float16).contiguous()
    return packed.reshape(n, k // 16, 16).transpose(0, 1).contiguous()


def pack_bias(bias):
    """Original Half bias ``(N,)`` -> exact Float32 ``(N,)``; no value rounding.

    The promotion Half -> Float32 is exact, so this cannot change the model's bias
    values. The caller zero pads this array to the kernel's padded N.
    """
    torch = _torch()
    if bias is None:
        raise VisionBiasError("vision linears are biased; a missing bias must not be faked")
    if bias.dim() != 1:
        raise VisionBiasError(f"pack_bias expects a 1-D (N,) tensor, got {tuple(bias.shape)}")
    if bias.dtype != torch.float16:
        raise VisionBiasError(f"pack_bias expects the original float16 bias, got {bias.dtype}")
    promoted = bias.detach().to(torch.float32).contiguous()
    if not torch.equal(promoted, bias.detach().float()):
        raise VisionBiasError("Float32 bias promotion is not exact")
    return promoted


def pad_k_activation(x, k_padded: int):
    """Zero pad activation columns ``K_logical .. K_padded-1`` (exactness checked)."""
    torch = _torch()
    if x.dim() < 1:
        raise VisionBiasError("pad_k_activation expects at least 1-D input")
    if x.dtype != torch.float16:
        raise VisionBiasError(f"pad_k_activation expects float16 activation, got {x.dtype}")
    k = int(x.shape[-1])
    if k_padded < k or k_padded % 16:
        raise VisionBiasError(f"bad padded K {k_padded} for K={k}")
    if k_padded == k:
        return x.detach().contiguous()
    out = torch.zeros(x.shape[:-1] + (k_padded,), dtype=x.dtype, device=x.device)
    out[..., :k] = x.detach()
    if not torch.equal(out[..., :k], x.detach()) or int(torch.count_nonzero(out[..., k:]).item()):
        raise VisionBiasError("K padding is not an exact zero extension")
    return out.contiguous()


def pad_n_weight(weight, n_padded: int):
    """Zero pad trailing weight rows ``N_logical .. N_padded-1`` (exactness checked)."""
    torch = _torch()
    if weight.dim() != 2:
        raise VisionBiasError(f"pad_n_weight expects a 2-D (N,K) weight, got {tuple(weight.shape)}")
    if weight.dtype != torch.float16:
        raise VisionBiasError(f"pad_n_weight expects original float16 weight, got {weight.dtype}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if n_padded < n or n_padded % 16:
        raise VisionBiasError(f"bad padded N {n_padded} for N={n}")
    if n_padded == n:
        return weight.detach().contiguous()
    out = torch.zeros((n_padded, k), dtype=weight.dtype, device=weight.device)
    out[:n] = weight.detach()
    if not torch.equal(out[:n], weight.detach()) or int(torch.count_nonzero(out[n:]).item()):
        raise VisionBiasError("N weight padding is not an exact zero extension")
    return out.contiguous()


def pad_n_bias(bias_promoted, n_padded: int):
    """Zero append bias entries for the added N columns ``N_logical .. N_padded-1``."""
    torch = _torch()
    if bias_promoted.dim() != 1 or bias_promoted.dtype != torch.float32:
        raise VisionBiasError(
            f"pad_n_bias expects the promoted 1-D Float32 bias, got "
            f"{tuple(bias_promoted.shape)}/{bias_promoted.dtype}")
    n = int(bias_promoted.shape[0])
    if n_padded < n or n_padded % 16:
        raise VisionBiasError(f"bad padded N {n_padded} for N={n}")
    if n_padded == n:
        return bias_promoted.detach().contiguous()
    out = torch.zeros((n_padded,), dtype=torch.float32, device=bias_promoted.device)
    out[:n] = bias_promoted.detach()
    if not torch.equal(out[:n], bias_promoted.detach()) or int(torch.count_nonzero(out[n:]).item()):
        raise VisionBiasError("N bias padding is not an exact zero extension")
    return out.contiguous()


def unpack_c(c, m: int, n_logical: int | None = None):
    """``(N1, Mp, 16)`` float16 -> ``(M, N)``; crops padded M rows and padded N."""
    torch = _torch()
    if c.dim() != 3 or int(c.shape[2]) != 16:
        raise VisionBiasError(f"unpack_c expects (N1, Mp, 16), got {tuple(c.shape)}")
    if c.dtype != torch.float16:
        raise VisionBiasError(f"unpack_c expects the kernel's Half output, got {c.dtype}")
    n1, m_pad = int(c.shape[0]), int(c.shape[1])
    if not 0 < m <= m_pad:
        raise VisionBiasError(f"m={m} outside padded rows 1..{m_pad}")
    n_padded = n1 * 16
    if n_logical is None:
        n_logical = n_padded
    if not 0 < n_logical <= n_padded:
        raise VisionBiasError(f"n_logical={n_logical} outside 1..{n_padded}")
    out = c[:, :m, :].transpose(0, 1).reshape(m, n_padded).contiguous()
    return out[:, :n_logical].contiguous()


def reference_pins() -> dict:
    """Plain dict of the immutable hashes this module depends on."""
    return {
        "reference_base_sha256": REFERENCE_BASE_SHA256,
        "reference_bias_epilogue_sha256": REFERENCE_BIAS_EPILOGUE_SHA256,
        "reference_executed_bias_micro_sha256": REFERENCE_EXECUTED_BIAS_MICRO_SHA256,
        "reference_runtime_library_sha256": REFERENCE_RUNTIME_LIBRARY_SHA256,
        "reference_runtime_module_sha256": REFERENCE_RUNTIME_MODULE_SHA256,
        "pinned_compile_soc": PINNED_COMPILE_SOC,
        "pinned_runtime_soc_label": PINNED_RUNTIME_SOC,
        "pinned_core_num": PINNED_CORE_NUM,
        "pinned_max_cores": PINNED_MAX_CORES,
        "kernel_prefix": "vision_bias87_",
    }


def main() -> None:
    print(json.dumps({"reference_pins": reference_pins(),
                      "reference_sources": reference_status(),
                      "planned_geometries": planned_geometries(),
                      "host_layouts": host_layouts(),
                      "suggested_smoke_geometry": list(SUGGESTED_SMOKE_GEOMETRY),
                      "executed": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()