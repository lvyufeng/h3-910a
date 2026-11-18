#!/usr/bin/env python3
"""Round85 batched per-head Half x Half -> FP32 L0C -> FP32 GM GEMM.

SOURCE ONLY. Nothing in this module has been compiled or executed here: the
author performed no SSH, no NPU/GPU work, no torch import, no remote compile,
no installation and no model run. Building a kernel says nothing about its
numbers. Every runtime receipt belongs to the root agent that deploys this.

Why a batched kernel
--------------------
The Round83 blocked GEMM computes one (M,K,N) product per launch. Real text
attention in the H3 encoder runs 8 local heads per TP rank, each with its own
Q/K/V matrices, so a per-head Python loop would issue 8 tiny launches and, for
the P@V product (N=128), give only four 128x128 output tiles per head - far
below the 30 physical cores.

This module keeps Round83's verified arithmetic *exactly* (Half operands, one
FP32 L0C accumulator live across every K block, FP32 GM output, no Half
partial, no FP32 UB partial sum, no FP32 weight shadow) and only adds a leading
batch (head) dimension to the GM operands and to the flattened task grid.

Layouts (leading head dimension added to the executed Round83 convention)
------------------------------------------------------------------------
* A GM: ``(H, K1, Mp, 16)`` float16, element ``(h,k1,m,k0) = A[h, m, k1*16+k0]``
* B GM: ``(H, K1, N, 16)``  float16, element ``(h,k1,n,k0) = B[h, n, k1*16+k0]``
* C GM: ``(H, N1, Mp, 16)`` float32, element ``(h,n1,m,n0) = C[h, m, n1*16+n0]``

``Mp = ceil(M/128)*128``; rows ``M..Mp-1`` of every head's A are zero, so the
padded rows produce exact zero output rows and the host crops them. **Every
head owns a distinct dense B matrix**; no head ever reads another head's
weights, and the head stride is part of each GM offset.

Target geometries (H = 8 local heads per TP rank)
-------------------------------------------------
* score:  H8 M512 K128  N2048   (query chunk; last chunk M507 padded to 512)
* P@V:    H8 M512 K2048 N128    (last chunk M507 padded to 512)

The score product's N is the padded key axis (true keys 2043 -> 2048), which the
host zero-fills; this kernel itself has no masking semantics.

GM argument ABI (5 pointers, unchanged from the proven fa910a launcher)
---------------------------------------------------------------------
``[A_Half, B_Half, aux0_Half16, aux1_Half16, C_Float32]``

The two aux operands are deliberately unused arithmetic placeholders that only
exist to preserve the proven 5-GM-pointer ABI. They are passed explicitly to
``BuildCCE`` so automatic pruning cannot drop them.

Usage (root driver, inside the private NPU run)
-----------------------------------------------
    import batch_gemm85 as g
    spec = g.build_kernel(8, 512, 128, 2048, out_dir=...)   # compiles once
    a = g.pack_a(q_half_h_m_k)      # (H, K1, Mp, 16) Half, zero M padding
    b = g.pack_w(k_half_h_n_k)      # (H, K1, N, 16)  Half, values untouched
    c = torch.empty(spec['gm_shapes']['c'], dtype=torch.float32, device=dev)
    ... launch 5 GM pointers per spec['kernel_arg_order'] ...
    out = g.unpack_c(c, m=512)      # (H, 512, 2048) float32
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

# --- pinned contract (identical to the executed Round83 kernel) --------------
PINNED_COMPILE_SOC = "Ascend910B"
PINNED_RUNTIME_SOC = "Ascend910B"
PINNED_CORE_NUM = 30          # live DCMI reported 30 usable cores on all 8 cards
PINNED_L1_SIZE = 1048576
PINNED_L0C_SIZE = 262144
PINNED_MAX_CORES = 30         # never launch more cube tasks than physical cores

#: Round83 module this file's arithmetic was copied from, pinned by SHA-256.
#: The Round83 file itself is never imported-for-edit and never modified.
PINNED_R83_SOURCE = "blocked_dynamic_steps83.py"
PINNED_R83_SHA256 = "b5a06cfdb4c83a2d2f8c84eb56e8794f8a38d2732c4ff5e1fbef0901b520facb"

ROOT = Path("/tmp/h3_round85_batch01")
DEFAULT_OUT_DIR = ROOT / "kernels"

DEFAULT_MB, DEFAULT_NB, DEFAULT_KB = 128, 128, 128

#: Documented target geometries: (H, M, K, N). N and K are tile multiples.
TARGET_SHAPES = {
    "score": (8, 512, 128, 2048),
    "score_last_chunk": (8, 507, 128, 2048),
    "pv": (8, 512, 2048, 128),
    "pv_last_chunk": (8, 507, 2048, 128),
}

#: Smallest geometry that proves real K accumulation AND distinct per-head B.
SUGGESTED_SMOKE_GEOMETRY = (2, 128, 256, 128)


class BatchGemmError(RuntimeError):
    """Raised instead of guessing when a shape or the environment is unexpected."""


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Pure-python tiling plan (stdlib only, unit-testable without TIK or torch)
# ---------------------------------------------------------------------------


def plan_tiling(batch: int, m: int, k: int, n: int,
                mb: int = DEFAULT_MB, nb: int = DEFAULT_NB,
                kb: int = DEFAULT_KB) -> dict:
    """Return every number the kernel and the host packing helpers must agree on.

    Deliberately torch-free and TIK-free so the DMA strides, the per-head GM
    offsets and the flattened task mapping can be checked with neither installed.
    """
    for name, value in (("batch", batch), ("m", m), ("k", k), ("n", n),
                        ("mb", mb), ("nb", nb), ("kb", kb)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise BatchGemmError(f"{name} must be a python int, got {type(value).__name__}")
        if value <= 0:
            raise BatchGemmError(f"{name} must be positive, got {value}")
    for name, value, unit in (("mb", mb, 16), ("nb", nb, 16), ("kb", kb, 16)):
        if value % unit:
            raise BatchGemmError(f"{name}={value} must be a multiple of {unit}")
    if k % kb:
        raise BatchGemmError(f"k={k} is not a multiple of the K tile kb={kb}")
    if n % nb:
        raise BatchGemmError(f"n={n} is not a multiple of the N tile nb={nb}")

    m_pad = _ceil_to(m, mb)
    k1, n1 = k // 16, n // 16
    m_tiles, n_tiles = m_pad // mb, n // nb
    per_head_tasks = m_tiles * n_tiles
    tasks = batch * per_head_tasks
    cores = min(PINNED_MAX_CORES, tasks)
    steps = (tasks + cores - 1) // cores
    k_tiles = k // kb
    kb1 = kb // 16

    # Per-head dense GM strides, in elements. The head stride is what keeps each
    # head's B distinct: it is added to every A/B/C offset in the kernel.
    head_stride = {
        "a": k1 * m_pad * 16,
        "b": k1 * n * 16,
        "c": n1 * m_pad * 16,
    }
    gm_shapes = {
        "a": [batch, k1, m_pad, 16],
        "b": [batch, k1, n, 16],
        "c": [batch, n1, m_pad, 16],
        "aux": [16],
    }
    gm_elems = {
        "a": batch * head_stride["a"],
        "b": batch * head_stride["b"],
        "c": batch * head_stride["c"],
    }
    gm_bytes = {
        "a": gm_elems["a"] * 2,
        "b": gm_elems["b"] * 2,
        "c": gm_elems["c"] * 4,
    }

    # DMA descriptions are per-head and therefore unchanged from Round83: within
    # one head the layout is dense, and data_move burst/strides are in 32-byte
    # units (a full K0=16 row of Half is exactly one unit; FP32 is two).
    dma = {
        "a_load": {"nburst": kb1, "burst": mb, "src_stride": m_pad - mb, "dst_stride": 0},
        "b_load": {"nburst": kb1, "burst": nb, "src_stride": n - nb, "dst_stride": 0},
        "c_store": {"nburst": nb // 16, "burst": 2 * mb, "src_stride": 0,
                    "dst_stride": 2 * (m_pad - mb)},
        "l0c_to_ub": {"block_mode": "m", "nburst": 1,
                      "burst_len_kb": mb * nb * 4 // 1024},
    }

    buffers = {
        "l1_a_bytes": kb1 * mb * 16 * 2,
        "l1_b_bytes": kb1 * nb * 16 * 2,
        "l0c_bytes": (nb // 16) * mb * 16 * 4,
        "ub_bytes": (nb // 16) * mb * 16 * 4,
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
        "accumulation": {
            "k_tiles": k_tiles,
            "single_l0c_across_all_k_blocks": True,
            "init_first_k_block_only": True,
            "half_partial": False,
            "fp32_ub_partial_sum": False,
            "final_output_dtype": "float32",
        },
        "batched": {
            "per_head_distinct_b": True,
            "head_offset_applied_to_a_b_c": True,
            "task_flatten": "head*(m_tiles*n_tiles) + m_tile*n_tiles + n_tile",
        },
        "kernel_arg_order": ["a_gm_float16", "b_gm_float16", "aux0_gm_float16",
                            "aux1_gm_float16", "c_gm_float32"],
        "limits": {"l1_size": PINNED_L1_SIZE, "l0c_size": PINNED_L0C_SIZE,
                   "max_cores": PINNED_MAX_CORES},
    }


def _check_budget(plan: dict) -> None:
    buf = plan["buffers"]
    if buf["l1_a_bytes"] + buf["l1_b_bytes"] > PINNED_L1_SIZE:
        raise BatchGemmError(
            f"L1 need {buf['l1_a_bytes'] + buf['l1_b_bytes']} B exceeds {PINNED_L1_SIZE} B")
    if buf["l0c_bytes"] > PINNED_L0C_SIZE:
        raise BatchGemmError(f"L0C need {buf['l0c_bytes']} B exceeds {PINNED_L0C_SIZE} B")
    if buf["ub_bytes"] > 196608:
        raise BatchGemmError(f"UB need {buf['ub_bytes']} B exceeds 192 KiB")


# --- task / offset arithmetic, pure python (checked by the stdlib tests) -----


def flatten_task(task: int, batch: int, n_tiles: int,
                 per_head_tasks: int) -> tuple[int, int, int]:
    """``task -> (head, m_tile, n_tile)`` for the round-robin flattened grid.

    ``task = head*(m_tiles*n_tiles) + m_tile*n_tiles + n_tile``. The mapping is
    bijective over ``[0, batch*per_head_tasks)``, which is what guarantees no
    head is skipped and no head is visited twice.
    """
    if not isinstance(task, int) or isinstance(task, bool):
        raise BatchGemmError(f"task must be a python int, got {type(task).__name__}")
    total = batch * per_head_tasks
    if not 0 <= task < total:
        raise BatchGemmError(f"task {task} outside [0, {total})")
    head, local = divmod(task, per_head_tasks)
    m_tile, n_tile = divmod(local, n_tiles)
    return head, m_tile, n_tile


def gm_offsets(plan: dict, head: int, m_tile: int, n_tile: int, kt: int) -> dict:
    """Absolute GM element offsets for one (head, tile, K block).

    The head stride is included in all three offsets, so a head can never read
    another head's A/B or write another head's C.
    """
    if not 0 <= head < plan["logical"]["batch"]:
        raise BatchGemmError(f"head {head} outside batch {plan['logical']['batch']}")
    gr, pd, tl = plan["grid"], plan["padded"], plan["tiles"]
    if not 0 <= m_tile < gr["m_tiles"]:
        raise BatchGemmError(f"m_tile {m_tile} outside {gr['m_tiles']}")
    if not 0 <= n_tile < gr["n_tiles"]:
        raise BatchGemmError(f"n_tile {n_tile} outside {gr['n_tiles']}")
    if not 0 <= kt < plan["accumulation"]["k_tiles"]:
        raise BatchGemmError(f"kt {kt} outside K tiles")
    mb, nb = tl["mb"], tl["nb"]
    kb1 = tl["kb"] // 16
    m_pad, n, k1, n1 = pd["m_pad"], plan["logical"]["n"], pd["k1"], pd["n1"]
    base_m, base_n = m_tile * mb, n_tile * nb
    hs = plan["head_stride_elems"]
    return {
        "a": head * hs["a"] + (kt * kb1 * m_pad + base_m) * 16,
        "b": head * hs["b"] + (kt * kb1 * n + base_n) * 16,
        "c": head * hs["c"] + ((base_n // 16) * m_pad + base_m) * 16,
        "c_elements": (nb // 16) * mb * 16,
    }


# ---------------------------------------------------------------------------
# Environment guards (lazy: importing this module must not need CANN or torch)
# ---------------------------------------------------------------------------


def capability_report() -> dict:
    """Query the two facts this kernel depends on, without building anything."""
    import tbe.common.platform as platform

    platform.set_current_compile_soc_info(PINNED_COMPILE_SOC)
    resolved = platform.get_soc_spec("FULL_SOC_VERSION")
    if resolved != PINNED_COMPILE_SOC:
        raise BatchGemmError(f"compile SoC resolved to {resolved!r}, expected "
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
        raise BatchGemmError("Intrinsic_mmad/f162f32 unsupported: the FP32 accumulator "
                             "route this kernel needs is unavailable and must not be faked")
    if report["core_num"] != PINNED_CORE_NUM:
        raise BatchGemmError(
            f"compiler CORE_NUM is {report['core_num']}, expected {PINNED_CORE_NUM}. The live "
            "DCMI audit reports 30 usable cores on all 8 original 910B chips; 32 is the older "
            "compiler default and 32 GiB of memory is not 32 cores.")
    if report["l1_size"] != PINNED_L1_SIZE or report["l0c_size"] != PINNED_L0C_SIZE:
        raise BatchGemmError("L1/L0C sizes do not match the pinned 910A-generation part")
    if report["ub_size"] is not None and report["ub_size"] < 196608:
        raise BatchGemmError(f"UB_SIZE {report['ub_size']} too small for the tile")
    return report


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------


def kernel_name_for(plan: dict) -> str:
    t, g, lg = plan["tiles"], plan["grid"], plan["logical"]
    return (f"batch_gemm85_h{lg['batch']}_m{lg['m']}_k{lg['k']}_n{lg['n']}"
            f"_mb{t['mb']}_nb{t['nb']}_kb{t['kb']}_c{g['cores']}")


def build_kernel(batch: int, m: int, k: int, n: int, out_dir=None,
                 mb: int = DEFAULT_MB, nb: int = DEFAULT_NB, kb: int = DEFAULT_KB) -> dict:
    """Build (or reuse) one batched FP32-output GEMM and describe it.

    Returns metadata with the same keys the deployed ``runtime5gm83`` bridge
    already consumes (``obj_path``, ``obj_sha256``, ``kernel_symbol``,
    ``block_dim``, ``dst_dtype``, GM shapes/bytes) plus the new batch fields.
    ``'executed': False`` is always set: building says nothing about numerics.
    """
    plan = plan_tiling(batch, m, k, n, mb=mb, nb=nb, kb=kb)
    _check_budget(plan)

    from tbe import tik

    capability = capability_report()
    if capability["ub_size"] is not None and plan["buffers"]["ub_bytes"] > capability["ub_size"]:
        raise BatchGemmError("UB tile does not fit the reported UB size")

    name = kernel_name_for(plan)
    out_root = Path(out_dir).resolve() if out_dir is not None else DEFAULT_OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec85.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability)

    with (out_root / ".compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability)

        work = Path(tempfile.mkdtemp(prefix="batch85_build_", dir=out_root)).resolve()
        if work.parent != out_root:
            raise BatchGemmError("unexpected build workspace")
        t = tik.Tik(disable_debug=True)
        gm = _emit_kernel(t, plan, tik)
        previous = os.getcwd()
        try:
            os.chdir(work)
            # aux operands are passed explicitly: they are unused arithmetic
            # placeholders that keep the proven 5-GM ABI, and auto-collection
            # would prune them.
            t.BuildCCE(kernel_name=name,
                       inputs=[gm["a"], gm["b"], gm["aux0"], gm["aux1"]],
                       outputs=[gm["c"]])
        finally:
            os.chdir(previous)
        built_obj = work / "kernel_meta" / (name + ".o")
        built_meta = work / "kernel_meta" / (name + ".json")
        if not built_obj.is_file() or not built_meta.is_file():
            # Retain the exact build workspace for failure diagnosis; never
            # delete or move generated paths based only on a string prefix.
            raise BatchGemmError(f"BuildCCE produced no kernel_meta artefacts for {name}")
        if dest.exists():
            raise BatchGemmError(f"refusing to overwrite existing kernel directory {dest}")
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
        raise BatchGemmError(
            f"cached kernel {name} has blockDim {meta.get('blockDim')} but the plan needs "
            f"{plan['grid']['cores']} cores; refusing to reuse a mismatched binary")
    obj_sha = hashlib.sha256(obj.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec = {
        "round": 85,
        "scope": ("batched per-head FP16xFP16 -> FP32 L0C -> FP32 GM; "
                  "no bias; no Half partial"),
        "executed": False,
        "compiled_now": compiled_now,
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "kernel_name": name,
        "kernel_symbol": meta["kernelName"],
        "block_dim": int(meta["blockDim"]),
        "core_type": meta.get("coreType"),
        "obj_path": str(obj),
        "obj_sha256": obj_sha,
        "json_path": str(meta_path),
        "source_sha256": source_sha,
        "reference_r83_source_sha256": PINNED_R83_SHA256,
        "dtype": "float16_in_float32_out",
        # R83/R82-compatible field names so the same bridge/probe checks work.
        "dst_dtype": "float32",
        "uses_fp32_bias": False,
        "input_dtype": "float16",
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
        "kernel_arg_order": plan["kernel_arg_order"],
        "GM_argument_count": 5,
        "logical_shape": plan["logical"],
        "batch": plan["logical"]["batch"],
        "padded_shape": {"m_pad": plan["padded"]["m_pad"], "k1": plan["padded"]["k1"],
                         "n1": plan["padded"]["n1"]},
        "tiles": plan["tiles"],
        "grid": plan["grid"],
        "head_stride_elems": plan["head_stride_elems"],
        "gm_shapes": plan["gm_shapes"],
        "gm_elems": plan["gm_elems"],
        "gm_bytes": plan["gm_bytes"],
        "dma": plan["dma"],
        "buffers": plan["buffers"],
        "accumulation": plan["accumulation"],
        "batched": plan["batched"],
        "capability": capability,
        "suggested_smoke_geometry": list(SUGGESTED_SMOKE_GEOMETRY),
        "no_bias_operand": True,
        "no_fp32_weight_shadow": True,
        "per_head_distinct_b_required": True,
        "host_packing_required": ["pack_a", "pack_w"],
        "host_unpacking_required": ["unpack_c"],
        "host_padding": {
            "m_pad_rows": plan["padded"]["m_pad_rows"],
            "a_zero_m_padding": True,
            "c_pad_rows_zero_or_cropped": "cropped by unpack_c",
            "score_n_padding": "host zero-fills padded key rows; kernel has no mask",
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec


def _emit_kernel(t, plan: dict, tik) -> dict:
    """Emit the TIK program. Every offset here is derived from `plan`.

    Design notes that matter for review:

    * ``init_l1out`` must be a Python ``bool`` (the SDK checks
      ``check_type_match(self.init_l1out, bool, ...)``), so it cannot be a
      dynamic scalar. The K loop is therefore a *static* Python loop and only
      ``kt == 0`` initialises L0C; every later block accumulates. K is fixed per
      kernel, so this does not grow with the batch dimension.
    * One single ``c0`` L0C tensor is allocated outside the K loop and stays live
      across all K blocks of that task: no Half partial, no FP32-UB partial sum.
    * Each K block is wrapped in ``new_stmt_scope(disable_sync=False)``, the same
      sync boundary the deployed TIK flash-attention helper uses around
      ``matmul`` + ``tensor_mov``; that orders the L1 re-load (WAR) against the
      previous cube read.
    * Only the outer tile-step iteration is dynamic (``for_range``); the head
      dimension is folded into the flattened task index, so the compiled IR does
      not grow with the number of heads.
    * Zero/padded handling is host-side only: padded A rows are zero, and padded
      C rows are cropped by the host. No other OOB access exists because Mp is a
      multiple of mb and N/K are tile multiples.
    """
    lg, tl, pd, gr, dma = (plan["logical"], plan["tiles"], plan["padded"],
                           plan["grid"], plan["dma"])
    batch, m_pad, n, k1, n1 = lg["batch"], pd["m_pad"], lg["n"], pd["k1"], pd["n1"]
    mb, nb, kb = tl["mb"], tl["nb"], tl["kb"]
    kb1 = kb // 16
    k_tiles = plan["accumulation"]["k_tiles"]
    cores, steps, tasks = gr["cores"], gr["steps"], gr["tasks"]
    n_tiles, per_head_tasks = gr["n_tiles"], gr["per_head_tasks"]
    hs = plan["head_stride_elems"]
    if per_head_tasks != gr["m_tiles"] * n_tiles:
        raise BatchGemmError("per-head task count disagrees with the tile grid")

    ag = t.Tensor("float16", (batch, k1, m_pad, 16), name="a_gm", scope=tik.scope_gm)
    bg = t.Tensor("float16", (batch, k1, n, 16), name="b_gm", scope=tik.scope_gm)
    aux0 = t.Tensor("float16", (16,), name="aux0_gm", scope=tik.scope_gm)
    aux1 = t.Tensor("float16", (16,), name="aux1_gm", scope=tik.scope_gm)
    cg = t.Tensor("float32", (batch, n1, m_pad, 16), name="c_gm", scope=tik.scope_gm)

    with t.for_range(0, cores, block_num=cores) as core:
        al = t.Tensor("float16", (kb1, mb, 16), name="a_l1", scope=tik.scope_cbuf)
        bl = t.Tensor("float16", (kb1, nb, 16), name="b_l1", scope=tik.scope_cbuf)
        # "local.L0C": the SDK requires dst.scope == scope_cbuf_out for matmul.
        c0 = t.Tensor("float32", (nb // 16, mb, 16), name="c_l0c", scope=tik.scope_cc)
        cu = t.Tensor("float32", (nb // 16, mb, 16), name="c_ub", scope=tik.scope_ubuf)

        with t.for_range(0, steps) as step:
            task = core + step * cores
            with t.if_scope(task < tasks):
                head = task // per_head_tasks
                local = task - head * per_head_tasks
                mt_idx = local // n_tiles
                nt_idx = local - mt_idx * n_tiles
                base_m = mt_idx * mb
                base_n = nt_idx * nb
                # Absolute per-head GM offsets (head stride included).
                head_a = head * hs["a"]
                head_b = head * hs["b"]
                head_c = head * hs["c"]
                for kt in range(k_tiles):
                    a_off = head_a + (kt * kb1 * m_pad + base_m) * 16
                    b_off = head_b + (kt * kb1 * n + base_n) * 16
                    with t.new_stmt_scope(disable_sync=False):
                        t.data_move(al, ag[a_off], 0, dma["a_load"]["nburst"],
                                    dma["a_load"]["burst"], dma["a_load"]["src_stride"],
                                    dma["a_load"]["dst_stride"])
                        t.data_move(bl, bg[b_off], 0, dma["b_load"]["nburst"],
                                    dma["b_load"]["burst"], dma["b_load"]["src_stride"],
                                    dma["b_load"]["dst_stride"])
                        t.matmul(c0, al, bl, mb, kb, nb, init_l1out=(kt == 0))
                c_off = head_c + ((base_n // 16) * m_pad + base_m) * 16
                with t.new_stmt_scope(disable_sync=False):
                    t.tensor_mov(cu, c0, dma["l0c_to_ub"]["block_mode"],
                                 dma["l0c_to_ub"]["nburst"],
                                 dma["l0c_to_ub"]["burst_len_kb"], 0, 0)
                    t.data_move(cg[c_off], cu, 0, dma["c_store"]["nburst"],
                                dma["c_store"]["burst"], dma["c_store"]["src_stride"],
                                dma["c_store"]["dst_stride"])
    return {"a": ag, "b": bg, "aux0": aux0, "aux1": aux1, "c": cg}


# ---------------------------------------------------------------------------
# Host packing helpers (lazy torch; never executed by this module's own tests)
# ---------------------------------------------------------------------------


def _shape_of(value, name: str, ndim: int) -> tuple:
    """Return the tensor shape, failing closed *before* importing torch.

    A plain Python object (no ``shape``) or a tensor of the wrong rank is a
    caller error, not a numeric condition, so it must be rejected cheaply.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        raise BatchGemmError(f"{name} expects a tensor-like operand with .shape, got "
                             f"{type(value).__name__}")
    dims = tuple(int(v) for v in shape)
    if len(dims) != ndim:
        raise BatchGemmError(f"{name} expects {ndim}-D input, got {dims}")
    return dims


def pack_a(a):
    """``(H, M, K)`` Half -> ``(H, K1, Mp, 16)`` Half with zero padded M rows.

    Only element order and zero M padding: no arithmetic, no dtype change, no
    FP32 shadow. Head ``h``, row ``m``, column ``k = k1*16+k0`` lands at
    ``[h, k1, m, k0]``.
    """
    h, m, k = _shape_of(a, "pack_a (H,M,K)", 3)
    import torch

    if a.dtype != torch.float16:
        raise BatchGemmError(f"pack_a expects original float16 values, got {a.dtype}")
    if k % 16:
        raise BatchGemmError(f"K={k} must be a multiple of 16")
    m_pad = _ceil_to(m, DEFAULT_MB)
    packed = a.detach().to(torch.float16).contiguous()
    out = torch.zeros((h, k // 16, m_pad, 16), dtype=torch.float16, device=packed.device)
    out[:, :, :m, :] = packed.reshape(h, m, k // 16, 16).permute(0, 2, 1, 3)
    return out.contiguous()


def pack_w(weight):
    """``(H, N, K)`` Half -> ``(H, K1, N, 16)`` Half, values untouched.

    Each head's matrix is packed separately and densely; the head stride makes
    the head's B addressable without ever mixing heads' weights.
    """
    h, n, k = _shape_of(weight, "pack_w (H,N,K)", 3)
    import torch

    if weight.dtype != torch.float16:
        raise BatchGemmError(f"pack_w expects original float16 values, got {weight.dtype}")
    if k % 16:
        raise BatchGemmError(f"K={k} must be a multiple of 16")
    packed = weight.detach().to(torch.float16).contiguous()
    return packed.reshape(h, n, k // 16, 16).permute(0, 2, 1, 3).contiguous()


def pack_p(p):
    """Alias for :func:`pack_a` used for the float probability operand.

    P is passed as an original Half matrix for the PV product; any FP32->Half
    decomposition (high + scaled residual) is the caller's concern and is
    launched as separate operator calls before the FP32 add.
    """
    return pack_a(p)


def unpack_c(c, m: int):
    """``(H, N1, Mp, 16)`` float32 -> ``(H, M, N)`` float32; crops padded rows."""
    shape = _shape_of(c, "unpack_c (H,N1,Mp,16)", 4)
    if shape[3] != 16:
        raise BatchGemmError(f"unpack_c expects a trailing K0=16 axis, got {shape}")
    import torch

    if c.dtype != torch.float32:
        raise BatchGemmError(f"unpack_c expects the FP32 kernel output, got {c.dtype}")
    h, n1, m_pad = shape[:3]
    if not 0 < m <= m_pad:
        raise BatchGemmError(f"m={m} outside padded rows 1..{m_pad}")
    return c[:, :, :m, :].permute(0, 2, 1, 3).reshape(h, m, n1 * 16).contiguous()


def check_reference_r83_source(path) -> str:
    """Verify the Round83 source this file copied its arithmetic from is intact."""
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != PINNED_R83_SHA256:
        raise BatchGemmError(
            f"{PINNED_R83_SOURCE} changed: {actual} != {PINNED_R83_SHA256}; this Round85 "
            "kernel's arithmetic assumption no longer matches the executed reference")
    return actual


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="print tiling plans only")
    parser.add_argument("--capability", action="store_true", help="query SoC capability")
    parser.add_argument("--build", nargs=4, type=int, metavar=("H", "M", "K", "N"),
                        help="build one kernel for these logical dimensions")
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