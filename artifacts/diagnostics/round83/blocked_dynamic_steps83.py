#!/usr/bin/env python3
"""Round83 blocked FP16xA x FP16xB -> FP32 L0C -> FP32 GM GEMM for H3 shapes.

SOURCE ONLY. This module has never been compiled or executed. Its author was
required to run no NPU/GPU work, no remote compile and no model execution, so
nothing here is a precision result, a speed result or an H3 improvement claim.
The root agent compiles it under its own supervisor and owns every runtime
receipt.

Why this kernel exists
----------------------
Round80 measured two real deviations in the native path:

* native biased ``Half`` Linear equals a no-bias Half GEMM plus a *Half* bias
  add, i.e. one extra output rounding the CPU FP16 reference does not do;
* tensor-parallel row projections round the FP16 partial GEMM output to FP16
  before the FP32 collective, and a FP32 all-reduce cannot restore bits that
  were already rounded away.

Round82 then proved on this exact device that a direct TIK kernel can express
FP16 operands with an FP32 accumulator and an FP32 GM destination, and really
executed it (M128/K512/N128; repeated outputs bit-identical; 4.081e-7 relative
RMS against an FP32 accumulator diagnostic). That micro kernel took the whole
A and B operands as single L1 tiles, which is why it was fixed to 128x512x128.

This module keeps that arithmetic but adds explicit blocked tiling so the real
H3 row shapes fit: the L0C accumulator stays FP32 and alive across *all* K
blocks, so there is no Half partial and no FP32-UB partial sum anywhere.

Layouts (identical convention to the executed Round82 kernel)
-------------------------------------------------------------
* A GM: ``(K1, Mp, 16)`` float16, element ``(k1, m, k0) = A[m, k1*16+k0]``
* B GM: ``(K1, N, 16)``  float16, element ``(k1, n, k0) = W[n, k1*16+k0]``
* C GM: ``(N1, Mp, 16)`` float32, element ``(n1, m, n0) = C[m, n1*16+n0]``

``Mp = ceil(M/128)*128``; rows ``M..Mp-1`` of A are zero, so the padded rows
produce exact zero output rows and the host crops them. N and K must be
multiples of the tile sizes; every H3 shape below satisfies that.

GM argument ABI (5 pointers, unchanged from the proven Round82 / fa910a
launcher): ``[A_Half, B_Half, aux0_Half16, aux1_Half16, C_Float32]``.

Usage (root driver, inside the private NPU run)
-----------------------------------------------
    import blocked_fp32_gemm83 as g
    spec = g.build_kernel(2043, 5120, 128)          # compiles once, cached
    a = g.pack_a(x_half_mk)                          # (K1, Mp, 16) Half
    b = g.pack_w(w_half_nk)                          # (K1, N, 16)  Half
    c = torch.empty(spec['gm_shapes']['c'], dtype=torch.float32, device=dev)
    ... launch 5 GM pointers per the ABI in spec['kernel_arg_order'] ...
    out = g.unpack_c(c, m=2043)                      # (M, N) float32
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

# --- pinned contract ---------------------------------------------------------
# Measured on this device: the short TIK SoC is Ascend910 while the full/runtime
# label is Ascend910B. Bias is gated on a *different* check that rejects this
# part, which is why this kernel has no bias operand at all (a sibling task owns
# the portable FP32-UB bias epilogue).
PINNED_COMPILE_SOC = "Ascend910B"
PINNED_RUNTIME_SOC = "Ascend910B"
PINNED_CORE_NUM = 30          # live DCMI reported 30 usable cores on all 8 cards
PINNED_L1_SIZE = 1048576
PINNED_L0C_SIZE = 262144
PINNED_MAX_CORES = 30         # never launch more cube tasks than physical cores

ROOT = Path("/tmp/h3_round83_blocked01")
DEFAULT_OUT_DIR = ROOT / "kernels"

# Initial tiling. All three stay configurable, but the H3 shapes only need 128.
DEFAULT_MB, DEFAULT_NB, DEFAULT_KB = 128, 128, 128

# Documented target geometries (M, K, N). N and K are tile multiples for all.
H3_SHAPES = {
    "kv_proj": (2043, 5120, 128),
    "q_proj": (2043, 5120, 1024),
    "mlp_up": (2043, 5120, 3456),
    "row_parallel": (2043, 3456, 5120),
}

# Smallest geometry that exercises real K-block accumulation (2 K tiles).
SUGGESTED_SMOKE_GEOMETRY = (128, 256, 128)


class BlockedGemmError(RuntimeError):
    """Raised instead of guessing when a shape or the environment is unexpected."""


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# Pure-python tiling plan (stdlib only, unit-testable without TIK or torch)
# ---------------------------------------------------------------------------


def plan_tiling(m: int, k: int, n: int, mb: int = DEFAULT_MB, nb: int = DEFAULT_NB,
                kb: int = DEFAULT_KB) -> dict:
    """Return every number the kernel and the host packing helpers must agree on.

    Deliberately torch-free and TIK-free so the arithmetic in the DMA strides can
    be verified on a machine with neither installed.
    """
    for name, value in (("m", m), ("k", k), ("n", n), ("mb", mb), ("nb", nb), ("kb", kb)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise BlockedGemmError(f"{name} must be a python int, got {type(value).__name__}")
        if value <= 0:
            raise BlockedGemmError(f"{name} must be positive, got {value}")
    for name, value, unit in (("mb", mb, 16), ("nb", nb, 16), ("kb", kb, 16)):
        if value % unit:
            raise BlockedGemmError(f"{name}={value} must be a multiple of {unit}")
    if k % kb:
        raise BlockedGemmError(f"k={k} is not a multiple of the K tile kb={kb}")
    if n % nb:
        raise BlockedGemmError(f"n={n} is not a multiple of the N tile nb={nb}")

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
        "c": [n1, m_pad, 16],
        "aux": [16],
    }
    gm_elems = {
        "a": k1 * m_pad * 16,
        "b": k1 * n * 16,
        "c": n1 * m_pad * 16,
    }
    gm_bytes = {
        "a": gm_elems["a"] * 2,
        "b": gm_elems["b"] * 2,
        "c": gm_elems["c"] * 4,
    }

    # DMA descriptions. For data_move the burst length and both strides are in
    # units of 32 bytes; a full K0=16 row of Half is exactly 32 bytes, so one
    # burst of `mb` rows is `mb` units. The gap between consecutive k1 blocks in
    # A is one whole K1 plane = m_pad*32 bytes = m_pad units, of which we consume
    # `mb`, hence src_stride = m_pad - mb (no factor of 2: the row is already
    # 32 bytes because K0 = 16 Half elements).
    dma = {
        "a_load": {"nburst": kb1, "burst": mb, "src_stride": m_pad - mb, "dst_stride": 0},
        "b_load": {"nburst": kb1, "burst": nb, "src_stride": n - nb, "dst_stride": 0},
        # C store: a full FP32 row of 16 elements is 64 bytes = 2 units, so the
        # per-group gap doubles: dst_stride = 2*(m_pad - mb), burst = 2*mb.
        "c_store": {"nburst": nb // 16, "burst": 2 * mb, "src_stride": 0,
                    "dst_stride": 2 * (m_pad - mb)},
        # L0C -> UB keeps the (N1, MB, 16) FP32 block intact; tensor_mov
        # burst_len is in units of 1024 bytes.
        "l0c_to_ub": {"block_mode": "m", "nburst": 1, "burst_len_kb": mb * nb * 4 // 1024},
    }

    buffers = {
        "l1_a_bytes": kb1 * mb * 16 * 2,
        "l1_b_bytes": kb1 * nb * 16 * 2,
        "l0c_bytes": (nb // 16) * mb * 16 * 4,
        "ub_bytes": (nb // 16) * mb * 16 * 4,
    }

    return {
        "logical": {"m": m, "k": k, "n": n},
        "tiles": {"mb": mb, "nb": nb, "kb": kb},
        "padded": {"m_pad": m_pad, "m_pad_rows": m_pad - m, "k1": k1, "n1": n1},
        "grid": {"m_tiles": m_tiles, "n_tiles": n_tiles, "tasks": tasks,
                 "cores": cores, "steps": steps},
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
        "kernel_arg_order": ["a_gm_float16", "b_gm_float16", "aux0_gm_float16",
                             "aux1_gm_float16", "c_gm_float32"],
        "limits": {"l1_size": PINNED_L1_SIZE, "l0c_size": PINNED_L0C_SIZE,
                   "max_cores": PINNED_MAX_CORES},
    }


def _check_budget(plan: dict) -> None:
    buf = plan["buffers"]
    l1_need = buf["l1_a_bytes"] + buf["l1_b_bytes"]
    if l1_need > PINNED_L1_SIZE:
        raise BlockedGemmError(f"L1 need {l1_need} B exceeds {PINNED_L1_SIZE} B")
    if buf["l0c_bytes"] > PINNED_L0C_SIZE:
        raise BlockedGemmError(f"L0C need {buf['l0c_bytes']} B exceeds {PINNED_L0C_SIZE} B")
    # Keep a conservative UB bound; the exact value is asserted at build time.
    if buf["ub_bytes"] > 196608:
        raise BlockedGemmError(f"UB need {buf['ub_bytes']} B exceeds 192 KiB")


# ---------------------------------------------------------------------------
# Environment guards (lazy: importing this module must not need CANN or torch)
# ---------------------------------------------------------------------------


def capability_report() -> dict:
    """Query the two facts this kernel depends on, without building anything."""
    import tbe.common.platform as platform

    platform.set_current_compile_soc_info(PINNED_COMPILE_SOC)
    resolved = platform.get_soc_spec("FULL_SOC_VERSION")
    if resolved != PINNED_COMPILE_SOC:
        raise BlockedGemmError(f"compile SoC resolved to {resolved!r}, expected "
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
        raise BlockedGemmError("Intrinsic_mmad/f162f32 unsupported: the FP32 accumulator "
                               "route this kernel needs is unavailable and must not be faked")
    if report["core_num"] != PINNED_CORE_NUM:
        raise BlockedGemmError(
            f"compiler CORE_NUM is {report['core_num']}, expected {PINNED_CORE_NUM}. The live "
            "DCMI audit reports 30 usable cores on all 8 original 910B chips; 32 is the older "
            "compiler default and 32 GiB of memory is not 32 cores.")
    if report["l1_size"] != PINNED_L1_SIZE or report["l0c_size"] != PINNED_L0C_SIZE:
        raise BlockedGemmError("L1/L0C sizes do not match the pinned 910A-generation part")
    if report["ub_size"] is not None and report["ub_size"] < 196608:
        raise BlockedGemmError(f"UB_SIZE {report['ub_size']} too small for the tile")
    return report


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------


def kernel_name_for(plan: dict) -> str:
    t = plan["tiles"]
    g = plan["grid"]
    lg = plan["logical"]
    return (f"blocked_dynamic83_m{lg['m']}_k{lg['k']}_n{lg['n']}"
            f"_mb{t['mb']}_nb{t['nb']}_kb{t['kb']}_c{g['cores']}")


def build_kernel(m: int, k: int, n: int, out_dir=None,
                 mb: int = DEFAULT_MB, nb: int = DEFAULT_NB, kb: int = DEFAULT_KB) -> dict:
    """Build (or reuse) one blocked FP32-output GEMM kernel and describe it.

    Returns a metadata dict. ``'executed': False`` is always set here: building a
    kernel says nothing about its numerics or speed, and this module never
    launches anything.
    """
    plan = plan_tiling(m, k, n, mb=mb, nb=nb, kb=kb)
    _check_budget(plan)

    from tbe import tik

    capability = capability_report()
    if capability["ub_size"] is not None and plan["buffers"]["ub_bytes"] > capability["ub_size"]:
        raise BlockedGemmError("UB tile does not fit the reported UB size")

    name = kernel_name_for(plan)
    out_root = Path(out_dir).resolve() if out_dir is not None else DEFAULT_OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec83.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability)

    with (out_root / ".compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability)

        work = Path(tempfile.mkdtemp(prefix="blocked83_build_", dir=out_root)).resolve()
        if work.parent != out_root:
            raise BlockedGemmError("unexpected build workspace")
        t = tik.Tik(disable_debug=True)
        gm = _emit_kernel(t, plan, tik)
        previous = os.getcwd()
        try:
            os.chdir(work)
            # inputs/outputs are passed explicitly: the two aux Half operands are
            # deliberately unused arithmetic placeholders that only exist to keep
            # the proven 5-GM-pointer ABI, and auto-collection would prune them.
            t.BuildCCE(kernel_name=name,
                       inputs=[gm["a"], gm["b"], gm["aux0"], gm["aux1"]],
                       outputs=[gm["c"]])
        finally:
            os.chdir(previous)
        built_obj = work / "kernel_meta" / (name + ".o")
        built_meta = work / "kernel_meta" / (name + ".json")
        if not built_obj.is_file() or not built_meta.is_file():
            # Retain the exact build workspace for failure diagnosis;do not
            # delete or move generated paths based only on a string prefix.
            raise BlockedGemmError(f"BuildCCE produced no kernel_meta artefacts for {name}")
        if dest.exists():
            raise BlockedGemmError(f"refusing to overwrite existing kernel directory {dest}")
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
        raise BlockedGemmError(
            f"cached kernel {name} has blockDim {meta.get('blockDim')} but the plan needs "
            f"{plan['grid']['cores']} cores; refusing to reuse a mismatched binary")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec = {
        "round": 83,
        "scope": "blocked FP16xFP16 -> FP32 L0C -> FP32 GM;no bias;no Half partial",
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
        "obj_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
        "json_path": str(meta_path),
        "source_sha256": source_sha,
        "dtype": "float16_in_float32_out",
        # R82-compatible field names so the same runtime bridge/probe checks work.
        "dst_dtype": "float32",
        "uses_fp32_bias": False,
        "input_dtype": "float16",
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
        "kernel_arg_order": plan["kernel_arg_order"],
        "GM_argument_count": 5,
        "logical_shape": plan["logical"],
        "padded_shape": {"m_pad": plan["padded"]["m_pad"], "k1": plan["padded"]["k1"],
                         "n1": plan["padded"]["n1"]},
        "tiles": plan["tiles"],
        "grid": plan["grid"],
        "gm_shapes": plan["gm_shapes"],
        "gm_elems": plan["gm_elems"],
        "gm_bytes": plan["gm_bytes"],
        "dma": plan["dma"],
        "buffers": plan["buffers"],
        "accumulation": plan["accumulation"],
        "capability": capability,
        "suggested_smoke_geometry": list(SUGGESTED_SMOKE_GEOMETRY),
        "no_bias_operand": True,
        "no_fp32_weight_shadow": True,
        "host_packing_required": ["pack_a", "pack_w"],
        "host_unpacking_required": ["unpack_c"],
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec


def _emit_kernel(t, plan: dict, tik) -> dict:
    """Emit the TIK program. Every offset here is derived from `plan`.

    Design notes that matter for review:

    * ``init_l1out`` must be a Python ``bool`` (the SDK checks
      ``check_type_match(self.init_l1out, bool, ...)``), so it cannot be a
      dynamic scalar. The K loop is therefore a *static* Python loop and only
      ``kt == 0`` initialises L0C; every later block accumulates. This is the
      same arithmetic the task described with an ``if_scope`` duplication, with
      less IR and no dynamic branch around a cube instruction.
    * One single ``c0`` L0C tensor is allocated outside the K loop and stays live
      across all K blocks: there is no Half partial and no FP32-UB partial sum.
    * Each K block is wrapped in ``new_stmt_scope(disable_sync=False)``, the same
      sync boundary the deployed TIK flash-attention helper uses around
      ``matmul`` + ``tensor_mov``; that is what orders the L1 re-load (WAR)
      against the previous cube read.
    * The only trailing/zero handling is: padded A rows are zero on the host, and
      the padded C rows are cropped on the host. No other OOB access exists
      because Mpad is a multiple of mb and N/K are tile multiples.
    """
    lg, tl, pd, gr, dma = plan["logical"], plan["tiles"], plan["padded"], plan["grid"], plan["dma"]
    m_pad, n, k1 = pd["m_pad"], lg["n"], pd["k1"]
    mb, nb, kb = tl["mb"], tl["nb"], tl["kb"]
    kb1 = kb // 16
    k_tiles = plan["accumulation"]["k_tiles"]
    cores, steps, tasks, n_tiles = gr["cores"], gr["steps"], gr["tasks"], gr["n_tiles"]

    ag = t.Tensor("float16", (k1, m_pad, 16), name="a_gm", scope=tik.scope_gm)
    bg = t.Tensor("float16", (k1, n, 16), name="b_gm", scope=tik.scope_gm)
    aux0 = t.Tensor("float16", (16,), name="aux0_gm", scope=tik.scope_gm)
    aux1 = t.Tensor("float16", (16,), name="aux1_gm", scope=tik.scope_gm)
    cg = t.Tensor("float32", (pd["n1"], m_pad, 16), name="c_gm", scope=tik.scope_gm)

    with t.for_range(0, cores, block_num=cores) as core:
        al = t.Tensor("float16", (kb1, mb, 16), name="a_l1", scope=tik.scope_cbuf)
        bl = t.Tensor("float16", (kb1, nb, 16), name="b_l1", scope=tik.scope_cbuf)
        # "local.L0C": the SDK requires dst.scope == scope_cbuf_out for matmul.
        c0 = t.Tensor("float32", (nb // 16, mb, 16), name="c_l0c", scope=tik.scope_cc)
        cu = t.Tensor("float32", (nb // 16, mb, 16), name="c_ub", scope=tik.scope_ubuf)

        with t.for_range(0, steps) as step:
            task = core + step * cores
            with t.if_scope(task < tasks):
                mb_idx = task // n_tiles
                nb_idx = task % n_tiles
                base_m = mb_idx * mb
                base_n = nb_idx * nb
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
                        t.matmul(c0, al, bl, mb, kb, nb, init_l1out=(kt == 0))
                c_off = ((base_n // 16) * m_pad + base_m) * 16
                with t.new_stmt_scope(disable_sync=False):
                    t.tensor_mov(cu, c0, dma["l0c_to_ub"]["block_mode"],
                                 dma["l0c_to_ub"]["nburst"], dma["l0c_to_ub"]["burst_len_kb"],
                                 0, 0)
                    t.data_move(cg[c_off], cu, 0, dma["c_store"]["nburst"],
                                dma["c_store"]["burst"], dma["c_store"]["src_stride"],
                                dma["c_store"]["dst_stride"])
    return {"a": ag, "b": bg, "aux0": aux0, "aux1": aux1, "c": cg}


# ---------------------------------------------------------------------------
# Host packing helpers (lazy torch; never executed by this module's own tests)
# ---------------------------------------------------------------------------


def pack_a(a):
    """``(M, K)`` float16 ND -> ``(K1, Mp, 16)`` float16 with zero padded rows.

    Only element order and zero M padding: no arithmetic, no dtype change, no
    FP32 shadow. Row ``m`` and column ``k = k1*16 + k0`` land at ``[k1, m, k0]``.
    """
    import torch

    if a.dim() != 2:
        raise BlockedGemmError(f"pack_a expects a 2-D (M,K) tensor, got {tuple(a.shape)}")
    if a.dtype != torch.float16:
        raise BlockedGemmError(f"pack_a expects original float16 values, got {a.dtype}")
    m, k = int(a.shape[0]), int(a.shape[1])
    if k % 16:
        raise BlockedGemmError(f"K={k} must be a multiple of 16")
    m_pad = _ceil_to(m, DEFAULT_MB)
    packed = a.detach().to(torch.float16).contiguous()
    out = torch.zeros((k // 16, m_pad, 16), dtype=torch.float16, device=packed.device)
    out[:, :m, :] = packed.reshape(m, k // 16, 16).transpose(0, 1)
    return out.contiguous()


def pack_w(weight):
    """``(N, K)`` float16 ND -> ``(K1, N, 16)`` float16, values untouched."""
    import torch

    if weight.dim() != 2:
        raise BlockedGemmError(f"pack_w expects a 2-D (N,K) tensor, got {tuple(weight.shape)}")
    if weight.dtype != torch.float16:
        raise BlockedGemmError(f"pack_w expects original float16 values, got {weight.dtype}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if k % 16:
        raise BlockedGemmError(f"K={k} must be a multiple of 16")
    packed = weight.detach().to(torch.float16).contiguous()
    return packed.reshape(n, k // 16, 16).transpose(0, 1).contiguous()


def unpack_c(c, m: int):
    """``(N1, Mp, 16)`` float32 -> ``(M, N)`` float32; crops the padded rows only."""
    import torch

    if c.dim() != 3 or int(c.shape[2]) != 16:
        raise BlockedGemmError(f"unpack_c expects (N1, Mp, 16), got {tuple(c.shape)}")
    if c.dtype != torch.float32:
        raise BlockedGemmError(f"unpack_c expects the FP32 kernel output, got {c.dtype}")
    n1, m_pad = int(c.shape[0]), int(c.shape[1])
    if not 0 < m <= m_pad:
        raise BlockedGemmError(f"m={m} outside padded rows 1..{m_pad}")
    return c[:, :m, :].transpose(0, 1).reshape(m, n1 * 16).contiguous()


class PaddedWeightCache:
    """Per-parameter padded-weight cache with a fail-closed identity signature.

    H3's K and V projections are two different views of the *same* QKV storage,
    so a shared ``data_ptr`` alone must never be treated as a cache collision:
    the signature includes ``storage_offset``. A changed signature for the same
    logical view raises instead of silently re-caching a mutated weight.
    """

    def __init__(self):
        self._entries = {}
        self._stats = {"hits": 0, "misses": 0, "bytes": 0}

    @staticmethod
    def signature(weight) -> tuple:
        return (weight.untyped_storage().data_ptr(), int(weight.storage_offset()),
                tuple(int(v) for v in weight.shape), tuple(int(v) for v in weight.stride()),
                str(weight.dtype), str(weight.device), int(getattr(weight, "_version", -1)))

    def get(self, weight):
        key = self.signature(weight)
        entry = self._entries.get(key)
        if entry is not None:
            self._stats["hits"] += 1
            return entry
        for known in self._entries:
            if known[0] == key[0] and known[1] == key[1] and known != key:
                raise BlockedGemmError(
                    "cached padded weight signature changed (version/shape/stride); refusing "
                    "to silently re-cache a mutated parameter")
        packed = pack_w(weight)
        self._entries[key] = packed
        self._stats["misses"] += 1
        self._stats["bytes"] += int(packed.numel() * packed.element_size())
        return packed

    def stats(self) -> dict:
        return dict(self._stats, entries=len(self._entries))

    def clear(self) -> None:
        self._entries.clear()
        self._stats.update({"hits": 0, "misses": 0, "bytes": 0})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="print the tiling plan only")
    parser.add_argument("--capability", action="store_true", help="query SoC capability")
    parser.add_argument("--build", nargs=3, type=int, metavar=("M", "K", "N"),
                        help="build one kernel for these logical dimensions")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({name: plan_tiling(*shape) for name, shape in H3_SHAPES.items()},
                         indent=2, sort_keys=True))
        return
    if args.capability:
        print(json.dumps(capability_report(), indent=2, sort_keys=True))
        return
    if args.build:
        print(json.dumps(build_kernel(*args.build, out_dir=args.out_dir), indent=2,
                         sort_keys=True))
        return
    parser.error("choose one of --plan, --capability, --build M K N")


if __name__ == "__main__":
    main()