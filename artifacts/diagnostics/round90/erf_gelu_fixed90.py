#!/usr/bin/env python3
"""Round90: Float32-vector erf-GELU TIK kernel for the four vision merger GELUs.

SOURCE ONLY. Nothing here has been compiled, launched or imported on an NPU by
its author: no SSH, no NPU/GPU work, no model run, no CANN build, no install, no
production edit and no commit. ``build_kernel()`` describes a kernel; only the
parent driver may compile and execute it, and every number belongs to that
driver's receipts.

WHY THIS KERNEL EXISTS
----------------------
Round89 measured, on the real frozen activation of the vision mergers, that the
native NPU ``GELU(approximate='none')`` differs from the CPU FP16 reference by
3.10e-4 (``merger.act_fn``) and 4.33e-4 (``deepstack_merger_list.0.act_fn``)
relative RMS, while:

* CPU ``F.gelu`` in Float32 then one Half cast is within 2.89e-7 / 1.13e-6 of
  CPU ``F.gelu`` directly in Half, and
* feeding the native NPU operator a Float32 operand and casting afterwards is
  **bit-identical** to the native Half output.

So the gap is not the caller's cast order: the native NPU erf-GELU arithmetic
itself is coarse. The 27 transformer-block GELUs use ``tanh`` and are already
within 5e-6, so they must NOT be touched and must NOT be unified with ``erf``.

This kernel computes the original erf-based GELU with an explicit, pinned
single-precision rational approximation and one final Half cast:

    x (Half, GM) -> x (Float32 UB)
                 -> z = |x| / sqrt(2)
                 -> t = 1 / (1 + p*z)                 (vec_rec + 2 Newton)
                 -> poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
                 -> erf_abs = 1 - poly * exp(-z*z)    (Abramowitz-Stegun 7.1.26)
                 -> gelu = 0.5*x + 0.5*|x|*erf_abs
                 -> ONE Float32 -> Half cast -> GM

WHY |x| INSTEAD OF A SIGN SELECT
--------------------------------
``erf`` is odd, and ``x * sign(x) == |x|``, so with ``erf_abs(z) = erf(|z|)``
the original Gelu formula collapses to the sign-free identity

    gelu(x) = 0.5*x + 0.5*|x|*erf_abs(|x|/sqrt2)

This needs no elementwise compare/select at all. ``vec_sel`` in the reviewed SDK
takes a per-element selection tensor that only a compare instruction could
produce, and no compare op is exposed, so a sign-select formulation could not be
expressed with verified primitives. The identity avoids that problem entirely and
has three useful properties, all re-checked by the stdlib tests here:

* ``x == 0`` gives exactly ``0`` (``t=1``, ``poly(1) == 1``, ``exp(0) == 1``);
* ``1 + p*|z| >= 1``, so the reciprocal can never divide by a vanishing
  denominator near ``x == 0``;
* the far negative tail is exact: for ``|x|`` large enough that ``exp(-z*z)``
  underflows to exactly 0, ``erf_abs == 1`` and ``0.5*x + 0.5*|x| == 0``.
  Writing the last term as ``(z*erf_abs)*(1/sqrt2)`` instead would round the same
  product twice and leave a spurious +2e-3 at ``x == -65504``; the explicit
  ``|x|`` buffer is carried for exactly that reason.

``erf_abs`` is additionally passed through ``vec_relu`` so a hardware rounding
anomaly can never flip the sign of a negative-input output. The approximation's
true range is already ``[0, 1]``.

SIGNED ZERO AND NON-CLAIMS
--------------------------
The arithmetic here is the sign-free form, so ``x == -0.0`` yields ``+0.0`` where
the CPU yields ``-0.0``: numerically equal, bit-different, recorded rather than
claimed away. No ``tanh``/sigmoid substitution (original ``erf`` semantics are
kept), no model weight of any kind, no accuracy acceptance and no speed claim.
The parent compares the actual output with the unchanged CPU FP16 gold; a leaf
that matches does not make the whole encoder pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

# --- pinned hardware contract (same part as the executed R83/R87 kernels) -----
PINNED_COMPILE_SOC = "Ascend910B"
PINNED_RUNTIME_SOC = "Ascend910B"
PINNED_CORE_NUM = 30
PINNED_L1_SIZE = 1048576
PINNED_L0C_SIZE = 262144
PINNED_UB_SAFE_BYTES = 196608
MAX_DMA_BURST_BLOCKS = 128          # 4096 B, the proven R83 burst ceiling

# --- pinned immutable collaborators (checked from disk, never edited here) ----
REFERENCE_RUNTIME_MODULE_SHA256 = (
    "c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6")
REFERENCE_RUNTIME_LIBRARY_SHA256 = (
    "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043")
REFERENCE_CANDIDATE_SHA256 = (
    "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78")
REFERENCE_RUNTIME_MODULE = "runtime5gm83"

# --- Giant Eigen/erf constants (Abramowitz & Stegun 7.1.26) ------------------
#: erf(z) = 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) e^(-z*z),
#: t = 1/(1 + p*z), for z >= 0. Maximum absolute error 1.5e-7 in exact math.
ERF_A1 = 0.254829592
ERF_A2 = -0.284496736
ERF_A3 = 1.421413741
ERF_A4 = -1.453152027
ERF_A5 = 1.061405429
ERF_P = 0.3275911
NEWTON_STEPS = 2

HALF_SQRT2 = 0.7071067811865476      # 1/sqrt(2) == sqrt(2)/2
HALF = 0.5

# --- geometry ---------------------------------------------------------------
#: One repeat of an FP32 vector op covers 64 elements = 256 B = 8 blocks of 32 B.
FP32_MASK = 64
FP32_REP_STRIDE = 8
#: One repeat of an FP16 vector op covers 128 elements = 256 B.
FP16_MASK = 128
FP16_REP_STRIDE = 8

TILE_ELEMS = 4096                    # divisible by 64; tile is one 32 B-aligned run
#: Real frozen merger activation from Round89: 2016 rows x 4608 columns.
REAL_ROWS = 2016
REAL_HIDDEN = 4608
REAL_NUMEL = REAL_ROWS * REAL_HIDDEN
SMOKE_NUMEL = 8192

#: The four original erf-mode GELUs. The 27 block MLPs use tanh and are excluded.
MERGER_GELU_NAMES = (
    "merger.act_fn",
    "deepstack_merger_list.0.act_fn",
    "deepstack_merger_list.1.act_fn",
    "deepstack_merger_list.2.act_fn",
)
EXPECTED_TOTAL_GELU_MODULES = 31     # 27 tanh (MLP) + 4 erf (merger)
EXPECTED_TANH_GELU_MODULES = 27
EXPECTED_ERF_GELU_MODULES = 4

#: Buffers, in the order they appear in the UB plan. Two are Half, the rest Float32.
HALF_BUFFERS = ("x_h", "c_h")
FLOAT_BUFFERS = ("x_f", "a_f", "z_f", "d_f", "t_f", "p_f", "e_f", "k_f")

#: TIK tensor names become C identifiers in the generated kernel. Round58's
#: compile failure was caused by a tensor literally named ``half`` shadowing the
#: ``half`` type, so the reserved names below are refused by the plan.
RESERVED_TENSOR_NAMES = frozenset({
    "half", "float", "double", "int", "void", "bool", "char", "short", "long",
    "signed", "unsigned", "auto", "register", "static", "struct", "union",
    "enum", "typedef", "sizeof", "return", "if", "else", "for", "while", "do",
    "switch", "case", "break", "continue", "goto", "extern", "inline",
    "const", "volatile", "class", "template", "namespace", "new", "delete",
})

ALLOWED_OPS = frozenset({
    "data_move", "h_cast", "vec_abs", "vec_muls", "vec_adds", "vec_mul",
    "vec_add", "vec_rec", "vec_exp", "vec_relu",
})


class ErfGeluError(RuntimeError):
    """Raised instead of guessing, silently falling back or relaxing a contract."""


# ---------------------------------------------------------------------------
# Pure geometry / plan helpers (stdlib only: unit-testable without TIK)
# ---------------------------------------------------------------------------


def _check_positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ErfGeluError("%s must be a positive int, got %r" % (label, value))
    return value


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def plan_tiling(numel):
    """Describe the fixed multi-core plan for ``numel`` contiguous Half elements.

    Every tile is exactly :data:`TILE_ELEMS` elements, so ``numel`` must be a
    multiple of it. The task loop is a dynamic ``core + step*cores`` walk with a
    bounds test, exactly like the executed Round83 blocked kernel: the step count
    is fixed at compile time while the last round simply has fewer live cores.
    """
    numel = _check_positive_int(numel, "numel")
    if numel % TILE_ELEMS != 0:
        raise ErfGeluError(
            "numel %d is not a multiple of the fixed tile %d" % (numel, TILE_ELEMS))
    tasks = numel // TILE_ELEMS
    cores = min(PINNED_CORE_NUM, tasks)
    steps = _ceil_div(tasks, cores)

    half_tile_bytes = TILE_ELEMS * 2
    float_tile_bytes = TILE_ELEMS * 4
    if half_tile_bytes % 32 or float_tile_bytes % 32:
        raise ErfGeluError("tile does not map onto 32 B blocks")

    dma_blocks = half_tile_bytes // 32
    if dma_blocks % MAX_DMA_BURST_BLOCKS:
        raise ErfGeluError(
            "tile of %d Half elements is %d blocks, which does not split into %d-block "
            "bursts" % (TILE_ELEMS, dma_blocks, MAX_DMA_BURST_BLOCKS))
    burst = MAX_DMA_BURST_BLOCKS
    bursts = dma_blocks // MAX_DMA_BURST_BLOCKS

    repeats = TILE_ELEMS // FP32_MASK
    if repeats * FP32_MASK != TILE_ELEMS:
        raise ErfGeluError("tile is not a whole number of FP32 repeats")

    ub_bytes = (len(HALF_BUFFERS) * half_tile_bytes
                + len(FLOAT_BUFFERS) * float_tile_bytes)
    if ub_bytes > PINNED_UB_SAFE_BYTES:
        raise ErfGeluError(
            "UB plan needs %d B, above the %d B safe budget"
            % (ub_bytes, PINNED_UB_SAFE_BYTES))

    return {
        "numel": numel,
        "tile_elems": TILE_ELEMS,
        "tasks": tasks,
        "cores": cores,
        "steps": steps,
        "physical_cores": PINNED_CORE_NUM,
        "live_tiles_in_last_step": tasks - (steps - 1) * cores,
        "tiles": {"count": tasks, "elements": TILE_ELEMS},
        "grid": {"cores": cores, "steps": steps, "tasks": tasks,
                 "dynamic_task_index": "core + step*cores"},
        "dma": {"nburst": bursts, "burst": burst, "src_stride": 0, "dst_stride": 0,
                "gm_bytes_per_tile": half_tile_bytes,
                "burst_bytes": burst * 32},
        "vector": {"mask": FP32_MASK, "repeat_times": repeats,
                   "rep_stride": FP32_REP_STRIDE,
                   "elements_per_repeat": FP32_MASK,
                   "bytes_per_repeat": FP32_MASK * 4},
        "buffers": {"half": list(HALF_BUFFERS), "float": list(FLOAT_BUFFERS),
                    "half_tile_bytes": half_tile_bytes,
                    "float_tile_bytes": float_tile_bytes,
                    "ub_bytes": ub_bytes,
                    "ub_safe_bytes": PINNED_UB_SAFE_BYTES},
        "newton_steps": NEWTON_STEPS,
        "mixed_precision": "HalfGM->FloatUBmath->oneHalfGM",
        "no_early_half_math": True,
        "bf16_used": False,
        "a2_only_features_used": False,
        "core_dim_32_assumed": False,
    }


def emission_steps(plan=None):
    """The exact ordered instruction plan, as plain data.

    ``_emit_kernel`` walks this very table, so what the stdlib tests assert is
    literally what gets emitted. ``data_move`` steps are filled in per tile at
    emission time because they depend on the tile offset.
    """
    if plan is None:
        plan = plan_tiling(REAL_NUMEL)
    repeats = plan["vector"]["repeat_times"]
    mask = plan["vector"]["mask"]
    stride = plan["vector"]["rep_stride"]

    def vec(op, dst, src, scalar=None, src0=None, src1=None):
        step = {"op": op, "dst": dst, "repeat_times": repeats, "mask": mask,
                "dst_rep_stride": stride}
        if op in ("vec_abs", "vec_rec", "vec_exp", "vec_relu"):
            if src is None:
                raise ErfGeluError("unary op %r needs a source" % (op,))
            step["src"] = src
            step["src_rep_stride"] = stride
        elif op in ("vec_muls", "vec_adds"):
            if src is None or scalar is None:
                raise ErfGeluError("scalar op %r needs both a source and a scalar" % (op,))
            step["src"] = src
            step["scalar"] = scalar
            step["src_rep_stride"] = stride
        elif op in ("vec_mul", "vec_add"):
            if src0 is None or src1 is None:
                raise ErfGeluError("binary op %r needs both operands" % (op,))
            step["src0"] = src0
            step["src1"] = src1
            step["src0_rep_stride"] = stride
            step["src1_rep_stride"] = stride
        else:
            raise ErfGeluError("unknown vector op %r" % (op,))
        return step

    steps = [
        {"op": "data_move", "direction": "gm_to_ub", "dst": "x_h", "buffer": "x_gm"},
        {"op": "h_cast", "dst": "x_f", "src": "x_h", "round_mode": "none",
         "note": "Half -> Float32, the only widening"},
        # |x| and z = |x|/sqrt2
        vec("vec_abs", "a_f", "x_f"),
        vec("vec_muls", "z_f", "a_f", scalar=HALF_SQRT2),
        # d = 1 + p*z, strictly >= 1, then t = 1/d
        vec("vec_muls", "d_f", "z_f", scalar=ERF_P),
        vec("vec_adds", "d_f", "d_f", scalar=1.0),
        vec("vec_rec", "t_f", "d_f"),
    ]
    # Newton refinement of the reciprocal: t = t*(2 - d*t)
    for _ in range(NEWTON_STEPS):
        steps.extend([
            vec("vec_mul", "p_f", None, src0="d_f", src1="t_f"),
            vec("vec_muls", "p_f", "p_f", scalar=-1.0),
            vec("vec_adds", "p_f", "p_f", scalar=2.0),
            vec("vec_mul", "t_f", None, src0="t_f", src1="p_f"),
        ])
    # Horner on t, then the leading t factor of A&S 7.1.26.
    steps.extend([
        vec("vec_muls", "p_f", "t_f", scalar=ERF_A5),
        vec("vec_adds", "p_f", "p_f", scalar=ERF_A4),
        vec("vec_mul", "p_f", None, src0="p_f", src1="t_f"),
        vec("vec_adds", "p_f", "p_f", scalar=ERF_A3),
        vec("vec_mul", "p_f", None, src0="p_f", src1="t_f"),
        vec("vec_adds", "p_f", "p_f", scalar=ERF_A2),
        vec("vec_mul", "p_f", None, src0="p_f", src1="t_f"),
        vec("vec_adds", "p_f", "p_f", scalar=ERF_A1),
        vec("vec_mul", "p_f", None, src0="p_f", src1="t_f"),
    ])
    # exp(-z*z) and erf_abs = 1 - poly*exp(-z*z)
    steps.extend([
        vec("vec_mul", "k_f", None, src0="z_f", src1="z_f"),
        vec("vec_muls", "k_f", "k_f", scalar=-1.0),
        vec("vec_exp", "e_f", "k_f"),
        vec("vec_mul", "k_f", None, src0="p_f", src1="e_f"),
        vec("vec_muls", "k_f", "k_f", scalar=-1.0),
        vec("vec_adds", "k_f", "k_f", scalar=1.0),
        # Original910AhasnoFloatvec_relu;exactpositivepartviaFloatabs/add/muls.
        vec("vec_abs", "e_f", "k_f"),
        vec("vec_add", "k_f", None, src0="k_f", src1="e_f"),
        vec("vec_muls", "k_f", "k_f", scalar=HALF),
    ])
    # gelu = 0.5*x + 0.5*|x|*erf_abs, with the explicit |x| buffer so the two
    # tails round symmetrically (see the module docstring).
    steps.extend([
        vec("vec_muls", "k_f", "k_f", scalar=HALF),
        vec("vec_mul", "k_f", None, src0="a_f", src1="k_f"),
        vec("vec_muls", "x_f", "x_f", scalar=HALF),
        vec("vec_add", "x_f", None, src0="x_f", src1="k_f"),
        {"op": "h_cast", "dst": "c_h", "src": "x_f", "round_mode": "none",
         "note": "the ONE Float32 -> Half narrowing"},
        {"op": "data_move", "direction": "ub_to_gm", "src": "c_h", "buffer": "c_gm"},
    ])
    for step in steps:
        if step["op"] not in ALLOWED_OPS:
            raise ErfGeluError("emission uses an unreviewed op %r" % (step["op"],))
        for role in ("dst", "src", "src0", "src1"):
            name = step.get(role)
            if name is not None and name in RESERVED_TENSOR_NAMES:
                raise ErfGeluError(
                    "tensor name %r collides with a generated C type/identifier" % (name,))
    return tuple(steps)


def narrowing_steps(plan=None):
    """The Float32 -> Half casts; the plan must contain exactly one of them."""
    return tuple(step for step in emission_steps(plan)
                 if step["op"] == "h_cast" and step["dst"] in HALF_BUFFERS
                 and step["dst"] != "x_h")


# ---------------------------------------------------------------------------
# Float32-faithful scalar emulation (stdlib only; NOT the hardware)
# ---------------------------------------------------------------------------


def _f32(value):
    """Round a Python float to IEEE-754 binary32, which is what the UB math uses."""
    import struct

    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f16(value):
    """Round a Python float to IEEE-754 binary16, the kernel's GM dtype."""
    import struct

    return struct.unpack("<e", struct.pack("<e", value))[0]


def _erf_abs(value, rec_initial_relative_error=0.0, newton_steps=NEWTON_STEPS):
    """The kernel's ``erf(|x|/sqrt2)`` for one scalar, in IEEE binary32 steps."""
    x = _f32(value)
    abs_x = _f32(abs(x))
    z = _f32(abs_x * HALF_SQRT2)
    d = _f32(_f32(z * ERF_P) + 1.0)
    t = _f32((1.0 / d) * (1.0 + rec_initial_relative_error))
    for _ in range(newton_steps):
        p = _f32(d * t)
        p = _f32(-p)
        p = _f32(p + 2.0)
        t = _f32(t * p)
    # Horner on t up to a1, then the leading t factor of A&S 7.1.26.
    poly = _f32(_f32(t * ERF_A5))
    poly = _f32(poly + ERF_A4)
    for coefficient in (ERF_A3, ERF_A2, ERF_A1):
        poly = _f32(_f32(poly * t) + coefficient)
    poly = _f32(poly * t)
    exp_term = _f32(math.exp(_f32(-_f32(z * z))))
    erf_abs = _f32(_f32(-_f32(poly * exp_term)) + 1.0)
    if erf_abs < 0.0:
        erf_abs = 0.0
    return abs_x, erf_abs


def scalar_emulation(value, rec_initial_relative_error=0.0, newton_steps=NEWTON_STEPS):
    """Emulate one element of this kernel's op sequence in IEEE binary32.

    ``rec_initial_relative_error`` degrades ``vec_rec``'s first estimate on
    purpose, so the tests can show that the Newton refinement, not an assumed
    perfect reciprocal, is what protects the arithmetic. ``exp`` comes from the
    platform libm in double and is rounded to binary32, so this bounds the
    *formula*; it cannot model the hardware vector unit's own ``exp`` ULP
    behaviour, and no accuracy acceptance is derived from it.
    """
    abs_x, erf_abs = _erf_abs(value, rec_initial_relative_error, newton_steps)
    x = _f32(value)
    tail = _f32(abs_x * _f32(erf_abs * HALF))
    return _f32(_f32(x * HALF) + tail)


def scalar_erf_abs(value):
    """The kernel's intermediate ``erf(|x|/sqrt2)``, exposed for the range tests."""
    return _erf_abs(value)[1]


def scalar_reference(value):
    """Double-precision erf-GELU: the mathematical target of this kernel."""
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def scalar_half_reference(value):
    """What a Float32-exact erf-GELU would emit in Half (reference control)."""
    x = _f32(value)
    exact = 0.5 * x * (1.0 + math.erf(_f32(x * _f32(1.0 / math.sqrt(2.0)))))
    return _f16(exact)


def scalar_erf_abs(value):
    """The kernel's intermediate ``erf(|x|/sqrt2)``, exposed for the range tests."""
    x = _f32(value)
    abs_x = _f32(abs(x))
    z = _f32(abs_x * HALF_SQRT2)
    d = _f32(_f32(z * ERF_P) + 1.0)
    t = _f32(1.0 / d)
    for _ in range(NEWTON_STEPS):
        p = _f32(d * t)
        p = _f32(-p)
        p = _f32(p + 2.0)
        t = _f32(t * p)
    poly = _f32(_f32(t * ERF_A5))
    poly = _f32(poly + ERF_A4)
    for coefficient in (ERF_A3, ERF_A2, ERF_A1):
        poly = _f32(_f32(poly * t) + coefficient)
    poly = _f32(poly * t)
    exp_term = _f32(math.exp(_f32(-_f32(z * z))))
    erf_abs = _f32(_f32(-_f32(poly * exp_term)) + 1.0)
    return 0.0 if erf_abs < 0.0 else erf_abs


# ---------------------------------------------------------------------------
# Environment guards and source pins (lazy: importing needs no CANN)
# ---------------------------------------------------------------------------


def capability_report():
    """Query the facts this kernel depends on, without building anything."""
    import tbe.common.platform as platform

    platform.set_current_compile_soc_info(PINNED_COMPILE_SOC)
    resolved = platform.get_soc_spec("FULL_SOC_VERSION")
    if resolved != PINNED_COMPILE_SOC:
        raise ErfGeluError(
            "compile SoC resolved to %r, expected %r; refusing to emit another ISA"
            % (resolved, PINNED_COMPILE_SOC))
    report = {
        "compile_soc": resolved,
        "runtime_expected_soc": PINNED_RUNTIME_SOC,
        "runtime_soc_label_note": "this original 910A-generation part reports the short "
                                  "SoC name Ascend910 at runtime and has 30 usable cores",
        "core_num": int(platform.get_soc_spec("CORE_NUM")),
        "l1_size": int(platform.get_soc_spec("L1_SIZE")),
        "l0c_size": int(platform.get_soc_spec("L0C_SIZE")),
    }
    try:
        report["ub_size"] = int(platform.get_soc_spec("UB_SIZE"))
    except Exception:  # noqa: BLE001 - older SDKs may not expose UB_SIZE
        report["ub_size"] = None
    for label, dtype_str in (("float16_to_float32", "f162f32"),
                             ("float32_to_float16", "f322f16")):
        try:
            supported = bool(platform.intrinsic_check_support("Intrinsic_vconv", dtype_str))
        except Exception:  # noqa: BLE001 - record the unknown, never assume support
            supported = None
        report["Intrinsic_vconv_" + label] = supported
    if report["core_num"] != PINNED_CORE_NUM:
        raise ErfGeluError(
            "compiler CORE_NUM is %d, expected %d; 32 GiB of HBM per chip is not 32 "
            "cube cores" % (report["core_num"], PINNED_CORE_NUM))
    if report["l1_size"] != PINNED_L1_SIZE or report["l0c_size"] != PINNED_L0C_SIZE:
        raise ErfGeluError("L1/L0C sizes do not match the pinned part")
    if report["ub_size"] is not None and report["ub_size"] < PINNED_UB_SAFE_BYTES:
        raise ErfGeluError(
            "UB size %d is below the %d B plan" % (report["ub_size"], PINNED_UB_SAFE_BYTES))
    for label in ("float16_to_float32", "float32_to_float16"):
        if report["Intrinsic_vconv_" + label] is False:
            raise ErfGeluError(
                "Intrinsic_vconv/%s is unsupported; the Half<->Float32 operand path this "
                "kernel needs cannot be faked" % label)
    return report


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reference_sources(strict=True):
    """Fail closed if a pinned collaborator on disk was modified."""
    import sys

    status = {"runtime_module": REFERENCE_RUNTIME_MODULE,
              "runtime_module_expected_sha256": REFERENCE_RUNTIME_MODULE_SHA256,
              "runtime_library_expected_sha256": REFERENCE_RUNTIME_LIBRARY_SHA256,
              "candidate_expected_sha256": REFERENCE_CANDIDATE_SHA256,
              "no_reference_source_edited_by_this_round": True}
    here = Path(__file__).resolve().parent
    candidates = [here / (REFERENCE_RUNTIME_MODULE + ".py")]
    module = sys.modules.get(REFERENCE_RUNTIME_MODULE)
    if module is not None and getattr(module, "__file__", None):
        candidates.insert(0, Path(module.__file__))
    found = None
    for candidate in candidates:
        if candidate.is_file():
            found = candidate
            break
    if found is None:
        status["runtime_module_sha256"] = None
        status["runtime_module_missing"] = True
        if strict:
            raise ErfGeluError(
                "cannot locate %s next to this module to verify its pin"
                % REFERENCE_RUNTIME_MODULE)
        return status
    actual = _sha256_file(found)
    status.update({"runtime_module_path": str(found), "runtime_module_sha256": actual})
    if actual != REFERENCE_RUNTIME_MODULE_SHA256:
        raise ErfGeluError(
            "%s changed: %s != %s; the launcher bridge this kernel targets is no longer "
            "the reviewed one" % (REFERENCE_RUNTIME_MODULE, actual,
                                  REFERENCE_RUNTIME_MODULE_SHA256))
    return status


# ---------------------------------------------------------------------------
# TIK emission
# ---------------------------------------------------------------------------


def _emit_kernel(t, plan, tik):
    """Declare the five GM operands and emit the reviewed vector plan."""
    numel = plan["numel"]
    cores, steps, tasks = plan["grid"]["cores"], plan["grid"]["steps"], plan["grid"]["tasks"]
    dma = plan["dma"]

    x_gm = t.Tensor("float16", (numel,), name="x_gm", scope=tik.scope_gm)
    aux1_gm = t.Tensor("float16", (16,), name="aux1_gm", scope=tik.scope_gm)
    aux2_gm = t.Tensor("float16", (16,), name="aux2_gm", scope=tik.scope_gm)
    aux3_gm = t.Tensor("float16", (16,), name="aux3_gm", scope=tik.scope_gm)
    c_gm = t.Tensor("float16", (numel,), name="c_gm", scope=tik.scope_gm)

    steps_plan = emission_steps(plan)
    tile = plan["tile_elems"]
    repeats = plan["vector"]["repeat_times"]
    half_tile_bytes = plan["buffers"]["half_tile_bytes"]
    float_tile_bytes = plan["buffers"]["float_tile_bytes"]

    with t.for_range(0, cores, block_num=cores) as core:
        ub = {
            "x_h": t.Tensor("float16", (tile,), name="x_h", scope=tik.scope_ubuf),
            "c_h": t.Tensor("float16", (tile,), name="c_h", scope=tik.scope_ubuf),
            "x_f": t.Tensor("float32", (tile,), name="x_f", scope=tik.scope_ubuf),
            "a_f": t.Tensor("float32", (tile,), name="a_f", scope=tik.scope_ubuf),
            "z_f": t.Tensor("float32", (tile,), name="z_f", scope=tik.scope_ubuf),
            "d_f": t.Tensor("float32", (tile,), name="d_f", scope=tik.scope_ubuf),
            "t_f": t.Tensor("float32", (tile,), name="t_f", scope=tik.scope_ubuf),
            "p_f": t.Tensor("float32", (tile,), name="p_f", scope=tik.scope_ubuf),
            "e_f": t.Tensor("float32", (tile,), name="e_f", scope=tik.scope_ubuf),
            "k_f": t.Tensor("float32", (tile,), name="k_f", scope=tik.scope_ubuf),
        }
        with t.for_range(0, steps) as step:
            task = core + step * cores
            with t.if_scope(task < tasks):
                base = task * tile
                for entry in steps_plan:
                    op = entry["op"]
                    if op == "data_move":
                        if entry["direction"] == "gm_to_ub":
                            t.data_move(ub["x_h"], x_gm[base], 0, dma["nburst"],
                                        dma["burst"], dma["src_stride"], dma["dst_stride"])
                        else:
                            t.data_move(c_gm[base], ub["c_h"], 0, dma["nburst"],
                                        dma["burst"], dma["src_stride"], dma["dst_stride"])
                    elif op == "h_cast":
                        t.h_cast(ub[entry["dst"]], ub[entry["src"]], entry["round_mode"])
                    elif op in ("vec_abs", "vec_rec", "vec_exp", "vec_relu"):
                        getattr(t, op)(entry["mask"], ub[entry["dst"]], ub[entry["src"]],
                                       entry["repeat_times"], entry["dst_rep_stride"],
                                       entry["src_rep_stride"])
                    elif op in ("vec_muls", "vec_adds"):
                        getattr(t, op)(entry["mask"], ub[entry["dst"]], ub[entry["src"]],
                                       entry["scalar"], entry["repeat_times"],
                                       entry["dst_rep_stride"], entry["src_rep_stride"])
                    elif op in ("vec_mul", "vec_add"):
                        getattr(t, op)(entry["mask"], ub[entry["dst"]], ub[entry["src0"]],
                                       ub[entry["src1"]], entry["repeat_times"],
                                       entry["dst_rep_stride"], entry["src0_rep_stride"],
                                       entry["src1_rep_stride"])
                    else:  # pragma: no cover - emission_steps already refuses these
                        raise ErfGeluError("unemitted op %r" % (op,))
                # The plan is element-count driven, so the declared buffer sizes are
                # part of the contract rather than an incidental assumption.
                assert half_tile_bytes == tile * 2 and float_tile_bytes == tile * 4
                assert repeats == tile // FP32_MASK
    return {"x": x_gm, "aux1": aux1_gm, "aux2": aux2_gm, "aux3": aux3_gm, "c": c_gm}


# ---------------------------------------------------------------------------
# Build (cached, locked, never overwriting a previous output directory)
# ---------------------------------------------------------------------------


def kernel_name_for(plan):
    return "erf_gelu_fixed90_n%d_t%d_c%d" % (plan["numel"], plan["tile_elems"], plan["grid"]["cores"])


def build_kernel(numel, out_dir=None):
    """Build (or reuse) the erf-GELU kernel for ``numel`` Half elements."""
    plan = plan_tiling(numel)
    from tbe import tik

    capability = capability_report()
    sources = check_reference_sources(strict=True)
    name = kernel_name_for(plan)
    out_root = Path(out_dir).resolve() if out_dir is not None else Path.cwd() / "erf_kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec90.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                                sources, compiled_now=False)

    with (out_root / ".erf90.compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                                    sources, compiled_now=False)
        work = Path(tempfile.mkdtemp(prefix="erf_gelu90_build_", dir=out_root)).resolve()
        if work.parent != out_root or not work.name.startswith("erf_gelu90_build_"):
            raise ErfGeluError("unexpected build workspace")
        t = tik.Tik(disable_debug=True)
        gm = _emit_kernel(t, plan, tik)
        previous = os.getcwd()
        try:
            os.chdir(work)
            # All five GM operands are passed explicitly: the three unused Half
            # auxiliaries exist only to keep the proven five-pointer launcher ABI,
            # and automatic collection would prune them.
            t.BuildCCE(kernel_name=name,
                       inputs=[gm["x"], gm["aux1"], gm["aux2"], gm["aux3"]],
                       outputs=[gm["c"]])
        finally:
            os.chdir(previous)
        built_obj = work / "kernel_meta" / (name + ".o")
        built_meta = work / "kernel_meta" / (name + ".json")
        if not built_obj.is_file() or not built_meta.is_file():
            # Keep the exact workspace for diagnosis: never delete or move paths.
            raise ErfGeluError(
                "BuildCCE produced no kernel_meta artefacts for %s; the build workspace %s "
                "is retained for diagnosis" % (name, work))
        if dest.exists():
            raise ErfGeluError("refusing to overwrite existing kernel directory %s" % dest)
        dest.mkdir(parents=True)
        import shutil

        shutil.copyfile(built_obj, obj)
        shutil.copyfile(built_meta, meta_path)
    return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                            sources, compiled_now=True)


def _write_spec_once(spec_path, spec):
    """Write the informational spec only when absent, never repairing it blindly."""
    if spec_path.is_file():
        try:
            existing = json.loads(spec_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            existing = None
        if isinstance(existing, dict) and \
                existing.get("obj_sha256") == spec["obj_sha256"] and \
                existing.get("kernel_name") == spec["kernel_name"]:
            return False
        # Only repair when both the resolved destination and the resolved temporary
        # path are the exact absolute paths inside the directory that holds this
        # verified object; otherwise leave the informational file untouched.
        try:
            intended = Path(spec["obj_path"]).resolve(strict=True).parent / "spec90.json"
        except OSError:
            return False
        resolved = spec_path.resolve(strict=False)
        if resolved != intended:
            return False
        tmp = resolved.with_suffix(".json.tmp90")
        if tmp.resolve(strict=False).parent != intended.parent:
            return False
        tmp.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, resolved)
        return True
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return True


def _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability, sources,
                     compiled_now=False):
    meta = json.loads(meta_path.read_text())
    block_dim = int(meta.get("blockDim", -1))
    if block_dim != plan["grid"]["cores"]:
        raise ErfGeluError(
            "cached kernel %s has blockDim %d but the plan needs %d; refusing to reuse a "
            "mismatched binary" % (name, block_dim, plan["grid"]["cores"]))
    result = {
        "kernel_name": name,
        "kernel_symbol": meta["kernelName"],
        "block_dim": block_dim,
        "obj_path": str(obj),
        "obj_sha256": _sha256_file(obj),
        "meta_path": str(meta_path),
        "dst_dtype": "float16",
        "input_dtype": "float16",
        "GM_argument_count": 5,
        "gm_shapes": {"x": [plan["numel"]], "aux1": [16], "aux2": [16], "aux3": [16],
                      "c": [plan["numel"]]},
        "numel": plan["numel"],
        "accumulator_or_math_dtype": "float32",
        "single_final_half_cast": True,
        "float_math_on_widened_operand": True,
        "erf_approximation": "Abramowitz-Stegun 7.1.26 rational, 2 Newton steps",
        "erf_polynomial_max_abs_error_exact_math": 1.5e-7,
        "newton_steps": NEWTON_STEPS,
        "no_model_weights": True,
        "no_float_model_weights": True,
        "original_erf_semantics_kept": True,
        "tanh_approximation_not_used": True,
        "bf16_used": False,
        "a2_only_features_used": False,
        "gm_aux_roles": "three unused Half aux pointers keep the proven 5-GM launcher ABI",
        "plan": plan,
        "capability": capability,
        "reference_sources": sources,
        "cache_source_epoch": {"erf_gelu_fixed90_py_sha256": _sha256_file(Path(__file__)),
                               "runtime_module_sha256": sources.get("runtime_module_sha256")},
        "compiled_now": compiled_now,
        "built": True,
        "executed": False,
        "built_unix_epoch": int(meta.get("buildTime", 0)) or None,
        "numerical_accuracy_claim": None,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
    }
    spec = dict(result)
    spec.pop("compiled_now", None)
    _write_spec_once(spec_path, spec)
    return result


# ---------------------------------------------------------------------------
# Gated per-instance adapter for the four vision merger GELUs
# ---------------------------------------------------------------------------


class ErfGeluAdapter90:
    """Replace ONLY the four original ``GELU(approximate='none')`` merger ops.

    The instance ``forward`` of each merger activation is patched with a bound
    method (``types.MethodType``) so nothing global changes: ``torch.nn.GELU``,
    ``torch.nn.functional``, every biased linear and the 27 ``tanh`` block GELUs
    are untouched. With the switch off the captured original bound method is
    called, which is exactly the native path.
    """

    def __init__(self, vision, runtime, kernel_dir):
        import torch
        from types import MethodType

        self.vision = vision
        self.runtime = runtime
        self.kernel_dir = Path(kernel_dir)
        self.metas = {}
        self.enabled = False
        self.calls = 0
        self.kernel_launches = 0
        self.tile_tasks_processed = 0
        self._aux = None
        self._torch = torch
        self._MethodType = MethodType
        modules = dict(vision.named_modules())
        self._modules = {}
        self._original = {}
        self._had_instance_forward = {}
        self._wrappers = {}
        for name in MERGER_GELU_NAMES:
            module = modules.get(name)
            if module is None:
                raise ErfGeluError("vision model has no module %r" % (name,))
            if not isinstance(module, torch.nn.GELU):
                raise ErfGeluError(
                    "%s is %s, not an nn.GELU; the erf activation moved"
                    % (name, type(module).__name__))
            approximate = str(getattr(module, "approximate", "none"))
            if approximate != "none":
                raise ErfGeluError(
                    "%s has approximate=%r; only the original exact-erf GELU may be "
                    "replaced here" % (name, approximate))
            self._modules[name] = module
            self._original[name] = getattr(module, "forward")
            self._had_instance_forward[name] = "forward" in module.__dict__
        self._check_vision_inventory(vision, torch)
        for name, module in self._modules.items():
            wrapper = MethodType(self._make_forward(name), module)
            self._wrappers[name] = wrapper
            module.forward = wrapper
        self._installed = True

    @staticmethod
    def _check_vision_inventory(vision, torch):
        """Refuse to proceed unless the 27 tanh + 4 erf GELU inventory is intact."""
        gelus = [m for m in vision.modules() if isinstance(m, torch.nn.GELU)]
        tanh = [m for m in gelus if str(getattr(m, "approximate", "none")) == "tanh"]
        erf = [m for m in gelus if str(getattr(m, "approximate", "none")) == "none"]
        if (len(gelus), len(tanh), len(erf)) != (
                EXPECTED_TOTAL_GELU_MODULES, EXPECTED_TANH_GELU_MODULES,
                EXPECTED_ERF_GELU_MODULES):
            raise ErfGeluError(
                "unexpected GELU inventory: %d total / %d tanh / %d erf, expected "
                "%d / %d / %d" % (len(gelus), len(tanh), len(erf),
                                  EXPECTED_TOTAL_GELU_MODULES, EXPECTED_TANH_GELU_MODULES,
                                  EXPECTED_ERF_GELU_MODULES))

    # -- forwarding -------------------------------------------------------
    def _make_forward(self, name):
        original = self._original[name]

        def forward(module, input_):
            if not self.enabled:
                return original(input_)
            return self._forward(name, module, input_)

        return forward

    def _forward(self, name, module, input_):
        torch = self._torch
        self._check_operand(name, input_, torch)
        if not self.metas:
            raise ErfGeluError(
                "accurate erf GELU used before prepare(); the kernel is not built")
        numel = int(self.metas["numel"])
        if input_.numel() != numel:
            raise ErfGeluError(
                "leaf %s has %d elements, this kernel was prepared for the fixed real "
                "fixture of %d" % (name, input_.numel(), numel))
        flat = input_.reshape(-1).contiguous()
        if str(flat.dtype) != "torch.float16":
            raise ErfGeluError("flattened operand is %s, expected Half" % flat.dtype)
        out = torch.empty((numel,), dtype=torch.float16, device=flat.device)
        self.runtime.launch(self.metas, [flat, self._aux[0], self._aux[1], self._aux[2], out],
                            synchronize=True)
        self.calls += 1
        self.kernel_launches += 1
        self.tile_tasks_processed += int(self.metas["plan"]["tiles"]["count"])
        return out.reshape(input_.shape)

    def _check_operand(self, name, input_, torch):
        if not torch.is_tensor(input_):
            raise ErfGeluError("leaf %s got a non-tensor input" % (name,))
        if input_.dtype is not torch.float16:
            raise ErfGeluError(
                "leaf %s input is %s; this kernel is Half-in/Half-out only"
                % (name, input_.dtype))
        if input_.device.type not in ("npu", "privateuseone"):
            raise ErfGeluError(
                "leaf %s input is on %r, expected an NPU device"
                % (name, input_.device.type))
        if input_.dim() != 2 or int(input_.shape[1]) != REAL_HIDDEN:
            raise ErfGeluError(
                "leaf %s input is %s; expected (rows, %d) like the real frozen fixture"
                % (name, tuple(input_.shape), REAL_HIDDEN))

    # -- public API -------------------------------------------------------
    def prepare(self):
        """Build the fixed-fixture kernel and the unused Half auxiliaries."""
        if not self._installed:
            raise ErfGeluError("adapter was uninstalled; it cannot be prepared again")
        import erf_gelu_fixed90 as kernel

        self.metas = kernel.build_kernel(REAL_NUMEL, out_dir=self.kernel_dir)
        if self.metas["numel"] != REAL_NUMEL:
            raise ErfGeluError("prepared kernel does not cover the real fixture")
        device = self._torch.device("npu:%d" % int(getattr(self.runtime, "device", 0)))
        self._aux = [self._torch.zeros(16, dtype=self._torch.float16, device=device)
                     for _ in range(3)]
        return self.snapshot()

    def set_enabled(self, value):
        if not isinstance(value, bool):
            raise ErfGeluError("set_enabled() must be a bool")
        if value and not self.metas:
            raise ErfGeluError("set_enabled(True) before prepare() is refused")
        self.enabled = value
        return self.snapshot()

    def snapshot(self):
        return {
            "round": 90,
            "installed": bool(self._installed),
            "enabled": bool(self.enabled),
            "patched_module_names": sorted(self._modules),
            "patched_module_count": len(self._modules),
            "expected_erf_gelu_modules": EXPECTED_ERF_GELU_MODULES,
            "tanh_gelu_modules_left_untouched": EXPECTED_TANH_GELU_MODULES,
            "calls": self.calls,
            "kernel_launches": self.kernel_launches,
            "tile_tasks_processed": self.tile_tasks_processed,
            "kernel_prepared": bool(self.metas),
            "kernel_name": self.metas.get("kernel_name"),
            "kernel_object_SHA256": self.metas.get("obj_sha256"),
            "numel": self.metas.get("numel"),
            "ub_bytes": (self.metas.get("plan") or {}).get("buffers", {}).get("ub_bytes"),
            "per_instance_forward_only": True,
            "patches_torch_global": False,
            "patches_torch_functional": False,
            "patches_other_activations": False,
            "patches_biased_linears": False,
            "disabled_is_exact_native": True,
            "cached_float_weights": False,
            "no_model_parameters": True,
            "source_sha256": _sha256_file(Path(__file__)),
            "numerical_tolerance": None,
            "full_encoder_precision_accepted": False,
            "whole_goal_complete": False,
        }

    def uninstall(self):
        if self._installed:
            for name, wrapper in self._wrappers.items():
                module = self._modules[name]
                current = getattr(module, "forward", None)
                # Bound-method identity, not "is": attribute access rebinds.
                is_ours = (getattr(current, "__self__", None) is module
                           and getattr(current, "__func__", None) is wrapper.__func__)
                if not is_ours:
                    raise ErfGeluError(
                        "leaf %s forward was replaced by a third party after install; the "
                        "original was not restored blindly" % (name,))
                if self._had_instance_forward[name]:
                    module.forward = self._original[name]
                else:
                    delattr(module, "forward")
            self._installed = False
        self.enabled = False
        self._wrappers.clear()
        self._original.clear()
        self._had_instance_forward.clear()
        summary = self.snapshot()
        summary["uninstalled"] = True
        return summary


def install(vision, runtime, kernel_dir):
    """Factory: an installed adapter that is disabled until ``prepare()`` + enable."""
    return ErfGeluAdapter90(vision, runtime, kernel_dir)


def cpu_half_reference(value):
    """Control only: CPU Half ``F.gelu(approximate='none')`` on the same operand.

    Not a teacher: it is the same ``erf`` operator on the same Half values, used by
    the driver to compare the kernel against the unchanged CPU FP16 reference.
    """
    import torch
    import torch.nn.functional as F

    source = value.detach().to("cpu").contiguous()
    return F.gelu(source, approximate="none")