#!/usr/bin/env python3
"""Round99: one-launch streaming vision flash-attention TIK kernel (source only).

STATUS: SOURCE ONLY. Nothing here is compiled, launched, imported on an NPU, or
mutated in production by its author: no SSH, no CANN build, no torch, no model,
no git. Every number belongs to the root driver, which compiles this builder and
compares it against the frozen real Q/K/V + original CPU Half MATH reference. This
file claims no accuracy, no speed, no tolerance and no default-enable result.

WHAT IT REPLACES
----------------
The executed Round97 route for the vision SDPA leaf costs, per query chunk, three
kernel launches plus a GM round trip: QK matmul -> FP32 score tile to GM ->
separate softmax/pack kernel -> separate fused P@V kernel. This kernel keeps the
whole row block in UB and performs the complete online softmax inside ONE launch
per (head, 64-query-row) tile:

    Q(64 x 72) , K(64 x 72) block , V(64 x 72) block  ->  O(64 x 72) Half

GEOMETRY (real H3 vision attention, before the 2x2 merger)
----------------------------------------------------------
H=16 heads, M=N=4032 query/key rows (48x84 patch grid), original head dim 72
zero-padded to 80 = 5 blocks of 16. Query rows are consumed in MB=BC=64 tiles:
4032/64 = 63 exact query tiles and 63 exact key/value blocks, so 16*63 = 1008
head-tile tasks over 30 cores (34 dynamic steps) with a dynamic 63-iteration KV
loop. A small smoke geometry (H=2, M=N=128, d=72) is accepted and uses
block_dim=min(30, tasks) so the same source compiles cheaply.

The vision attention is UNMASKED (not causal) and keeps the original scale
``72 ** -0.5``: the padded width 80 is never the scale.

MATH (per tile, online softmax; no Half cumulative state)
---------------------------------------------------------
1. ``QK`` is one HalfxHalf->FP32 L0C matmul over the full padded head dim (80);
   the score tile stays FP32 in UB: no narrowing of the main scores.
2. A **Half** row maximum is computed only as a *stable shift estimate*: fold the
   four Half NZ key groups and ``h_reduce_max(axis=1)``. Half ``vcmax`` is a
   supported instruction on this part (Float ``vcmax`` is not), and the shift does
   not need to be exact because softmax is shift invariant in exact arithmetic.
3. The shift error is bounded by the host protocol (below), so ``exp(S - shift)``
   stays inside the Half range and the two-Half probability split is safe.
4. Every value accumulated across the 63 KV blocks is FP32: running shift ``M``,
   denominator ``L`` and output ``O``. ``M``/``L``/``O`` are initialised by
   explicit ``vector_dup``/per-repeat fills so no first-block ``if``/``else``
   vector path is needed (the 910A CANN SPR pass has crashed on that pattern).
5. ``P_hi = Half(exp)`` and ``P_lo = Half((exp - Float(P_hi)) * 1024)``; two FP32
   L0C P@V products, FP32 UB combine ``hi + lo/1024``, and the only narrowing of
   the accumulator is the single final output cast.
6. The true denominator is the FP32 sum of ``exp(S - shift)`` folded over the four
   key groups, never a Half sum.

ROW BROADCASTS ARE EXPLICIT (no implicit broadcasting)
-----------------------------------------------------
``h_broadcast`` does not exist in this SDK and the high-level ops are not relied
on to broadcast a row across an NZ tile. Instead a per-row value is expanded with
:func:`_broadcast_row`: for each of the 64 rows, one ``Scalar.set_as`` loads the
row value from UB and one ``vector_dup`` writes that 16-value row slot for all
``n1`` NZ groups (``repeat_times=n1``, ``dst_rep_stride=2*MB`` blocks, aligned
32 B). Every subsequent ``h_sub``/``h_mul``/``h_div`` therefore has *identical*
operand shapes. The scalar loop is 64 unrolled iterations per broadcast site; its
cost is NOT measured here and is not claimed.

HOST PROTOCOL (required, enforced by the caller, never clamped here)
-------------------------------------------------------------------
``max |scaled QK| <= 16384`` (a quarter of the Half maximum magnitude, so a Half
shift error of up to 8 keeps the maximum-score exponent inside ``[e^-8, e^8]``;
other entries may underflow to zero, as in stable softmax. The peak therefore stays
inside Half). The caller either measures the scaled QK maximum finitely or bounds
it with Cauchy-Schwarz: ``sqrt(72) * max|q| * max|k| <= 16384``. This module
exposes :func:`check_host_protocol`; it raises on violation and never clamps.

REMAINING SDK ASSUMPTIONS THE ROOT CANN SMOKE MUST VERIFY
---------------------------------------------------------
* ``h_reduce_max``/``h_reduce_sum`` with ``axis=1`` on a ``(MB,16)`` UB tile
  return a ``(MB,)`` row tensor (the same contract the executed flash-attention
  source in this repository uses for ``mij_ub``/``mn0_block_max``).
* ``Tensor.reshape((n1, MB, 16))`` is a metadata view of the same flat UB buffer
  (also used by the executed flash-attention source).
* masked ``vector_dup`` with ``repeat_times=n1``, ``blk_stride=1`` and
  ``rep_stride=2*MB=128`` blocks writes exactly 16 Float values into every
  ``(group, row)`` NZ slot; ``Scalar.set_as(<UB element>)`` is a legal load.
* ``h_reduce_sum`` compiles to Float ``vcadd``, ``h_max`` to Float ``vmax``, and
  the Half shift uses Half ``vcmax``.
* in-place same-shape ``h_sub``/``h_mul``/``h_add`` with ``dst == src0`` (the
  executed flash-attention source does this on its row vectors).

Non-claims: no FP32 linear weight, no teacher tensor, no BF16 gold cast, no cached
hidden/patch/conditioning value, no Half cumulative accumulator, no clamp, no
post-hoc tolerance.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

# --- real H3 vision attention geometry --------------------------------------
HEADS = 16
SEQ = 4032                 # 48 * 84 patch grid, attention runs before the merger
HEAD_DIM = 72              # original head width; NOT the padded 80
PAD_DIM = 80               # 72 zero-padded to 5 blocks of 16
MB = 64                    # query rows per tile
BC = 64                    # key/value rows per block
MAX_CORES = 30             # usable AI cores per original 910A-generation chip
PINNED_COMPILE_SOC = "Ascend910"

D1 = PAD_DIM // 16         # 5 head-dim NZ blocks
KB1 = BC // 16             # 4 key NZ blocks per KV block
F32_MASK = 64              # one Float32 repeat (64 values = 256 B)
F32_REP_STRIDE = 8         # 8 x 32 B = 256 B = 64 Float32 values (contiguous)
BROADCAST_MASK = 16        # one NZ row slot = 16 Float32 = 64 B
BROADCAST_BLK_STRIDE = 1
RESIDUAL_SCALE = 1024.0
LOW_DOWNSCALE = 1.0 / RESIDUAL_SCALE
NEG_MAX = -3.4028234663852886e+38
SHIFT_BOUND = 16384.0      # protocol: exp(S-shift) stays inside Half
SHIFT_HALF_ULP_AT_BOUND = 8.0   # Half ulp at 2**14 is 16, so half-ulp is 8
EXP_MAX_AT_BOUND = 2980.9579870417283   # e**8
PINNED_UB_SAFE_BYTES = 196608
SMOKE_GEOMETRY = (2, 128, 128, HEAD_DIM)

KERNEL_ARG_ORDER = ["q_gm_float16", "k_gm_float16", "v_gm_float16",
                    "aux_gm_float32", "out_gm_float16"]

ASSUMPTIONS = (
    "h_reduce_max/h_reduce_sum(axis=1) on a (MB,16) tile yields a (MB,) row tensor",
    "Tensor.reshape((n1,MB,16)) is a metadata view of the same flat UB buffer",
    "masked vector_dup(repeat_times=n1, blk_stride=1, rep_stride=2*MB) writes one 16-value NZ row slot per group",
    "Scalar.set_as(<UB element>) is a legal UB-to-scalar load",
    "h_reduce_sum on Float compiles to Float vcadd; h_max to Float vmax",
    "the Half shift estimate uses Half vcmax (Float vcmax is unsupported here)",
    "vector_dup(mask,dst,scalar,1,1,1) with an explicit repeat slice fills exactly one repeat",
    "in-place same-shape h_sub/h_mul/h_add/h_exp with dst==src0 is supported",
    "a Float UB buffer can be reused sequentially (score->exp, Half score->P_hi, "
    "row expansion->widened P_hi) without an explicit alias API",
)


class FlashVisionError(RuntimeError):
    """Refusal: geometry, budget, capability or build contract violated."""


# ---------------------------------------------------------------------------
# Planning (pure python; the stdlib tests import this without CANN/torch)
# ---------------------------------------------------------------------------


def scale_for(d: int = HEAD_DIM) -> float:
    """Original attention scale ``head_dim ** -0.5`` for the real head width."""
    return float(d) ** -0.5


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FlashVisionError(f"{name} must be a python int, got {type(value).__name__}")
    if value <= 0:
        raise FlashVisionError(f"{name} must be positive, got {value}")
    return value


def plan_kernel(batch: int = HEADS, m: int = SEQ, n: int = SEQ,
                d: int = HEAD_DIM, mb: int = MB, bc: int = BC,
                cores_max: int = MAX_CORES) -> dict:
    """Return every number the emitter, the host packing and the tests share."""
    for name, value in (("batch", batch), ("m", m), ("n", n), ("bc", bc),
                        ("cores_max", cores_max)):
        _positive_int(value, name)
    _positive_int(mb, "mb")
    if d > PAD_DIM:
        raise FlashVisionError(f"head dim {d} exceeds the {PAD_DIM}-wide padded layout")
    if d <= PAD_DIM - 16:
        # The last padded 16-block must contain real values; a head dim of 64 or
        # less would silently change the NZ block count.
        raise FlashVisionError(f"head dim {d} would leave a fully padded NZ block")
    if mb != MB or bc != BC:
        raise FlashVisionError(
            f"this kernel is specialised to mb=bc={MB}; got mb={mb}, bc={bc}")
    if m % mb:
        raise FlashVisionError(f"m={m} is not a multiple of the query tile mb={mb}")
    if n % bc:
        raise FlashVisionError(f"n={n} is not a multiple of the KV block bc={bc}")

    m_tiles, n_tiles = m // mb, n // bc
    tasks = batch * m_tiles
    cores = min(cores_max, tasks)
    steps = (tasks + cores - 1) // cores
    score_elems = (bc // 16) * mb * 16          # (BC/16, MB, 16) FP32 scores
    pv_elems = D1 * mb * 16                     # (D1, MB, 16) FP32 P@V tile
    head_stride = {"q": D1 * m * 16, "k": D1 * n * 16,
                   "v": (n // 16) * PAD_DIM * 16, "out": D1 * m * 16}
    gm_shapes = {
        "q": [batch, D1, m, 16],
        "k": [batch, D1, n, 16],
        "v": [batch, n // 16, PAD_DIM, 16],
        "aux": [16],
        "aux_dtype": "float32",
        "out": [batch, D1, m, 16],
    }
    dma = {
        # One 32 B unit is one 16-value Half NZ row. Q/K blocks are m/n rows with
        # the remaining rows skipped; V is a contiguous run of 4 key blocks x 80
        # dim rows for the current KV block.
        "q_load": {"nburst": D1, "burst": mb, "src_stride": m - mb, "dst_stride": 0},
        "k_load": {"nburst": D1, "burst": bc, "src_stride": n - bc, "dst_stride": 0},
        "v_load": {"nburst": bc // 16, "burst": PAD_DIM, "src_stride": 0, "dst_stride": 0},
        "p_to_l1": {"nburst": bc // 16, "burst": mb},   # 4 x 64 units, proven burst size
        "score_l0c_to_ub": {"block_mode": "m", "nburst": 1,
                            "burst_len_kb": score_elems * 4 // 1024},
        "pv_l0c_to_ub": {"block_mode": "m", "nburst": 1,
                         "burst_len_kb": pv_elems * 4 // 1024},
        "out_store": {"nburst": D1, "burst": mb, "src_stride": 0, "dst_stride": m - mb},
    }
    buffers = {
        "l1_q_bytes": D1 * mb * 16 * 2,
        "l1_k_bytes": D1 * bc * 16 * 2,
        "l1_v_bytes": (bc // 16) * PAD_DIM * 16 * 2,
        "l1_p_hi_bytes": (bc // 16) * mb * 16 * 2,
        "l1_p_lo_bytes": (bc // 16) * mb * 16 * 2,
        "l0c_score_bytes": score_elems * 4,
        "l0c_pv_bytes": pv_elems * 4,
        "ub_score_bytes": score_elems * 4,
        "ub_half_score_bytes": score_elems * 2,
        "ub_exp_bytes": score_elems * 4,
        "ub_prob_half_bytes": score_elems * 2,   # hi and lo are separate tensors
        "ub_prob_float_bytes": score_elems * 4,
        "ub_pv_bytes": pv_elems * 4,
        "ub_out_bytes": pv_elems * 2,
    }
    buffers["l1_total_bytes"] = sum(v for k, v in buffers.items() if k.startswith("l1_"))
    buffers["l0c_total_bytes"] = (buffers["l0c_score_bytes"]
                                  + 2 * buffers["l0c_pv_bytes"])
    # A single flat accounting of every UB allocation the emitter declares.
    ub_named = {
        # The score tile is reused as the exp tile (in-place h_exp, the pattern the
        # executed flash-attention source uses), the Half score tile is reused as
        # P_hi, and the row-expansion tile is reused as the widened P_hi. Those
        # three reuses are what keeps this under the safe UB bound.
        "score_float_also_exp": score_elems * 4,
        "fold_half_a": mb * 16 * 2,
        "fold_half_b": mb * 16 * 2,
        "shift_half_row": mb * 2,
        "fold_float_a": mb * 16 * 4,
        "fold_float_b": mb * 16 * 4,
        "prob_hi_half_also_half_score": score_elems * 2,
        "prob_hi_float_also_row_broadcast": score_elems * 4,
        "prob_lo_float": score_elems * 4,
        "prob_lo_half": score_elems * 2,
        "pv_hi_float": pv_elems * 4,
        "pv_lo_float": pv_elems * 4,
        "out_acc_float": pv_elems * 4,
        "out_half": pv_elems * 2,
        "row_prev": mb * 4,
        "row_new": mb * 4,
        "row_diff": mb * 4,
        "row_rescale": mb * 4,
        "row_zero": mb * 4,
        "row_sum": mb * 4,
        "row_denominator": mb * 4,
    }
    buffers["ub_named"] = ub_named
    buffers["ub_total_bytes"] = sum(ub_named.values())
    vector = {
        "f32_mask": F32_MASK, "f32_rep_stride": F32_REP_STRIDE,
        "group_repeats": (mb * 16) // F32_MASK,
        "broadcast_mask": BROADCAST_MASK,
        "broadcast_blk_stride": BROADCAST_BLK_STRIDE,
        "broadcast_rep_stride": 2 * mb,
        "score_elems": score_elems, "pv_elems": pv_elems,
        "score_groups": bc // 16, "pv_groups": D1,
        "row_elems": mb,
    }
    plan = {
        "round": 99,
        "scope": ("one-launch streaming vision flash attention: FP32 online softmax, "
                  "Half shift estimate, two-Half probability P@V, one final Half cast"),
        "logical": {"heads": batch, "m": m, "n": n, "d": d,
                    "padded_head_dim": PAD_DIM, "d1": D1, "kb1": KB1,
                    "mb": mb, "bc": bc, "scale": scale_for(d),
                    "unmasked": True, "causal": False},
        "grid": {"tasks": tasks, "cores": cores, "steps": steps,
                 "m_tiles": m_tiles, "n_tiles": n_tiles,
                 "kv_tiles": n_tiles, "dynamic_kv_loop": True},
        "head_stride_elems": head_stride,
        "gm_shapes": gm_shapes,
        "gm_elems": {name: batch * head_stride[name] for name in ("q", "k", "v", "out")},
        "gm_bytes": {name: batch * head_stride[name] * 2 for name in ("q", "k", "v", "out")},
        "dma": dma,
        "buffers": buffers,
        "vector": vector,
        "math": {
            "main_scores_dtype": "float32",
            "main_softmax_dtype": "float32",
            "shift_estimate_dtype": "float16",
            "shift_is_exact": False,
            "accumulator_dtype": "float32",
            "denominator_dtype": "float32",
            "final_output_dtype": "float16",
            "probability_split": ["hi = Half(exp)", "lo = Half((exp - Float(hi)) * 1024)"],
            "residual_scale": RESIDUAL_SCALE,
            "low_downscale": LOW_DOWNSCALE,
            "init": {"row_max": NEG_MAX, "denominator": 0.0, "output": 0.0},
            "row_broadcast": ("explicit per-row vector_dup expansion; no implicit "
                              "h_* broadcasting is relied on"),
            "buffer_reuse": ["score FP32 tile -> exp tile (in-place h_exp)",
                             "Half score tile -> P_hi",
                             "row-expansion FP32 tile -> widened P_hi",
                             "PV low product -> rescale row expansion",
                             "PV high product -> final quotient tile"],
            "order": ["qk_matmul_full_padded_dim", "score_fp32_to_ub",
                      "scale_by_head_dim_inv_sqrt", "half_row_shift_estimate",
                      "online_max_and_rescale", "broadcast_shift_explicitly",
                      "exp_fp32", "fold_true_float_denominator",
                      "half_hi_half_lo_split", "two_fp32_pv_l0c",
                      "fp32_combine_hi_plus_lo_over_1024", "rescale_and_accumulate_output",
                      "final_float_div_then_one_half_cast"],
            "no_half_cumulative_output": True,
            "no_half_denominator": True,
            "no_clamp": True,
            "no_mid_product_fp32_gm": True,
            "no_hidden_or_conditioning_cache": True,
            "no_implicit_broadcast": True,
        },
        "host_protocol": {
            "bound": SHIFT_BOUND,
            "formula": ("max|scaled QK| <= 16384, measured finitely by the caller or "
                        "bounded by sqrt(d) * max|q| * max|k|"),
            "half_shift_ulp_at_bound": SHIFT_HALF_ULP_AT_BOUND,
            "max_exp_at_bound": EXP_MAX_AT_BOUND,
            "clamped": False,
            "enforced_by": "check_host_protocol",
        },
        "assumptions": list(ASSUMPTIONS),
        "compile_soc": PINNED_COMPILE_SOC,
    }
    buffers["ub_total_bytes"] += 6656  # exact native shift-copy512B + Halfbroadcast2048B
    _check_budget(plan)
    return plan


def _check_budget(plan: dict) -> None:
    buf = plan["buffers"]
    vec = plan["vector"]
    if buf["ub_total_bytes"] > PINNED_UB_SAFE_BYTES:
        raise FlashVisionError(
            f"UB need {buf['ub_total_bytes']} B exceeds the pinned safe bound "
            f"{PINNED_UB_SAFE_BYTES} B")
    if not 0 < plan["grid"]["cores"] <= MAX_CORES:
        raise FlashVisionError("core count outside the launched range")
    if vec["group_repeats"] > 255:
        raise FlashVisionError("a group vector op would need more than 255 repeats")
    if vec["broadcast_rep_stride"] > 255:
        raise FlashVisionError("the broadcast repeat stride exceeds the 8-bit immediate")
    if vec["broadcast_mask"] * 4 % 32:
        raise FlashVisionError("a broadcast row slot is not 32 B aligned")


def check_host_protocol(max_abs_scaled_qk=None, q_absmax=None, k_absmax=None,
                        d: int = HEAD_DIM) -> dict:
    """Validate the documented shift bound; raise (never clamp) on violation."""
    import math

    if max_abs_scaled_qk is None and (q_absmax is None or k_absmax is None):
        raise FlashVisionError('A finite measured or complete Q/K bound is required; no unguarded Half shift')
    if (q_absmax is None) != (k_absmax is None):
        raise FlashVisionError('Both absolute Q/K maxima are required together')
    if d != HEAD_DIM:
        raise FlashVisionError('The current kernel uses the original head dimension 72')
    for value in (max_abs_scaled_qk, q_absmax, k_absmax):
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise FlashVisionError('Absolute input bounds must be finite and nonnegative')
    report = {"bound": SHIFT_BOUND, "clamped": False,
              "measured_max_abs_scaled_qk": max_abs_scaled_qk,
              "cauchy_bound": None, "ok": True}
    if q_absmax is not None and k_absmax is not None:
        cauchy = math.sqrt(float(d)) * float(q_absmax) * float(k_absmax)
        report["cauchy_bound"] = cauchy
        if not math.isfinite(cauchy) or cauchy > SHIFT_BOUND:
            report["ok"] = False
            raise FlashVisionError(
                f"Cauchy bound sqrt({d})*|q|max*|k|max = {cauchy} exceeds the "
                f"{SHIFT_BOUND} shift protocol; the caller must measure a tighter "
                "finite bound and use a supported input domain (no rescaling or clamp is applied)")
    if max_abs_scaled_qk is not None:
        value = float(max_abs_scaled_qk)
        if not math.isfinite(value) or value > SHIFT_BOUND:
            report["ok"] = False
            raise FlashVisionError(
                f"max|scaled QK| = {value} exceeds the {SHIFT_BOUND} shift protocol")
    return report


# ---------------------------------------------------------------------------
# Capability probe (lazy CANN import; fail closed, never fake an A2 instruction)
# ---------------------------------------------------------------------------

REQUIRED_INTRINSICS = (
    ("Intrinsic_vcmax", "float16"),     # Half shift reduce
    ("Intrinsic_vor", "uint16"),        # exact Half row-bit copies
    ("Intrinsic_vtranspose", "float16"),# native16x16Halftranspose
    ("Intrinsic_vmax", "float16"),      # explicit Half fold, avoids high-level slice step inference
    ("Intrinsic_vcadd", "float32"),     # true Float denominator
    ("Intrinsic_vmax", "float32"),      # online shift update
    ("Intrinsic_vadd", "float32"),      # output accumulation
    ("Intrinsic_vmul", "float32"),      # rescale
    ("Intrinsic_vexp", "float32"),      # softmax exponential
    ("Intrinsic_vdiv", "float32"),      # final O / L
)
OPTIONAL_INTRINSICS = (
    ("Intrinsic_vcmax", "float32"),     # reported unsupported: never used
    ("Intrinsic_vexp", "float16"),
)


def capability_report(plan: dict | None = None) -> dict:
    """Resolve the compile SoC and refuse when a required instruction is missing."""
    import tbe.common.platform as platform

    plan = plan or plan_kernel()
    platform.set_current_compile_soc_info(PINNED_COMPILE_SOC)
    report = {"compile_soc": None, "core_num": None, "ub_size": None,
              "l1_size": None, "l0c_size": None, "intrinsic_checks": {},
              "required_missing": [], "query_answered": False,
              "launched_usable_cores": MAX_CORES,
              "assumptions": list(ASSUMPTIONS)}
    try:
        report["compile_soc"] = platform.get_soc_spec("SHORT_SOC_VERSION")
    except Exception:                                    # noqa: BLE001 - older SDK
        report["compile_soc"] = None
    if report["compile_soc"] not in (None, PINNED_COMPILE_SOC):
        raise FlashVisionError(
            f"compile SoC resolved to {report['compile_soc']!r}, expected "
            f"{PINNED_COMPILE_SOC!r}; refusing to emit a different ISA")
    for key, field in (("CORE_NUM", "core_num"), ("UB_SIZE", "ub_size"),
                       ("L1_SIZE", "l1_size"), ("L0C_SIZE", "l0c_size")):
        try:
            report[field] = int(platform.get_soc_spec(key))
        except Exception:                                # noqa: BLE001
            report[field] = None
    if report["core_num"] is not None and report["core_num"] < MAX_CORES:
        raise FlashVisionError(
            f"compiler CORE_NUM={report['core_num']} is below the {MAX_CORES} "
            "launched cores")
    buf = plan["buffers"]
    if report["ub_size"] is not None and report["ub_size"] < buf["ub_total_bytes"]:
        raise FlashVisionError(
            f"UB_SIZE={report['ub_size']} B cannot hold {buf['ub_total_bytes']} B")
    if report["l1_size"] is not None and report["l1_size"] < buf["l1_total_bytes"]:
        raise FlashVisionError(
            f"L1_SIZE={report['l1_size']} B cannot hold {buf['l1_total_bytes']} B")
    if report["l0c_size"] is not None and report["l0c_size"] < buf["l0c_total_bytes"]:
        raise FlashVisionError(
            f"L0C_SIZE={report['l0c_size']} B cannot hold {buf['l0c_total_bytes']} B")
    answered = False
    for intrinsic, dtype in REQUIRED_INTRINSICS + OPTIONAL_INTRINSICS:
        try:
            value = bool(platform.intrinsic_check_support(intrinsic, dtype))
            answered = True
        except Exception:                                # noqa: BLE001 - no such API
            value = None
        report["intrinsic_checks"][f"{intrinsic}:{dtype}"] = value
    report["query_answered"] = answered
    if answered:
        for intrinsic, dtype in REQUIRED_INTRINSICS:
            if report["intrinsic_checks"].get(f"{intrinsic}:{dtype}") is not True:
                report["required_missing"].append(f"{intrinsic}:{dtype}")
        if report["required_missing"]:
            raise FlashVisionError(
                "required instruction(s) unavailable: "
                f"{report['required_missing']}; refusing to substitute a non-Float "
                "reduction, a Half cumulative accumulator or an A2-only instruction. "
                f"Probe: {json.dumps(report['intrinsic_checks'])}")
    report["all_required_supported"] = not report["required_missing"]
    return report


# ---------------------------------------------------------------------------
# Kernel build (cached, locked, never overwriting a previous directory)
# ---------------------------------------------------------------------------


def kernel_name_for(plan: dict) -> str:
    lg, gr = plan["logical"], plan["grid"]
    return ("flash_vision99_h%d_m%d_n%d_d%d_mb%d_c%d_kv%d"
            % (lg["heads"], lg["m"], lg["n"], lg["d"], lg["mb"],
               gr["cores"], gr["kv_tiles"]))


def _fill_flat(t, dst, value, elements: int, per_repeat: int) -> int:
    """Fill ``elements`` scalars with one explicit ``vector_dup`` repeat per call.

    ``repeat_times=1`` with an explicit element slice keeps the fill independent
    of the SDK's multi-repeat stride convention, which the executed Round72 and
    Round97 sources do not agree on. This kernel never relies on that convention.
    The caller must pass ``per_repeat`` equal to one full repeat
    (64 Float32 values = 256 B).
    """
    if elements % per_repeat:
        raise FlashVisionError(f"cannot fill {elements} with {per_repeat}-value repeats")
    repeats = elements // per_repeat
    for index in range(repeats):
        t.vector_dup(per_repeat, dst[index * per_repeat], value, 1, 1, 1)
    return repeats


def _broadcast_row(t, dst_flat, src_row, n1: int, mb: int, scalar) -> int:
    """Expand one Float32 row vector over every NZ group of ``dst_flat``.

    For each of the ``mb`` rows: load the row value into a scalar register, then
    one ``vector_dup`` writes the 16-value (64 B, 32 B aligned) row slot for all
    ``n1`` NZ groups. Group ``g`` row ``r`` lives at ``(g*mb + r)*16`` elements, so
    the repeat stride is ``2*mb`` 32 B blocks and the mask is 16 values. This
    replaces implicit broadcasting, which this SDK does not provide.
    """
    rep_stride = 2 * mb
    if rep_stride > 255:
        raise FlashVisionError("broadcast repeat stride exceeds the 8-bit immediate")
    for row in range(mb):
        scalar.set_as(src_row[row])
        t.vector_dup(BROADCAST_MASK, dst_flat[row * 16], scalar, n1,
                     BROADCAST_BLK_STRIDE, rep_stride)
    return mb


def _broadcast_half_shift(t, dst_float_flat, half_row, scratch_half, broad_half, broad_float, n1, mb):
    """Exact native bit copy/transpose for Half-representable shift constants.

    Proof: results/broadcast01/broadcast_copy99.json. This never casts a general
    Float rescale/denominator to Half; only the already-Half online max is used.
    """
    if mb != 64 or n1 != 4:
        raise FlashVisionError('native shift broadcast currently only score4x64')
    words=half_row.reinterpret_cast_to('uint16')
    matrix=scratch_half.reinterpret_cast_to('uint16')
    for block in range(4):
        t.vec_or(16, matrix, words[block*16], words[block*16], 16, 1, 0, 0)
        t.vec_trans(broad_half[block*256], scratch_half, 1, 0, 0)
    t.h_cast(broad_float, broad_half, 'none')
    for group in range(n1):
        t.data_move(dst_float_flat[group*mb*16], broad_float, 0, 1, mb*16*4//32, 0, 0)


def _emit_kernel(t, plan: dict, tik) -> dict:
    """Emit the program. Every offset and stride comes from ``plan``."""
    lg, gr, dma, vec = plan["logical"], plan["grid"], plan["dma"], plan["vector"]
    mb, bc, d1 = lg["mb"], lg["bc"], lg["d1"]
    batch, m, n = lg["heads"], lg["m"], lg["n"]
    kb1, pad = lg["kb1"], lg["padded_head_dim"]
    cores, steps, tasks, m_tiles, kv_tiles = (gr["cores"], gr["steps"], gr["tasks"],
                                              gr["m_tiles"], gr["kv_tiles"])
    hs = plan["head_stride_elems"]
    score_elems, pv_elems = vec["score_elems"], vec["pv_elems"]
    f32_mask, f32_stride = vec["f32_mask"], vec["f32_rep_stride"]
    group_repeats = vec["group_repeats"]
    score_groups, pv_groups = vec["score_groups"], vec["pv_groups"]
    scale = lg["scale"]

    for attribute in ("vector_dup", "vec_muls", "vec_add", "vec_max", "data_move", "matmul",
                      "tensor_mov", "h_cast", "vec_or", "vec_trans", "h_reduce_max", "h_reduce_sum",
                      "h_sub", "h_add", "h_mul", "h_max", "h_exp", "h_div", "Scalar"):
        if not hasattr(t, attribute):
            raise FlashVisionError(
                f"the installed TIK instance has no {attribute!r}; refusing to emit a "
                "kernel whose online softmax cannot be expressed")

    q_gm = t.Tensor("float16", (batch, d1, m, 16), name="q_gm", scope=tik.scope_gm)
    k_gm = t.Tensor("float16", (batch, d1, n, 16), name="k_gm", scope=tik.scope_gm)
    v_gm = t.Tensor("float16", (batch, n // 16, pad, 16), name="v_gm", scope=tik.scope_gm)
    aux_gm = t.Tensor("float32", (16,), name="aux_gm", scope=tik.scope_gm)
    out_gm = t.Tensor("float16", (batch, d1, m, 16), name="out_gm", scope=tik.scope_gm)

    with t.for_range(0, cores, block_num=cores) as core:
        # L1: staged operands only (no cumulative state).
        q_l1 = t.Tensor("float16", (d1, mb, 16), name="q_l1", scope=tik.scope_cbuf)
        k_l1 = t.Tensor("float16", (d1, bc, 16), name="k_l1", scope=tik.scope_cbuf)
        v_l1 = t.Tensor("float16", (kb1, pad, 16), name="v_l1", scope=tik.scope_cbuf)
        p_hi_l1 = t.Tensor("float16", (kb1, mb, 16), name="p_hi_l1", scope=tik.scope_cbuf)
        p_lo_l1 = t.Tensor("float16", (kb1, mb, 16), name="p_lo_l1", scope=tik.scope_cbuf)
        # L0C: one FP32 score product and two FP32 P@V products.
        score_cc = t.Tensor("float32", (score_groups, mb, 16), name="score_l0c",
                            scope=tik.scope_cc)
        pv_hi_cc = t.Tensor("float32", (pv_groups, mb, 16), name="pv_hi_l0c",
                            scope=tik.scope_cc)
        pv_lo_cc = t.Tensor("float32", (pv_groups, mb, 16), name="pv_lo_l0c",
                            scope=tik.scope_cc)
        # UB. Big tiles are declared flat (so the per-repeat fills and the
        # scale/combine vector ops have explicit slices) with reshaped NZ views.
        s_f_flat = t.Tensor("float32", (score_elems,), name="score_f_ub",
                            scope=tik.scope_ubuf)
        fold_h_a = t.Tensor("float16", (mb, 16), name="fold_h_a", scope=tik.scope_ubuf)
        fold_h_b = t.Tensor("float16", (mb, 16), name="fold_h_b", scope=tik.scope_ubuf)
        fold_f_a = t.Tensor("float32", (mb, 16), name="fold_f_a", scope=tik.scope_ubuf)
        fold_f_b = t.Tensor("float32", (mb, 16), name="fold_f_b", scope=tik.scope_ubuf)
        m_half = t.Tensor("float16", (mb,), name="shift_half", scope=tik.scope_ubuf)
        row_prev = t.Tensor("float32", (mb,), name="row_prev", scope=tik.scope_ubuf)
        row_new = t.Tensor("float32", (mb,), name="row_new", scope=tik.scope_ubuf)
        row_diff = t.Tensor("float32", (mb,), name="row_diff", scope=tik.scope_ubuf)
        row_resc = t.Tensor("float32", (mb,), name="row_rescale", scope=tik.scope_ubuf)
        row_zero = t.Tensor("float32", (mb,), name="row_zero", scope=tik.scope_ubuf)
        row_sum = t.Tensor("float32", (mb,), name="row_sum", scope=tik.scope_ubuf)
        l_f = t.Tensor("float32", (mb,), name="denominator", scope=tik.scope_ubuf)
        p_hi_h_flat = t.Tensor("float16", (score_elems,), name="p_hi_h",
                               scope=tik.scope_ubuf)
        p_hi_f_flat = t.Tensor("float32", (score_elems,), name="p_hi_f",
                               scope=tik.scope_ubuf)
        p_lo_f_flat = t.Tensor("float32", (score_elems,), name="p_lo_f",
                               scope=tik.scope_ubuf)
        p_lo_h_flat = t.Tensor("float16", (score_elems,), name="p_lo_h",
                               scope=tik.scope_ubuf)
        u_hi_flat = t.Tensor("float32", (pv_elems,), name="pv_hi_ub",
                             scope=tik.scope_ubuf)
        u_lo_flat = t.Tensor("float32", (pv_elems,), name="pv_lo_ub",
                             scope=tik.scope_ubuf)
        o_f_flat = t.Tensor("float32", (pv_elems,), name="out_acc_ub",
                            scope=tik.scope_ubuf)
        out_h_flat = t.Tensor("float16", (pv_elems,), name="out_h_ub",
                              scope=tik.scope_ubuf)

        # Reshaped NZ views of the same buffers (metadata only).
        s_f = s_f_flat.reshape((score_groups, mb, 16))
        p_hi_h = p_hi_h_flat.reshape((score_groups, mb, 16))
        p_hi_f = p_hi_f_flat.reshape((score_groups, mb, 16))
        p_lo_f = p_lo_f_flat.reshape((score_groups, mb, 16))
        u_hi = u_hi_flat.reshape((pv_groups, mb, 16))
        u_lo = u_lo_flat.reshape((pv_groups, mb, 16))
        o_f = o_f_flat.reshape((pv_groups, mb, 16))
        out_h = out_h_flat.reshape((pv_groups, mb, 16))

        # Row-broadcast scalar registers (one per broadcast site).
        shift_scalar = t.Scalar("float32", name="shift_broadcast")
        rescale_scalar = t.Scalar("float32", name="rescale_broadcast")
        denom_scalar = t.Scalar("float32", name="denominator_broadcast")

        # Constant that never changes per task.
        shift_copy=t.Tensor('float16',(256,),name='shift_copy',scope=tik.scope_ubuf)
        shift_broad=t.Tensor('float16',(1024,),name='shift_broad',scope=tik.scope_ubuf)
        shift_broad_float=t.Tensor('float32',(1024,),name='shift_broad_float',scope=tik.scope_ubuf)
        _fill_flat(t, row_zero, 0.0, mb, f32_mask)

        with t.for_range(0, steps) as step:
            task = core + step * cores
            with t.if_scope(task < tasks):
                head = task // m_tiles
                mt_idx = task - head * m_tiles
                base_m = mt_idx * mb
                head_q = head * hs["q"]
                head_k = head * hs["k"]
                head_v = head * hs["v"]
                head_out = head * hs["out"]
                q_off = head_q + base_m * 16
                out_off = head_out + base_m * 16

                # 1. Q tile once per task (5 D-blocks x 64 query rows).
                t.data_move(q_l1, q_gm[q_off], 0, dma["q_load"]["nburst"],
                            dma["q_load"]["burst"], dma["q_load"]["src_stride"],
                            dma["q_load"]["dst_stride"])
                # 2. Online state: M=-FLT_MAX, L=0, O=0. No first-block branch.
                _fill_flat(t, row_prev, NEG_MAX, mb, f32_mask)
                _fill_flat(t, l_f, 0.0, mb, f32_mask)
                _fill_flat(t, o_f_flat, 0.0, pv_elems, f32_mask)

                with t.for_range(0, kv_tiles) as kv:
                    # 3. K block (5 x 64) and the matching V block (4 x 80 rows).
                    k_off = head_k + kv * bc * 16
                    v_off = head_v + kv * kb1 * pad * 16
                    with t.new_stmt_scope(disable_sync=False):
                        t.data_move(k_l1, k_gm[k_off], 0, dma["k_load"]["nburst"],
                                    dma["k_load"]["burst"], dma["k_load"]["src_stride"],
                                    dma["k_load"]["dst_stride"])
                        t.data_move(v_l1, v_gm[v_off], 0, dma["v_load"]["nburst"],
                                    dma["v_load"]["burst"], dma["v_load"]["src_stride"],
                                    dma["v_load"]["dst_stride"])
                        # 4. FP32 scores = Q(64x80) @ K(64x80)^T; no narrowing.
                        t.matmul(score_cc, q_l1, k_l1, mb, d1 * 16, bc,
                                 init_l1out=True)
                    with t.new_stmt_scope(disable_sync=False):
                        t.tensor_mov(s_f_flat, score_cc,
                                     dma["score_l0c_to_ub"]["block_mode"],
                                     dma["score_l0c_to_ub"]["nburst"],
                                     dma["score_l0c_to_ub"]["burst_len_kb"], 0, 0)
                    # 5. Original head_dim**-0.5 scale (never the padded 80).
                    for group in range(score_groups):
                        t.vec_muls(f32_mask, s_f[group, :, :], s_f[group, :, :], scale,
                                   group_repeats, f32_stride, f32_stride)
                    # 6. Half row shift estimate: fold the 4 key groups, reduce the
                    #    16 lanes. Half vcmax is supported; Float vcmax is not. The
                    #    Half P_hi buffer is reused as the Half score tile here and is
                    #    rewritten with Half(exp) in step 11.
                    t.h_cast(p_hi_h_flat, s_f_flat, "none")
                    # High-level h_max rejects the sliced tensor's inferred step.
                    # Explicit vector operands preserve the original pairwise fold.
                    t.vec_max(128, fold_h_a, p_hi_h_flat[0], p_hi_h_flat[mb * 16], mb * 16 // 128, 8, 8, 8)
                    t.vec_max(128, fold_h_b, p_hi_h_flat[2 * mb * 16], p_hi_h_flat[3 * mb * 16], mb * 16 // 128, 8, 8, 8)
                    t.vec_max(128, fold_h_a, fold_h_a, fold_h_b, mb * 16 // 128, 8, 8, 8)
                    t.h_reduce_max(m_half, fold_h_a, 1)
                    t.h_cast(row_new, m_half, "none")
                    # 7. Online update: new = max(prev, row); rescale = exp(prev-new).
                    t.h_max(row_new, row_prev, row_new)
                    t.h_sub(row_diff, row_prev, row_new)
                    t.h_exp(row_resc, row_diff)
                    t.h_add(row_prev, row_new, row_zero)
                    # 8. exp(S - shift) in FP32 with an explicit row expansion. The
                    #    P_hi Float buffer holds the expansion and is then overwritten
                    #    with the widened Half(exp); the exponential is in place, the
                    #    same pattern the executed flash-attention source uses.
                    # row_new remains exactly Half-representable: a max of Half
                    # estimates. Copy/transpose its bits rather than 64 Scalar ops.
                    t.h_cast(m_half, row_new, 'none')
                    _broadcast_half_shift(t, p_hi_f_flat, m_half, shift_copy, shift_broad, shift_broad_float, score_groups, mb)
                    t.h_sub(s_f, s_f, p_hi_f)
                    t.h_exp(s_f, s_f)
                    # 9. TRUE FP32 denominator of this block: fold 4 groups, sum 16.
                    t.vec_add(f32_mask, fold_f_a, s_f_flat[0], s_f_flat[mb * 16], group_repeats, f32_stride, f32_stride, f32_stride)
                    t.vec_add(f32_mask, fold_f_b, s_f_flat[2 * mb * 16], s_f_flat[3 * mb * 16], group_repeats, f32_stride, f32_stride, f32_stride)
                    t.vec_add(f32_mask, fold_f_a, fold_f_a, fold_f_b, group_repeats, f32_stride, f32_stride, f32_stride)
                    t.h_reduce_sum(row_sum, fold_f_a, 1)
                    # 10. L = rescale * L + block_sum (FP32, per row).
                    t.h_mul(row_diff, l_f, row_resc)
                    t.h_add(l_f, row_diff, row_sum)
                    # 11. Two-Half probability split, no Torch and no GM round trip.
                    t.h_cast(p_hi_h_flat, s_f_flat, "none")
                    t.h_cast(p_hi_f_flat, p_hi_h_flat, "none")
                    t.h_sub(p_lo_f, s_f, p_hi_f)
                    for group in range(score_groups):
                        t.vec_muls(f32_mask, p_lo_f[group, :, :], p_lo_f[group, :, :],
                                   RESIDUAL_SCALE, group_repeats, f32_stride, f32_stride)
                    t.h_cast(p_lo_h_flat, p_lo_f_flat, "none")
                    # 12. P tiles to L1, then two full-K FP32 P@V products.
                    with t.new_stmt_scope(disable_sync=False):
                        t.data_move(p_hi_l1, p_hi_h_flat[0], 0,
                                    dma["p_to_l1"]["nburst"], dma["p_to_l1"]["burst"], 0, 0)
                        t.data_move(p_lo_l1, p_lo_h_flat[0], 0,
                                    dma["p_to_l1"]["nburst"], dma["p_to_l1"]["burst"], 0, 0)
                        t.matmul(pv_hi_cc, p_hi_l1, v_l1, mb, bc, pad, init_l1out=True)
                        t.matmul(pv_lo_cc, p_lo_l1, v_l1, mb, bc, pad, init_l1out=True)
                    # 13. FP32 combine: low first, then high; hi + lo/1024.
                    with t.new_stmt_scope(disable_sync=False):
                        t.tensor_mov(u_lo_flat, pv_lo_cc, dma["pv_l0c_to_ub"]["block_mode"],
                                     dma["pv_l0c_to_ub"]["nburst"],
                                     dma["pv_l0c_to_ub"]["burst_len_kb"], 0, 0)
                        t.tensor_mov(u_hi_flat, pv_hi_cc, dma["pv_l0c_to_ub"]["block_mode"],
                                     dma["pv_l0c_to_ub"]["nburst"],
                                     dma["pv_l0c_to_ub"]["burst_len_kb"], 0, 0)
                        for group in range(pv_groups):
                            t.vec_muls(f32_mask, u_lo[group, :, :], u_lo[group, :, :],
                                       LOW_DOWNSCALE, group_repeats, f32_stride, f32_stride)
                        for group in range(pv_groups):
                            t.vec_add(f32_mask, u_hi[group, :, :], u_hi[group, :, :],
                                      u_lo[group, :, :], group_repeats, f32_stride,
                                      f32_stride, f32_stride)
                    # 14. O = rescale * O + Y (FP32). The low product buffer is dead
                    #     after the combine and is reused for the row expansion.
                    _broadcast_row(t, u_lo_flat, row_resc, pv_groups, mb, rescale_scalar)
                    t.h_mul(o_f, o_f, u_lo)
                    t.h_add(o_f, o_f, u_hi)

                # 15. O / L in FP32 (the high product buffer is dead), then ONE Half
                #     cast and the store: 5 blocks x 64 rows, stride M-64.
                _broadcast_row(t, u_lo_flat, l_f, pv_groups, mb, denom_scalar)
                t.h_div(u_hi, o_f, u_lo)
                t.h_cast(out_h_flat, u_hi_flat, "none")
                t.data_move(out_gm[out_off], out_h_flat[0], 0,
                            dma["out_store"]["nburst"], dma["out_store"]["burst"],
                            dma["out_store"]["src_stride"], dma["out_store"]["dst_stride"])

    return {"q": q_gm, "k": k_gm, "v": v_gm, "aux": aux_gm, "out": out_gm}


# ---------------------------------------------------------------------------
# Build / describe
# ---------------------------------------------------------------------------


def build_kernel(batch: int = HEADS, m: int = SEQ, n: int = SEQ,
                 d: int = HEAD_DIM, out_dir=None, cores_max: int = MAX_CORES) -> dict:
    """Build (or reuse) one kernel and describe it; never overwrite a directory."""
    plan = plan_kernel(batch, m, n, d, cores_max=cores_max)
    _check_budget(plan)

    from tbe import tik

    capability = capability_report(plan)
    name = kernel_name_for(plan)
    out_root = Path(out_dir).resolve() if out_dir is not None else \
        Path.cwd() / "flash_vision99_kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec99.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability)

    with (out_root / ".compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan,
                                    capability)
        work = Path(tempfile.mkdtemp(prefix="flash99_build_", dir=out_root)).resolve()
        if work.parent != out_root:
            raise FlashVisionError("unexpected build workspace")
        t = tik.Tik(disable_debug=True)
        gm = _emit_kernel(t, plan, tik)
        previous = os.getcwd()
        try:
            os.chdir(work)
            # All four inputs are explicit so the unused Float aux placeholder keeps
            # the proven five-pointer ABI visible to the compiler.
            t.BuildCCE(kernel_name=name,
                       inputs=[gm["q"], gm["k"], gm["v"], gm["aux"]],
                       outputs=[gm["out"]])
        finally:
            os.chdir(previous)
        built_obj = work / "kernel_meta" / (name + ".o")
        built_meta = work / "kernel_meta" / (name + ".json")
        if not built_obj.is_file() or not built_meta.is_file():
            raise FlashVisionError(f"BuildCCE produced no kernel_meta artefacts for {name}")
        if dest.exists():
            raise FlashVisionError(f"refusing to overwrite existing kernel directory {dest}")
        dest.mkdir(parents=True)
        import shutil

        shutil.copyfile(built_obj, obj)
        shutil.copyfile(built_meta, meta_path)

    return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                            compiled_now=True)


def scale_check(plan: dict) -> dict:
    """Prove the emitted scale is the original head dim, not the padded width."""
    d = plan["logical"]["d"]
    return {"scale": plan["logical"]["scale"], "original_head_dim": d,
            "padded_head_dim": plan["logical"]["padded_head_dim"],
            "matches_original": abs(plan["logical"]["scale"] - scale_for(d)) < 1e-15,
            "padded_width_scale_used": False}


def _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                     compiled_now=False) -> dict:
    meta = json.loads(meta_path.read_text())
    if int(meta.get("blockDim", -1)) != plan["grid"]["cores"]:
        raise FlashVisionError(
            f"cached kernel {name} has blockDim {meta.get('blockDim')} but the plan "
            f"needs {plan['grid']['cores']}; refusing to reuse a mismatched binary")
    spec = {
        "round": 99,
        "scope": plan["scope"],
        "executed": False,
        "compiled_now": compiled_now,
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "numerical_tolerance": None,
        "kernel_name": name,
        "kernel_symbol": meta["kernelName"],
        "block_dim": int(meta["blockDim"]),
        "core_type": meta.get("coreType"),
        "obj_path": str(obj),
        "obj_sha256": _sha256_file(obj),
        "json_path": str(meta_path),
        "source_sha256": _sha256_file(Path(__file__)),
        "dtype": "float16_in_float16_out",
        "input_dtype": "float16",
        "accumulator_dtype": "float32",
        "output_dtype": "float16",
        "kernel_arg_order": list(KERNEL_ARG_ORDER),
        "GM_argument_count": 5,
        "logical": plan["logical"],
        "grid": plan["grid"],
        "head_stride_elems": plan["head_stride_elems"],
        "gm_shapes": plan["gm_shapes"],
        "gm_bytes": plan["gm_bytes"],
        "dma": plan["dma"],
        "buffers": plan["buffers"],
        "vector": plan["vector"],
        "math": plan["math"],
        "host_protocol": plan["host_protocol"],
        "assumptions": plan["assumptions"],
        "single_launch_streaming": True,
        "no_mid_product_fp32_gm": True,
        "no_half_cumulative_state": True,
        "no_implicit_broadcast": True,
        "unmasked_attention": True,
        "scale_check": scale_check(plan),
        "capability": capability,
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", nargs=4, type=int, metavar=("H", "M", "N", "D"))
    parser.add_argument("--capability", action="store_true")
    parser.add_argument("--build", nargs=4, type=int, metavar=("H", "M", "N", "D"))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps(plan_kernel(*args.plan), indent=2, sort_keys=True))
    elif args.capability:
        print(json.dumps(capability_report(), indent=2, sort_keys=True))
    elif args.build:
        print(json.dumps(build_kernel(*args.build, out_dir=args.out_dir), indent=2,
                         sort_keys=True))
    else:
        parser.error("choose one of --plan, --capability or --build H M N D")


if __name__ == "__main__":
    main()