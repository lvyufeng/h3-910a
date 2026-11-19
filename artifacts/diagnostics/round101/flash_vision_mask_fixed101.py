#!/usr/bin/env python3
"""Round101: one-launch streaming vision flash attention, MB=64 x BC=128 (source only).

STATUS: SOURCE ONLY. Its author never compiles, launches or imports this on an NPU:
no SSH, no CANN build, no torch, no torch_npu, no model, no GPU/NPU, no git, no
production edit. Every number belongs to the root driver, which compiles this
builder and compares the result against the frozen real Q/K/V and the original
Round78 CPU FP16 reference. This file claims no accuracy, no speed, no tolerance
and no default-enable result.

WHAT CHANGES VERSUS THE EXECUTED ROUND99 KERNEL
-----------------------------------------------
Round99 used MB=BC=64, i.e. 63 KV iterations per query tile. This variant keeps
MB=64 for the query tile but doubles the KV block to BC=128, so the dynamic KV
loop drops to 32 iterations. Two consequences drive the whole design:

* The FP32 score tile doubles to 8 NZ key groups (32 KiB in UB, 32 KiB in L0C).
  The real key length 4032 is padded to n_pad = 4096, so the LAST KV tile holds
  only 64 real keys: its groups 4..7 are *phantom* keys made of the zero-padded
  K rows. Their scores are exactly 0 before scaling, which after ``exp`` would
  contribute a full row of 1.0 to the denominator. Zero-padded V alone does NOT
  fix that: an inflated denominator shrinks every real probability. Therefore
  the phantom score groups are explicitly forced to ``-FLT_MAX`` *before* the
  Half shift estimate and the Float exponential, and explicitly zeroed again
  after ``exp`` and *before* the Float denominator.
* UB is tight, so one Float probability buffer is removed. After the TRUE Float
  denominator has been folded and summed, ``P_hi = Half(exp)`` is widened back to
  Float, and the score buffer itself is overwritten in place with
  ``(exp - Float(P_hi)) * 1024`` to produce ``P_lo``. The next KV iteration
  reloads the score tile from the QK L0C product, so no separate low-residual
  Float tile is needed.

MATH (unchanged where it matters)
---------------------------------
``QK`` stays a HalfxHalf -> FP32 L0C product over the padded head width 80 and
the score tile stays FP32 in UB: the main scores are never narrowed. A **Half**
row maximum is used only as a stable *shift estimate* (Float ``vcmax`` does not
exist on this part); the host protocol below keeps ``exp(S - shift)`` inside the
Half range. Everything accumulated across KV blocks -- running shift ``M``,
denominator ``L`` and output ``O`` -- is FP32. ``P_hi``/``P_lo`` feed two FP32
L0C P@V products whose FP32 UB combination ``hi + lo/1024`` is narrowed exactly
once, in the final output cast. The true denominator is the FP32 sum of
``exp(S - shift)``; it is never a Half sum.

Vision attention is UNMASKED (not causal) and the scale is the original
``72 ** -0.5``: the padded width 80 is never the scale.

GEOMETRY
--------
Real: H=16 heads, M=N=4032 rows (48x84 patch grid, before the 2x2 merger),
D=72 zero-padded to 80 = 5 NZ blocks. Query tiles: 4032/64 = 63. Padded key
length: ceil(4032/128)*128 = 4096 -> 32 KV blocks, last one half real. Tasks:
16*63 = 1008 over 30 cores = 34 dynamic steps. A small smoke geometry
(H=2, M=128, N=192, d=72 -> n_pad=256) is accepted and also ends with 64 real
keys plus 64 phantom keys in its last KV block, so the smoke exercises the
padding path; it uses block_dim=min(30, tasks)=4.

HOST PROTOCOL (enforced by the caller; never clamped here)
---------------------------------------------------------
``max |scaled QK| <= 16384`` measured finitely, or bounded by
``sqrt(72) * max|q| * max|k| <= 16384``. At that bound the Half shift error is at
most 8 > 0, so the largest exponent stays in ``[e^-8, e^8]`` and the two-Half
probability split is safe; smaller exponents may underflow to zero exactly as in
any stable softmax. :func:`check_host_protocol` raises on violation.

NON-CLAIMS
----------
No FP32 linear weight, no teacher tensor, no BF16 gold cast, no cached
hidden/patch/conditioning value, no Half cumulative accumulator, no clamp, no
post-hoc tolerance, no measured speed. The changed KV blocking and the changed
Float reduction association can move the low bits relative to Round99: this
builder does not promise Round99's bits, only the same probability formula. The
root's frozen real-input CPU Half MATH comparison is the authority.
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
MB = 64                    # query rows per tile (unchanged)
BC = 128                   # key/value rows per KV block (was 64 in Round99)
MAX_CORES = 30             # usable AI cores per original 910A-generation chip
PINNED_COMPILE_SOC = "Ascend910"

D1 = PAD_DIM // 16         # 5 head-dim NZ blocks
KB1 = BC // 16             # 8 key NZ blocks per KV block
PHANTOM_KEYS = 64          # nominal padded key tail (4032 -> 4096)
F32_MASK = 64              # one Float32 repeat (64 values = 256 B)
F32_REP_STRIDE = 8         # 8 x 32 B = 256 B = 64 Float32 values (contiguous)
HALF_MASK = 128            # one Half repeat (128 values = 256 B)
HALF_REP_STRIDE = 8        # 8 x 32 B = 256 B = 128 Half values (contiguous)
BROADCAST_MASK = 16        # one NZ row slot = 16 Float32 = 64 B
BROADCAST_BLK_STRIDE = 1
RESIDUAL_SCALE = 1024.0
LOW_DOWNSCALE = 1.0 / RESIDUAL_SCALE
NEG_MAX = -3.4028234663852886e+38
SHIFT_BOUND = 16384.0      # protocol: exp(S-shift) stays inside Half
SHIFT_HALF_ULP_AT_BOUND = 8.0   # Half ulp at 2**14 is 16, so half-ulp is 8
EXP_MAX_AT_BOUND = 2980.9579870417283   # e**8
PINNED_UB_SAFE_BYTES = 196608
SMOKE_GEOMETRY = (2, 128, 192, HEAD_DIM)   # n_pad 256: last KV block half real

KERNEL_ARG_ORDER = ["q_gm_float16", "k_gm_float16", "v_gm_float16",
                    "aux_gm_float32", "out_gm_float16"]

ASSUMPTIONS = (
    "h_reduce_max/h_reduce_sum(axis=1) on a (MB,16) tile yields a (MB,) row tensor",
    "Tensor.reshape((n1,MB,16)) is a metadata view of the same flat UB buffer",
    "masked vector_dup(repeat_times=n1, blk_stride=1, rep_stride=2*MB=128) writes "
    "one 16-value NZ row slot per group; 128 is inside the 8-bit immediate",
    "Scalar.set_as(<UB element>) is a legal UB-to-scalar load",
    "h_reduce_sum on Float compiles to Float vcadd; h_max to Float vmax",
    "the 8-group Half fold is emitted as explicit vec_max calls, because the "
    "high-level h_max rejected the inferred step of a sliced NZ tensor in Round99",
    "the Half shift estimate uses Half vcmax (Float vcmax is unsupported here)",
    "vector_dup(mask,dst,scalar,1,1,1) with an explicit repeat slice fills exactly one repeat",
    "in-place same-shape h_sub/h_mul/h_add/h_exp with dst==src0 is supported",
    "a Float UB buffer can be reused sequentially (score->exp->low residual, "
    "Half score->P_hi, row expansion->widened P_hi) without an explicit alias API",
    "h_cast of -FLT_MAX to Half saturates rather than producing NaN; the phantom "
    "mask is still correct if it saturates to -inf, because every real scaled "
    "score is above -16384 so max() selects a real value",
    "a dynamic if_scope (no else_scope) around only vector_dup phantom fills does "
    "not re-trigger the Round99 high-level if/else canonicalisation crash",
    "a 32 KiB single tensor_mov L0C->UB burst is legal (Round85 executed a 64 KiB "
    "single burst of the same form)",
    "the 8-group Float denominator fold is emitted as three pairwise vec_add "
    "stages, so its association differs from Round99's 4-group fold",
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


def pad_keys(n: int, bc: int = BC) -> int:
    """Padded key length: the smallest multiple of ``bc`` covering ``n`` keys."""
    return ((n + bc - 1) // bc) * bc


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
            f"this kernel is specialised to mb={MB}, bc={BC}; got mb={mb}, bc={bc}")
    if m % mb:
        raise FlashVisionError(f"m={m} is not a multiple of the query tile mb={mb}")
    if n % 64:
        # Real key length must be a multiple of 64 so a padded KV block ends with
        # either a fully real tail or exactly one 64-key phantom tail.
        raise FlashVisionError(f"n={n} is not a multiple of 64")

    n_pad = pad_keys(n, bc)
    tail_keys = n - (n_pad // bc - 1) * bc
    phantom_keys = bc - tail_keys
    if phantom_keys not in (0, 64):
        raise FlashVisionError(
            f"n={n} leaves a {phantom_keys}-key phantom tail; only 0 or 64 is supported")

    m_tiles = m // mb
    kv_tiles = n_pad // bc
    tasks = batch * m_tiles
    cores = min(cores_max, tasks)
    steps = (tasks + cores - 1) // cores
    score_groups = bc // 16                     # 8 FP32/Half NZ key groups
    pv_groups = D1                              # 5 NZ head-dim groups
    score_elems = score_groups * mb * 16        # 8192
    pv_elems = pv_groups * mb * 16              # 5120
    phantom_groups = phantom_keys // 16         # 4 for the real 4032 geometry
    real_groups_last = tail_keys // 16          # 4 for the real 4032 geometry

    head_stride = {"q": D1 * m * 16, "k": D1 * n_pad * 16,
                   "v": (n_pad // 16) * PAD_DIM * 16, "out": D1 * m * 16}
    gm_shapes = {
        "q": [batch, D1, m, 16],
        "k": [batch, D1, n_pad, 16],
        "v": [batch, n_pad // 16, PAD_DIM, 16],
        "aux": [16],
        "aux_dtype": "float32",
        "out": [batch, D1, m, 16],
    }
    dma = {
        # One 32 B unit is one 16-value Half NZ row. Q/K blocks are m/n_pad rows
        # with the remaining rows skipped; V is a contiguous run of 8 key blocks
        # x 80 dim rows for the current KV block.
        "q_load": {"nburst": D1, "burst": mb, "src_stride": m - mb, "dst_stride": 0},
        "k_load": {"nburst": D1, "burst": bc, "src_stride": n_pad - bc, "dst_stride": 0},
        "v_load": {"nburst": bc // 16, "burst": PAD_DIM, "src_stride": 0, "dst_stride": 0},
        "p_to_l1": {"nburst": bc // 16, "burst": mb},   # 8 x 64 units
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
        "ub_prob_half_bytes": score_elems * 2,
        "ub_prob_float_bytes": score_elems * 4,
        "ub_pv_bytes": pv_elems * 4,
        "ub_out_bytes": pv_elems * 2,
    }
    # These per-buffer aliases are documentation only (they name the same bytes
    # that appear in ub_named under their reuse names); they are NEVER summed into
    # ub_total_bytes, so an audit must not add them again.
    buffers["informational_ub_aliases_not_summed"] = [
        "ub_score_bytes", "ub_half_score_bytes", "ub_exp_bytes",
        "ub_prob_half_bytes", "ub_prob_float_bytes", "ub_pv_bytes", "ub_out_bytes"]
    buffers["l1_total_bytes"] = sum(v for k, v in buffers.items() if k.startswith("l1_"))
    buffers["l0c_total_bytes"] = (buffers["l0c_score_bytes"]
                                  + 2 * buffers["l0c_pv_bytes"])
    # Flat accounting of every UB allocation the emitter declares. The score tile
    # is reused as the exp tile and then as the low-residual tile; the Half score
    # tile is reused as P_hi; the widened P_hi tile is first the row-expansion
    # destination. Round99's separate low-residual Float tile is gone.
    ub_named = {
        "score_float_also_exp_also_lo": score_elems * 4,
        "fold_half_a": mb * 16 * 2,
        "fold_half_b": mb * 16 * 2,
        "shift_half_row": mb * 2,
        "fold_float_a": mb * 16 * 4,
        "fold_float_b": mb * 16 * 4,
        "prob_hi_half_also_half_score": score_elems * 2,
        "prob_hi_float_also_row_broadcast": score_elems * 4,
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
    buffers["ub_named_total_bytes"] = sum(ub_named.values())
    # Extra UB for the exact native Half-shift broadcast (Round99 helper).
    buffers["ub_shift_copy_bytes"] = 256 * 2
    buffers["ub_shift_broad_half_bytes"] = 1024 * 2
    buffers["ub_shift_broad_float_bytes"] = 1024 * 4
    buffers["ub_shift_extra_bytes"] = sum(
        buffers[k] for k in ("ub_shift_copy_bytes", "ub_shift_broad_half_bytes",
                             "ub_shift_broad_float_bytes"))
    buffers["ub_total_bytes"] = (buffers["ub_named_total_bytes"]
                                 + buffers["ub_shift_extra_bytes"])
    vector = {
        "f32_mask": F32_MASK, "f32_rep_stride": F32_REP_STRIDE,
        "group_repeats": (mb * 16) // F32_MASK,
        "half_mask": HALF_MASK, "half_rep_stride": HALF_REP_STRIDE,
        "half_repeats": (mb * 16) // HALF_MASK,
        "broadcast_mask": BROADCAST_MASK,
        "broadcast_blk_stride": BROADCAST_BLK_STRIDE,
        "broadcast_rep_stride": 2 * mb,
        "score_elems": score_elems, "pv_elems": pv_elems,
        "score_groups": score_groups, "pv_groups": pv_groups,
        "row_elems": mb,
    }
    plan = {
        "round": 101,
        "scope": ("one-launch streaming vision flash attention, MB=64 x BC=128: FP32 "
                  "online softmax, Half shift estimate, two-Half probability P@V, one "
                  "final Half cast, phantom key tail masked out of the denominator"),
        "logical": {"heads": batch, "m": m, "n": n, "n_pad": n_pad, "d": d,
                    "padded_head_dim": PAD_DIM, "d1": D1, "kb1": KB1,
                    "mb": mb, "bc": bc, "scale": scale_for(d),
                    "score_groups": score_groups, "pv_groups": pv_groups,
                    "tail_keys": tail_keys, "phantom_keys": phantom_keys,
                    "phantom_groups": phantom_groups, "real_groups_last": real_groups_last,
                    "unmasked": True, "causal": False},
        "grid": {"tasks": tasks, "cores": cores, "steps": steps,
                 "m_tiles": m_tiles, "kv_tiles": kv_tiles,
                 "dynamic_kv_loop": True},
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
            "probability_split": ["hi = Half(exp)",
                                  "store_lo = (exp - Float(hi)) * 1024 in place",
                                  "lo = Half(store_lo)"],
            "residual_scale": RESIDUAL_SCALE,
            "low_downscale": LOW_DOWNSCALE,
            "init": {"row_max": NEG_MAX, "denominator": 0.0, "output": 0.0},
            "row_broadcast": ("explicit per-row vector_dup expansion (no implicit "
                              "broadcast); the Half-representable shift uses the "
                              "exact native uint16 copy + Half transpose path"),
            "phantom_key_mask": {
                "keys": phantom_keys, "groups": phantom_groups,
                "before_shift_and_exp": NEG_MAX,
                "after_exp_before_denominator": 0.0,
                "why": ("zero-padded K rows make the phantom scores exactly 0; "
                        "exp(0 - shift) would add 1.0 to every denominator row"),
                "padded_v_alone_is_not_sufficient": True,
            },
            "denominator_fold": {
                "groups": score_groups,
                "order": ("fold_f_a=(g0+g1); fold_f_b=(g2+g3); fold_f_a=fold_f_a+fold_f_b; "
                          "fold_f_b=(g4+g5); fold_f_a=fold_f_a+fold_f_b; "
                          "fold_f_b=(g6+g7); fold_f_a=fold_f_a+fold_f_b"),
                "association_changes_low_bits_vs_round99": True,
                "reduction": "h_reduce_sum(axis=1) -> Float vcadd",
            },
            "shift_fold": {
                "groups": score_groups,
                "order": ("pairwise Half vec_max in the same 3+4 stage pattern as the "
                          "denominator fold; max is exact under any association"),
            },
            "score_l0c_burst_kb": score_elems * 4 // 1024,
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
    _check_budget(plan)
    return plan


def _naive_alias_sum_guard(plan: dict) -> int:
    """Return the UB total that ignores the informational alias entries.

    Exists so the tests can prove the alias keys are not double counted.
    """
    buf = plan["buffers"]
    return buf["ub_named_total_bytes"] + buf["ub_shift_extra_bytes"]


def _check_budget(plan: dict) -> None:
    buf = plan["buffers"]
    vec = plan["vector"]
    lg = plan["logical"]
    if vec["group_repeats"] > 255 or vec["half_repeats"] > 255:
        raise FlashVisionError("vector repeat count exceeds the 8-bit immediate")
    if vec["broadcast_rep_stride"] > 255:
        raise FlashVisionError("broadcast repeat stride exceeds the 8-bit immediate")
    if vec["broadcast_mask"] * 4 % 32:
        raise FlashVisionError("broadcast row slot is not 32 B aligned")
    if vec["score_elems"] != vec["score_groups"] * lg["mb"] * 16:
        raise FlashVisionError("score tile element count disagrees with its NZ shape")
    if vec["pv_elems"] != vec["pv_groups"] * lg["mb"] * 16:
        raise FlashVisionError("P@V tile element count disagrees with its NZ shape")
    if buf["l1_total_bytes"] > 1024 * 1024:
        raise FlashVisionError(f"L1 need {buf['l1_total_bytes']} B exceeds a safe bound")
    if buf["l0c_total_bytes"] > 262144:
        raise FlashVisionError(f"L0C need {buf['l0c_total_bytes']} B exceeds the part")
    if buf["ub_total_bytes"] > PINNED_UB_SAFE_BYTES:
        raise FlashVisionError(
            f"UB need {buf['ub_total_bytes']} B exceeds the {PINNED_UB_SAFE_BYTES} B "
            "safe bound")


def check_host_protocol(max_abs_scaled_qk=None, q_absmax=None, k_absmax=None,
                        d: int = HEAD_DIM) -> dict:
    """Validate the documented shift bound; raise (never clamp/shrink) on violation."""
    import math

    if max_abs_scaled_qk is None and (q_absmax is None or k_absmax is None):
        raise FlashVisionError(
            "a finite measured or complete Q/K bound is required; no unguarded Half shift")
    if (q_absmax is None) != (k_absmax is None):
        raise FlashVisionError("both absolute Q/K maxima are required together")
    if d != HEAD_DIM:
        raise FlashVisionError(f"the current kernel uses the original head dimension "
                               f"{HEAD_DIM}")
    for value in (max_abs_scaled_qk, q_absmax, k_absmax):
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise FlashVisionError("absolute input bounds must be finite and nonnegative")
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
                f"{SHIFT_BOUND} shift protocol; supply a finite bound inside a "
                "supported input domain (no rescaling or clamp is applied)")
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
    ("Intrinsic_vtranspose", "float16"),# native 16x16 Half transpose
    ("Intrinsic_vmax", "float16"),      # explicit Half fold (high-level slices failed)
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
              "launched_usable_cores": plan["grid"]["cores"],
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
    return ("flash_vision101_h%d_m%d_n%d_np%d_d%d_mb%d_bc%d_c%d_kv%d"
            % (lg["heads"], lg["m"], lg["n"], lg["n_pad"], lg["d"], lg["mb"],
               lg["bc"], gr["cores"], gr["kv_tiles"]))


def _fill_flat(t, dst, value, elements: int, per_repeat: int) -> int:
    """Fill ``elements`` scalars with one explicit ``vector_dup`` repeat per call.

    ``repeat_times=1`` with an explicit element slice keeps the fill independent of
    the SDK multi-repeat stride convention (the executed Round72 and Round97
    sources do not agree on it). ``per_repeat`` must be one full repeat
    (64 Float32 = 256 B).
    """
    if elements % per_repeat:
        raise FlashVisionError(f"cannot fill {elements} with {per_repeat}-value repeats")
    repeats = elements // per_repeat
    for index in range(repeats):
        t.vector_dup(per_repeat, dst[index * per_repeat], value, 1, 1, 1)
    return repeats


def _mask_phantom_scores(t, s_f_flat, plan: dict, value: float) -> int:
    """Force the phantom key groups of the last KV block to ``value``.

    The phantom groups are the trailing ``phantom_groups`` NZ groups of the score
    tile, i.e. the key slots at or beyond the real key length. They are filled with
    ``-FLT_MAX`` before the shift/exp (so they can never win the max and their
    exponential underflows to zero) and with ``0.0`` after exp (so the Float
    denominator is exactly the sum over real keys).
    """
    lg = plan["logical"]
    groups = lg["phantom_groups"]
    if not groups:
        return 0
    mb = lg["mb"]
    start_group = lg["score_groups"] - groups
    start = start_group * mb * 16
    elements = groups * mb * 16
    # A Tensor element slice has logical extent1 in actual TIK; do not
    # re-index that slice. Address each full64-lane vector from the base tensor.
    for offset in range(0,elements,F32_MASK):
        t.vector_dup(F32_MASK, s_f_flat[start+offset], value, 1, 1, 1)
    return elements // F32_MASK


def _broadcast_half_shift(t, dst_float_flat, half_row, scratch_half, broad_half,
                          broad_float, n1: int, mb: int) -> int:
    """Expand one Half-representable row value across every NZ group, exactly.

    Only an already-Half online max is broadcast this way: its bits are copied
    through uint16 ``vec_or`` and a native Half ``vec_trans``, cast once into a
    1024-element Float holding tile, then copied into each of the ``n1`` NZ groups.
    No general Float rescale or denominator is ever narrowed to Half.
    """
    if mb != MB:
        raise FlashVisionError(f"shift broadcast is specialised to mb={MB}")
    if n1 not in (4, 8):
        raise FlashVisionError(f"shift broadcast supports 4 or 8 NZ groups, got {n1}")
    words = half_row.reinterpret_cast_to("uint16")
    matrix = scratch_half.reinterpret_cast_to("uint16")
    for block in range(4):
        t.vec_or(16, matrix, words[block * 16], words[block * 16], 16, 1, 0, 0)
        t.vec_trans(broad_half[block * 256], scratch_half, 1, 0, 0)
    t.h_cast(broad_float, broad_half, "none")
    for group in range(n1):
        t.data_move(dst_float_flat[group * mb * 16], broad_float, 0, 1,
                    mb * 16 * 4 // 32, 0, 0)
    return n1


def _broadcast_row(t, dst_flat, src_row, n1: int, mb: int, scalar) -> int:
    """Expand one Float32 row vector over every NZ group of ``dst_flat``.

    For each of the ``mb`` rows: load the row value into a scalar register, then
    one ``vector_dup`` writes the 16-value (64 B, 32 B aligned) row slot for all
    ``n1`` NZ groups. Group ``g`` row ``r`` lives at ``(g*mb + r)*16`` elements, so
    the repeat stride is ``2*mb`` 32 B blocks and the mask is 16 values.
    """
    rep_stride = 2 * mb
    if rep_stride > 255:
        raise FlashVisionError("broadcast repeat stride exceeds the 8-bit immediate")
    for row in range(mb):
        scalar.set_as(src_row[row])
        t.vector_dup(BROADCAST_MASK, dst_flat[row * 16], scalar, n1,
                     BROADCAST_BLK_STRIDE, rep_stride)
    return mb


def _fold_half_max(t, fold_a, fold_b, flat, groups: int, mb: int) -> int:
    """Pairwise Half max over ``groups`` NZ groups with explicit vec_max operands.

    The high-level ``h_max`` rejects the inferred step of a sliced NZ tensor on
    this CANN release (Round99 found that at compile time), so every stage is an
    explicit ``vec_max`` on flat operands. Max is exact under any association.
    """
    repeats = mb * 16 // HALF_MASK
    stride = HALF_REP_STRIDE
    t.vec_max(HALF_MASK, fold_a, flat[0], flat[mb * 16], repeats, stride, stride, stride)
    pending = None
    for left in range(2, groups, 2):
        t.vec_max(HALF_MASK, fold_b, flat[left * mb * 16], flat[(left + 1) * mb * 16],
                  repeats, stride, stride, stride)
        t.vec_max(HALF_MASK, fold_a, fold_a, fold_b, repeats, stride, stride, stride)
        pending = left + 2
    if pending != groups:
        raise FlashVisionError("Half fold group count is not a power-of-two walk")
    return groups - 1


def _fold_float_sum(t, fold_a, fold_b, flat, groups: int, mb: int,
                    mask: int, stride: int, repeats: int) -> int:
    """Pairwise Float sum over ``groups`` NZ groups, three explicit stages.

    Association (evidence, not an optimisation claim):
    ``a=g0+g1; b=g2+g3; a=a+b; b=g4+g5; a=a+b; b=g6+g7; a=a+b``.
    """
    if groups != 8:
        raise FlashVisionError(f"the Float fold is specialised to 8 NZ groups, got {groups}")
    e = mb * 16
    t.vec_add(mask, fold_a, flat[0], flat[e], repeats, stride, stride, stride)
    t.vec_add(mask, fold_b, flat[2 * e], flat[3 * e], repeats, stride, stride, stride)
    t.vec_add(mask, fold_a, fold_a, fold_b, repeats, stride, stride, stride)
    t.vec_add(mask, fold_b, flat[4 * e], flat[5 * e], repeats, stride, stride, stride)
    t.vec_add(mask, fold_a, fold_a, fold_b, repeats, stride, stride, stride)
    t.vec_add(mask, fold_b, flat[6 * e], flat[7 * e], repeats, stride, stride, stride)
    t.vec_add(mask, fold_a, fold_a, fold_b, repeats, stride, stride, stride)
    return 7


def _emit_kernel(t, plan: dict, tik) -> dict:
    """Emit the program. Every offset and stride comes from ``plan``."""
    lg, gr, dma, vec = plan["logical"], plan["grid"], plan["dma"], plan["vector"]
    mb, bc, d1 = lg["mb"], lg["bc"], lg["d1"]
    batch, m, n_pad = lg["heads"], lg["m"], lg["n_pad"]
    kb1, pad = lg["kb1"], lg["padded_head_dim"]
    cores, steps, tasks, m_tiles, kv_tiles = (gr["cores"], gr["steps"], gr["tasks"],
                                              gr["m_tiles"], gr["kv_tiles"])
    score_groups, pv_groups = lg["score_groups"], lg["pv_groups"]
    phantom_groups = lg["phantom_groups"]
    hs = plan["head_stride_elems"]
    score_elems, pv_elems = vec["score_elems"], vec["pv_elems"]
    f32_mask, f32_stride = vec["f32_mask"], vec["f32_rep_stride"]
    group_repeats = vec["group_repeats"]
    scale = lg["scale"]

    for attribute in ("vector_dup", "vec_muls", "vec_add", "vec_max", "data_move",
                      "matmul", "tensor_mov", "h_cast", "vec_or", "vec_trans",
                      "h_reduce_max", "h_reduce_sum", "h_sub", "h_add", "h_mul",
                      "h_max", "h_exp", "h_div", "Scalar"):
        if not hasattr(t, attribute):
            raise FlashVisionError(
                f"the installed TIK instance has no {attribute!r}; refusing to emit a "
                "kernel whose online softmax cannot be expressed")

    q_gm = t.Tensor("float16", (batch, d1, m, 16), name="q_gm", scope=tik.scope_gm)
    k_gm = t.Tensor("float16", (batch, d1, n_pad, 16), name="k_gm", scope=tik.scope_gm)
    v_gm = t.Tensor("float16", (batch, n_pad // 16, pad, 16), name="v_gm",
                    scope=tik.scope_gm)
    aux_gm = t.Tensor("float32", (16,), name="aux_gm", scope=tik.scope_gm)
    out_gm = t.Tensor("float16", (batch, d1, m, 16), name="out_gm", scope=tik.scope_gm)

    with t.for_range(0, cores, block_num=cores) as core:
        q_l1 = t.Tensor("float16", (d1, mb, 16), name="q_l1", scope=tik.scope_cbuf)
        k_l1 = t.Tensor("float16", (d1, bc, 16), name="k_l1", scope=tik.scope_cbuf)
        v_l1 = t.Tensor("float16", (kb1, pad, 16), name="v_l1", scope=tik.scope_cbuf)
        p_hi_l1 = t.Tensor("float16", (kb1, mb, 16), name="p_hi_l1", scope=tik.scope_cbuf)
        p_lo_l1 = t.Tensor("float16", (kb1, mb, 16), name="p_lo_l1", scope=tik.scope_cbuf)
        # L0C: one FP32 score product (8x64x16 = 32 KiB) and two FP32 P@V products.
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
        u_hi = u_hi_flat.reshape((pv_groups, mb, 16))
        u_lo = u_lo_flat.reshape((pv_groups, mb, 16))
        o_f = o_f_flat.reshape((pv_groups, mb, 16))
        out_h = out_h_flat.reshape((pv_groups, mb, 16))

        # Row-broadcast scalar registers (one per Float broadcast site).
        rescale_scalar = t.Scalar("float32", name="rescale_broadcast")
        denom_scalar = t.Scalar("float32", name="denominator_broadcast")

        # Exact native Half-shift broadcast helpers (Round99-proven path).
        shift_copy = t.Tensor("float16", (256,), name="shift_copy", scope=tik.scope_ubuf)
        shift_broad = t.Tensor("float16", (1024,), name="shift_broad", scope=tik.scope_ubuf)
        shift_broad_float = t.Tensor("float32", (1024,), name="shift_broad_float",
                                     scope=tik.scope_ubuf)
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
                head_o = head * hs["out"]
                q_off = head_q + base_m * 16

                with t.new_stmt_scope(disable_sync=False):
                    t.data_move(q_l1, q_gm[q_off], 0, dma["q_load"]["nburst"],
                                dma["q_load"]["burst"], dma["q_load"]["src_stride"],
                                dma["q_load"]["dst_stride"])
                # FP32 online state initialised without a first-block branch.
                _fill_flat(t, row_prev, NEG_MAX, mb, f32_mask)
                _fill_flat(t, l_f, 0.0, mb, f32_mask)
                _fill_flat(t, o_f_flat, 0.0, pv_elems, f32_mask)

                with t.for_range(0, kv_tiles) as kv:
                    k_off = head_k + kv * bc * 16
                    v_off = head_v + kv * kb1 * pad * 16
                    # 1. K/V load plus the FP32 QK product (all of K is reloadable:
                    #    the L1 K tile is rewritten every KV iteration).
                    with t.new_stmt_scope(disable_sync=False):
                        t.data_move(k_l1, k_gm[k_off], 0, dma["k_load"]["nburst"],
                                    dma["k_load"]["burst"], dma["k_load"]["src_stride"],
                                    dma["k_load"]["dst_stride"])
                        t.data_move(v_l1, v_gm[v_off], 0, dma["v_load"]["nburst"],
                                    dma["v_load"]["burst"], dma["v_load"]["src_stride"],
                                    dma["v_load"]["dst_stride"])
                        t.matmul(score_cc, q_l1, k_l1, mb, d1 * 16, bc, init_l1out=True)
                    # 2. FP32 score tile to UB (never narrowed here).
                    with t.new_stmt_scope(disable_sync=False):
                        t.tensor_mov(s_f_flat, score_cc,
                                     dma["score_l0c_to_ub"]["block_mode"],
                                     dma["score_l0c_to_ub"]["nburst"],
                                     dma["score_l0c_to_ub"]["burst_len_kb"], 0, 0)
                    # 3. Original head-dim scale on every group.
                    for group in range(score_groups):
                        t.vec_muls(f32_mask, s_f[group, :, :], s_f[group, :, :], scale,
                                   group_repeats, f32_stride, f32_stride)
                    # 4. Phantom key slots must never reach the max or the
                    #    denominator. Zero-padded K rows give score exactly 0, and
                    #    exp(0 - shift) would add a full row of 1.0 per phantom key.
                    if phantom_groups:
                        with t.if_scope(kv == kv_tiles - 1):
                            _mask_phantom_scores(t, s_f_flat, plan, NEG_MAX)
                    # 5. Half row shift estimate: explicit pairwise vec_max folds
                    #    (high-level h_max rejected the sliced NZ step on this CANN).
                    t.h_cast(p_hi_h_flat, s_f_flat, "none")
                    _fold_half_max(t, fold_h_a, fold_h_b, p_hi_h_flat, score_groups, mb)
                    t.h_reduce_max(m_half, fold_h_a, 1)
                    t.h_cast(row_new, m_half, "none")
                    # 6. Online update: new = max(prev, row); rescale = exp(prev-new).
                    t.h_max(row_new, row_prev, row_new)
                    t.h_sub(row_diff, row_prev, row_new)
                    t.h_exp(row_resc, row_diff)
                    t.h_add(row_prev, row_new, row_zero)
                    # 7. exp(S - shift) in FP32. The shift is exactly Half
                    #    representable, so it is broadcast with native bit copies
                    #    into the widened-P_hi buffer, which is dead until step 10.
                    t.h_cast(m_half, row_new, "none")
                    _broadcast_half_shift(t, p_hi_f_flat, m_half, shift_copy,
                                          shift_broad, shift_broad_float,
                                          score_groups, mb)
                    t.h_sub(s_f, s_f, p_hi_f)
                    t.h_exp(s_f, s_f)
                    # 8. Explicitly zero the phantom groups after exp and before the
                    #    true Float denominator (belt and braces on top of step 4).
                    if phantom_groups:
                        with t.if_scope(kv == kv_tiles - 1):
                            _mask_phantom_scores(t, s_f_flat, plan, 0.0)
                    # 9. TRUE FP32 denominator of this block over all 8 groups.
                    _fold_float_sum(t, fold_f_a, fold_f_b, s_f_flat, score_groups, mb,
                                    f32_mask, f32_stride, group_repeats)
                    t.h_reduce_sum(row_sum, fold_f_a, 1)
                    # 10. L = rescale * L + block_sum (FP32 per row).
                    t.h_mul(row_diff, l_f, row_resc)
                    t.h_add(l_f, row_diff, row_sum)
                    # 11. Two-Half probability split with the low residual written
                    #     over the score buffer in place (no extra Float tile). The
                    #     next KV iteration reloads the scores from the QK product.
                    t.h_cast(p_hi_h_flat, s_f_flat, "none")
                    t.h_cast(p_hi_f_flat, p_hi_h_flat, "none")
                    t.h_sub(s_f, s_f, p_hi_f)
                    for group in range(score_groups):
                        t.vec_muls(f32_mask, s_f[group, :, :], s_f[group, :, :],
                                   RESIDUAL_SCALE, group_repeats, f32_stride, f32_stride)
                    t.h_cast(p_lo_h_flat, s_f_flat, "none")
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
                        t.tensor_mov(u_lo_flat, pv_lo_cc,
                                     dma["pv_l0c_to_ub"]["block_mode"],
                                     dma["pv_l0c_to_ub"]["nburst"],
                                     dma["pv_l0c_to_ub"]["burst_len_kb"], 0, 0)
                        t.tensor_mov(u_hi_flat, pv_hi_cc,
                                     dma["pv_l0c_to_ub"]["block_mode"],
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
                t.data_move(out_gm[head_o + base_m * 16], out_h_flat[0], 0,
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
        Path.cwd() / "flash_vision101_kernels"
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec101.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability)

    with (out_root / ".compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan,
                                    capability)
        work = Path(tempfile.mkdtemp(prefix="flash101_build_", dir=out_root)).resolve()
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
        "round": 101,
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
        "phantom_key_tail_excluded_from_denominator": True,
        "no_separate_low_residual_float_tile": True,
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