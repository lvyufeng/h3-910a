#!/usr/bin/env python3
"""Round97: Float-NZ score row -> true-4032 unmasked Float softmax -> Half hi/lo NZ.

STATUS: SOURCE ONLY. Never executed/compiled by its author against CANN, TIK,
torch, torch_npu, a model or the NPU: no SSH, no build, no launch, no install,
no production edit, no commit. Every number this module reports belongs to the
root agent that compiles and measures it. This file claims no accuracy result, no
speed result and no lossless pass.

What the single kernel does (one launch, five GM pointers)
----------------------------------------------------------
For one head and one query row it replaces the whole Python/Torch stage
"unpack FP32 scores -> pad -> softmax -> build hi + low Half probability packs":

1. ``data_move`` the row out of the NZ FP32 score tensor. The score operand is
   exactly the same layout the PV stage consumes as ``A_NZ``:
   ``(head, n1=256, m_pad, 16)`` FP32, so one query row is 256 blocks of 16 FP32
   values = 4096 keys.
2. ``vec_muls`` by the original vision head scale ``72 ** -0.5`` (NOT ``128**-0.5``:
   the padded head width is not the model scale).
3. Exclude the 64 padded key slots by writing ``-FLT_MAX`` over the tail, so the
   tail can never win ``vec_reduce_max`` nor contribute after ``vec_exp``.
4. Stable softmax in Float32: ``vec_reduce_max`` -> subtract -> ``vec_exp``.
5. Explicitly zero the tail again *before* the denominator, so the sum is over
   exactly the 4032 real keys even if a hardware ``vexp`` produced anything odd
   for the extreme masked input.
6. ``vec_reduce_add`` the denominator, then ``vec_rec`` + two Newton steps for
   ``1/sum`` and ``vec_muls`` the probabilities.
7. Split each probability into ``hi = Half(p)`` and
   ``lo = Half((p - float(hi)) * 1024)`` with one ``h_cast`` each, and ``data_move``
   both into the Half NZ tensors ``(head, n1, m_pad, 16)``.
8. Rows at or beyond the real query count ``m`` (the 512-448 endpoint pad) write
   **all-zero** hi/lo for the whole row. They must never become a uniform
   ``1/4096`` softmax row, because the PV stage reads every padded row.

Why the layout matches: with ``n1=256`` and block of 16 the row read/write uses
``nburst=256``, ``burst=2`` 32 B units for FP32 (16*4 B) with
``src_stride=2*(m_pad-1)``, and ``burst=1`` unit for Half (16*2 B) with
``dst_stride=m_pad-1``. The FP32 factor of two is the byte width, not an element
index: using ``2*(m_pad-1)`` for the Half store would corrupt every second block.

Honest limits
-------------
* This is a *probability formula* implementation, not a bit-identical replay of
  any Torch softmax: the Float reduction order (``vcadd``/``vcmax``) and the
  reciprocal are hardware orders and can change low bits. The authority for
  numerics is the root's frozen-lever CPU Half MATH comparison.
* No Float linear weight, no BF16 gold cast, no teacher tensor, and no cached
  pixel/patch/hidden/DeepStack/conditioning value is added here. The two
  auxiliary Float GM operands are unused placeholders that keep the proven
  five-pointer ABI visible to the compiler.
* ``vec_reduce_max``/``vec_reduce_add`` are gated by
  ``platform.intrinsic_check_support``: an unsupported Float reduce fails closed
  instead of silently emitting an A2-only instruction.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

# --- fixed vision-attention geometry ---------------------------------------
BATCH = 16                 # vision heads
M_PAD = 512                # padded query rows in GM
M_CHOICES = (512, 448)     # real query rows (448 pads 64 rows to zero)
N_KEYS = 4096              # padded key slots
VALID_KEYS = 4032          # real key slots: one image's 48x84 patch grid,
                           # attention runs BEFORE the 2x2 merger
TAIL_KEYS = N_KEYS - VALID_KEYS
HEAD_DIM = 72              # original head width; NOT the padded 128
N1 = N_KEYS // 16          # 256 NZ blocks per row
BLOCK = 16                 # NZ block of 16 values

CORE_NUM = 30              # short Ascend910 TIK SoC: 30 usable AI cores
PINNED_COMPILE_SOC = "Ascend910"
PINNED_UB_SAFE_BYTES = 196608

FP32_MASK = 64             # one Ascend repeat of FP32
FP32_REP_STRIDE = 8        # 8 x 32 B blocks = 256 B = 64 FP32 elements
SCALAR_MASK = 8            # the 8-slot rows/scalar scratch tensors are 8 wide
REPEATS = N_KEYS // FP32_MASK            # 64 repeats cover the row
REDUCE_WORK = REPEATS                    # vreduce work tensor slots
NEG_MAX = -3.4028234663852886e+38        # -FLT_MAX: finite, never a NaN
LO_SCALE = 1024.0                        # lo = (p - float(hi)) * 1024
NEWTON_STEPS = 2

DEFAULT_OUT_DIR = Path.cwd() / "softmax_pack97_kernels"

MATH_STEPS = ("load_score_row", "scale_head_dim", "mask_key_tail", "reduce_row_max",
              "subtract_max", "exp_row", "zero_key_tail", "reduce_row_sum",
              "reciprocal_newton", "normalise", "half_hi", "widen_hi",
              "residual_scaled", "half_lo", "store_hi", "store_lo", "zero_pad_rows")


class SoftmaxPackError(RuntimeError):
    """A refusal: geometry, budget, capability or build contract violated."""


# ---------------------------------------------------------------------------
# Planning (pure python, no CANN/torch import: the stdlib tests import this)
# ---------------------------------------------------------------------------


def score_scale() -> float:
    """Original vision attention scale ``head_dim ** -0.5`` for the real 72."""
    return float(HEAD_DIM) ** -0.5


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SoftmaxPackError(f"{name} must be a python int, got {type(value).__name__}")
    if value <= 0:
        raise SoftmaxPackError(f"{name} must be positive, got {value}")
    return value


def plan_kernel(batch: int = BATCH, m: int = M_PAD, n: int = N_KEYS,
                valid_keys: int = VALID_KEYS, cores: int = CORE_NUM) -> dict:
    """Validate the geometry and derive every offset/DMA/buffer constant."""
    batch = _positive_int(batch, "batch")
    m = _positive_int(m, "m")
    n = _positive_int(n, "n")
    valid_keys = _positive_int(valid_keys, "valid_keys")
    cores = _positive_int(cores, "cores")

    if m not in M_CHOICES:
        raise SoftmaxPackError(f"m must be one of {M_CHOICES}, got {m}")
    if m > M_PAD:
        raise SoftmaxPackError(f"m {m} exceeds the padded query rows {M_PAD}")
    if n != N_KEYS:
        raise SoftmaxPackError(f"n must stay the padded NZ key count {N_KEYS}, got {n}")
    if valid_keys != VALID_KEYS:
        raise SoftmaxPackError(
            f"valid_keys must stay {VALID_KEYS} (N - {TAIL_KEYS} tail), got {valid_keys}")
    if n % FP32_MASK or valid_keys % FP32_MASK:
        raise SoftmaxPackError("key count and valid keys must be whole FP32 repeats")
    if TAIL_KEYS != FP32_MASK:
        raise SoftmaxPackError("the key tail must be exactly one FP32 repeat block")
    if n % BLOCK:
        raise SoftmaxPackError("key count is not a whole NZ block of 16")

    zero_rows = M_PAD - m
    tasks = batch * M_PAD
    steps = (tasks + cores - 1) // cores
    half_row_elems = N1 * M_PAD * BLOCK

    ub_bytes = {
        "s_f": n * 4, "aux_f": n * 4, "zero_f": n * 4,
        "hi_h": n * 2, "lo_h": n * 2,
        "max_t": 8 * 4, "sum_t": 8 * 4,
        "max_work": REDUCE_WORK * 4, "sum_work": REDUCE_WORK * 4,
        "den_f": 8 * 4, "inv_f": 8 * 4, "newton_f": 8 * 4, "neg_f": 8 * 4,
    }
    ub_total = sum(ub_bytes.values())

    plan = {
        "logical": {"batch": batch, "m": m, "n": n, "valid_keys": valid_keys,
                    "head_dim": HEAD_DIM, "score_scale": score_scale(),
                    "lo_scale": LO_SCALE},
        "padded": {"m_pad": M_PAD, "zero_rows": zero_rows, "n1": N1, "block": BLOCK,
                   "tail_keys": TAIL_KEYS, "n1_blocks_in_row": N1},
        "grid": {"tasks": tasks, "cores": cores, "steps": steps,
                 "last_step_live": tasks - (steps - 1) * cores,
                 "per_head_rows": M_PAD},
        "head_stride_elems": {"score_f32": half_row_elems, "hi_half": half_row_elems,
                             "lo_half": half_row_elems},
        "gm_shapes": {"score_f32": [batch, N1, M_PAD, BLOCK],
                      "aux0_f32": [BLOCK], "aux1_f32": [BLOCK],
                      "hi_half": [batch, N1, M_PAD, BLOCK],
                      "lo_half": [batch, N1, M_PAD, BLOCK]},
        "gm_bytes": {"score_f32": batch * half_row_elems * 4,
                     "aux0_f32": BLOCK * 4, "aux1_f32": BLOCK * 4,
                     "hi_half": batch * half_row_elems * 2,
                     "lo_half": batch * half_row_elems * 2},
        "dma": {
            # 16 FP32 = 64 B = 2 x 32 B units; next n1 block is m_pad*2 units away.
            "score_load": {"nburst": N1, "burst": 2,
                           "src_stride": 2 * (M_PAD - 1), "dst_stride": 0},
            # 16 Half = 32 B = 1 unit; next n1 block is m_pad units away.
            "prob_store": {"nburst": N1, "burst": 1,
                           "src_stride": 0, "dst_stride": M_PAD - 1},
        },
        "vector": {"mask": FP32_MASK, "repeats": REPEATS,
                   "rep_stride": FP32_REP_STRIDE, "reduce_work": REDUCE_WORK,
                   "newton_steps": NEWTON_STEPS,
                   "masked_tail_keys": TAIL_KEYS,
                   "negative_max": NEG_MAX},
        "buffers": {**ub_bytes, "ub_total_bytes": ub_total,
                    "ub_safe_bytes": PINNED_UB_SAFE_BYTES},
        "kernel_arg_order": ["scores_gm_float32", "aux0_gm_float32",
                             "aux1_gm_float32", "hi_gm_float16", "lo_gm_float16"],
        "math_steps": list(MATH_STEPS),
        "limits": {"cores": CORE_NUM, "compile_soc": PINNED_COMPILE_SOC,
                   "mask_repeat_max": 255, "vreduce_repeat_max": 4095},
    }
    if ub_total > PINNED_UB_SAFE_BYTES:
        raise SoftmaxPackError(
            f"UB need {ub_total} B exceeds the {PINNED_UB_SAFE_BYTES} B safe bound")
    if REPEATS > 255:
        raise SoftmaxPackError(f"{REPEATS} repeats exceed the 255 vector repeat limit")
    if REPEATS > 4095:
        raise SoftmaxPackError("repeats exceed the vreduce repeat limit")
    return plan


def flatten_task(task: int, plan: dict) -> tuple:
    """``task -> (head, row)``; ``tasks == batch*m_pad`` so no head is skipped twice."""
    total = plan["grid"]["tasks"]
    if not isinstance(task, int) or isinstance(task, bool):
        raise SoftmaxPackError("task must be a python int")
    if not 0 <= task < total:
        raise SoftmaxPackError(f"task {task} outside [0, {total})")
    head, row = divmod(task, M_PAD)
    return head, row


def row_offsets(plan: dict, head: int, row: int) -> dict:
    """Absolute GM element offsets of one (head, row) score and prob row."""
    if not 0 <= head < plan["logical"]["batch"]:
        raise SoftmaxPackError(f"head {head} outside batch")
    if not 0 <= row < M_PAD:
        raise SoftmaxPackError(f"row {row} outside padded rows")
    hs = plan["head_stride_elems"]
    return {"score_f32": head * hs["score_f32"] + row * BLOCK,
            "hi_half": head * hs["hi_half"] + row * BLOCK,
            "lo_half": head * hs["lo_half"] + row * BLOCK}


def is_padded_row(plan: dict, row: int) -> bool:
    """True when this row must be written as all zeros instead of a softmax."""
    if not 0 <= row < M_PAD:
        raise SoftmaxPackError(f"row {row} outside padded rows")
    return row >= plan["logical"]["m"]


def row_element_coverage(plan: dict) -> dict:
    """Element reach of the DMA descriptors, checked by the stdlib tests."""
    load, store = plan["dma"]["score_load"], plan["dma"]["prob_store"]
    f32_units, half_units = M_PAD * BLOCK * 4 // 32, M_PAD * BLOCK * 2 // 32
    return {
        "load_units_per_n1_step": load["burst"] + load["src_stride"],
        "expected_row_units_f32": f32_units,
        "store_units_per_n1_step": store["burst"] + store["dst_stride"],
        "expected_row_units_half": half_units,
        "load_covers_row": load["burst"] + load["src_stride"] == f32_units,
        "store_covers_row": store["burst"] + store["dst_stride"] == half_units,
        "load_elements": load["nburst"] * load["burst"] * 8,
        "store_elements": store["nburst"] * store["burst"] * 16,
    }


# ---------------------------------------------------------------------------
# Capability gate (lazy CANN import; nothing is compiled here)
# ---------------------------------------------------------------------------

REDUCE_INTRINSICS = ("Intrinsic_vmax", "Intrinsic_vcadd")
DTYPE_SPELLINGS = ("float32", "f32")


def capability_report() -> dict:
    """Resolve the compile SoC and refuse when Float vreduce is unsupported."""
    import tbe.common.platform as platform

    platform.set_current_compile_soc_info(PINNED_COMPILE_SOC)
    report = {"compile_soc": None, "core_num": None, "ub_size": None,
              "intrinsic_checks": {}, "float_vreduce_supported": None,
              "float_vreduce_query_answered": False}
    try:
        report["compile_soc"] = platform.get_soc_spec("SHORT_SOC_VERSION")
    except Exception:                                    # noqa: BLE001 - older SDK
        report["compile_soc"] = None
    if report["compile_soc"] not in (None, PINNED_COMPILE_SOC):
        raise SoftmaxPackError(
            f"compile SoC resolved to {report['compile_soc']!r}, expected "
            f"{PINNED_COMPILE_SOC!r}; refusing to emit a different ISA")
    for key in ("CORE_NUM", "UB_SIZE"):
        try:
            report[key.lower()] = int(platform.get_soc_spec(key))
        except Exception:                                # noqa: BLE001
            report[key.lower()] = None
    if report["core_num"] is not None and report["core_num"] < CORE_NUM:
        raise SoftmaxPackError(
            f"compiler CORE_NUM is {report['core_num']}, below the {CORE_NUM} launched cores")
    report['launched_usable_cores'] = CORE_NUM
    report['physical_core_spec_may_be_32'] = True
    if report["ub_size"] is not None and report["ub_size"] < PINNED_UB_SAFE_BYTES:
        raise SoftmaxPackError(f"UB_SIZE {report['ub_size']} below the pinned bound")

    answered, supported = False, {}
    for intrinsic in REDUCE_INTRINSICS:
        for spelling in DTYPE_SPELLINGS:
            try:
                value = bool(platform.intrinsic_check_support(intrinsic, spelling))
                answered = True
            except Exception:                            # noqa: BLE001 - no such API
                value = None
            report["intrinsic_checks"][f"{intrinsic}:{spelling}"] = value
        supported[intrinsic] = any(
            report["intrinsic_checks"].get(f"{intrinsic}:{spelling}") is True
            for spelling in DTYPE_SPELLINGS)
    report["float_vreduce_query_answered"] = answered
    if answered:
        missing = [name for name, ok in supported.items() if not ok]
        report["float_vreduce_supported"] = not missing
        if missing:
            raise SoftmaxPackError(
                "Float32 vector reduce is not supported by this compile target: "
                f"{missing}. The stable softmax needs Float vcmax/vcadd; refusing to "
                "substitute a non-Float reduction or an A2-only instruction. Report "
                f"the exact checks: {json.dumps(report['intrinsic_checks'])}")
    return report


# ---------------------------------------------------------------------------
# Kernel build (cached, locked, never overwriting a previous directory)
# ---------------------------------------------------------------------------


def kernel_name_for(plan: dict) -> str:
    lg, gr = plan["logical"], plan["grid"]
    return ("softmax_pack97_h%d_m%d_n%d_vk%d_c%d"
            % (lg["batch"], lg["m"], lg["n"], lg["valid_keys"], gr["cores"]))


def _emit_kernel(t, plan: dict, tik) -> dict:
    """Emit the reviewed instruction sequence for one head/row per task."""
    for attribute in ("vec_max", "vec_reduce_add", "vec_rec", "vec_exp",
                      "vec_adds", "vec_muls", "vec_add", "vec_mul", "h_cast",
                      "vector_dup", "data_move"):
        if not hasattr(t, attribute):
            raise SoftmaxPackError(
                f"the installed TIK instance has no {attribute!r}; refusing to emit a "
                "kernel whose softmax reduction cannot be expressed")

    lg, pd, gr, dma = plan["logical"], plan["padded"], plan["grid"], plan["dma"]
    vec = plan["vector"]
    batch, m, m_pad = lg["batch"], lg["m"], pd["m_pad"]
    mask, repeats, stride = vec["mask"], vec["repeats"], vec["rep_stride"]
    work_slots, valid_keys = vec["reduce_work"], lg["valid_keys"]
    scale, lo_scale = lg["score_scale"], lg["lo_scale"]
    load, store = dma["score_load"], dma["prob_store"]
    cores, steps, tasks = gr["cores"], gr["steps"], gr["tasks"]

    score_gm = t.Tensor("float32", (batch, N1, m_pad, BLOCK), name="scores_gm",
                        scope=tik.scope_gm)
    aux0_gm = t.Tensor("float32", (BLOCK,), name="aux0_gm", scope=tik.scope_gm)
    aux1_gm = t.Tensor("float32", (BLOCK,), name="aux1_gm", scope=tik.scope_gm)
    hi_gm = t.Tensor("float16", (batch, N1, m_pad, BLOCK), name="hi_gm",
                     scope=tik.scope_gm)
    lo_gm = t.Tensor("float16", (batch, N1, m_pad, BLOCK), name="lo_gm",
                     scope=tik.scope_gm)

    with t.for_range(0, cores, block_num=cores) as core:
        ubuf = tik.scope_ubuf
        s_f = t.Tensor("float32", (N_KEYS,), name="s_f", scope=ubuf)
        aux_f = t.Tensor("float32", (N_KEYS,), name="aux_f", scope=ubuf)
        zero_f = t.Tensor("float32", (N_KEYS,), name="zero_f", scope=ubuf)
        hi_h = t.Tensor("float16", (N_KEYS,), name="hi_h", scope=ubuf)
        lo_h = t.Tensor("float16", (N_KEYS,), name="lo_h", scope=ubuf)
        max_t = t.Tensor("float32", (8,), name="max_t", scope=ubuf)
        sum_t = t.Tensor("float32", (8,), name="sum_t", scope=ubuf)
        max_work = t.Tensor("float32", (work_slots,), name="max_work", scope=ubuf)
        sum_work = t.Tensor("float32", (work_slots,), name="sum_work", scope=ubuf)
        den_f = t.Tensor("float32", (8,), name="den_f", scope=ubuf)
        inv_f = t.Tensor("float32", (8,), name="inv_f", scope=ubuf)
        newton_f = t.Tensor("float32", (8,), name="newton_f", scope=ubuf)
        neg_f = t.Tensor("float32", (8,), name="neg_f", scope=ubuf)

        # Constant zero row for the query-row endpoint pad, filled once outside the
        # task loop with the verified FP32 repeat stride convention.
        t.vector_dup(mask, zero_f, 0.0, repeats, 1, stride)

        with t.for_range(0, steps) as step:
            task = core + step * cores
            with t.if_scope(task < tasks):
                head = task // m_pad
                row = task - head * m_pad
                score_off = head * plan["head_stride_elems"]["score_f32"] + row * BLOCK
                hi_off = head * plan["head_stride_elems"]["hi_half"] + row * BLOCK
                lo_off = head * plan["head_stride_elems"]["lo_half"] + row * BLOCK
                # 1-2: row out of the NZ FP32 operand, then the real head scale.
                t.data_move(s_f, score_gm[score_off], 0, load["nburst"],
                            load["burst"], load["src_stride"], load["dst_stride"])
                t.vec_muls(mask, s_f, s_f, scale, repeats, stride, stride)
                # 3: the 64 padded key slots can never win max or survive exp.
                t.vector_dup(mask, s_f[valid_keys], vec["negative_max"], 1, 1, stride)
                # Original910 lacks Float vcmax, but supports Float vmax.
                # Exact Float maximum: fold 64 groups into a 64-value vector,
                # then halve it; short peers use aligned scalar UB copies.
                t.data_move(max_work, s_f, 0, 1, 8, 0, 0)
                with t.for_range(1, 64) as group:
                    t.vec_max(mask, max_work, max_work, s_f[group * 64], 1, stride, stride, stride)
                for half in (32, 16, 8):
                    t.data_move(aux_f, max_work[half], 0, 1, half // 8, 0, 0)
                    t.vec_max(half, max_work, max_work, aux_f, 1, stride, stride, stride)
                for half in (4, 2, 1):
                    for item in range(half):
                        aux_f[item].set_as(max_work[half + item])
                    t.vec_max(half, max_work, max_work, aux_f, 1, stride, stride, stride)
                max_scalar = t.Scalar("float32", name="row_max")
                max_scalar.set_as(max_work[0])
                # The 8-slot scratch tensors are 8 wide: mask them with 8, never
                # the 64-wide row mask, or the op would address past the buffer.
                t.vector_dup(SCALAR_MASK, neg_f, max_scalar, 1, 1, stride)
                t.vec_muls(SCALAR_MASK, neg_f, neg_f, -1.0, 1, 1, 1)
                neg_scalar = t.Scalar("float32", name="row_neg_max")
                neg_scalar.set_as(neg_f[0])
                t.vec_adds(mask, s_f, s_f, neg_scalar, repeats, stride, stride)
                t.vec_exp(mask, s_f, s_f, repeats, stride, stride)
                # 5: explicit tail zero before the denominator: the 4032 real keys
                # alone define the row sum regardless of the extreme exp input.
                t.vector_dup(mask, s_f[valid_keys], 0.0, 1, 1, stride)
                # 6: denominator, then 1/sum with a Float reciprocal + 2 Newton.
                t.vec_reduce_add(mask, sum_t, s_f, sum_work, repeats, stride)
                sum_scalar = t.Scalar("float32", name="row_sum")
                sum_scalar.set_as(sum_t[0])
                t.vector_dup(SCALAR_MASK, den_f, sum_scalar, 1, 1, stride)
                t.vec_rec(SCALAR_MASK, inv_f, den_f, 1, 1, 1)
                for _ in range(NEWTON_STEPS):
                    t.vec_mul(SCALAR_MASK, newton_f, den_f, inv_f, 1, 1, 1, 1)
                    t.vec_muls(SCALAR_MASK, newton_f, newton_f, -1.0, 1, 1, 1)
                    t.vec_adds(SCALAR_MASK, newton_f, newton_f, 2.0, 1, 1, 1)
                    t.vec_mul(SCALAR_MASK, inv_f, inv_f, newton_f, 1, 1, 1, 1)
                inv_scalar = t.Scalar("float32", name="row_inv_sum")
                inv_scalar.set_as(inv_f[0])
                t.vec_muls(mask, s_f, s_f, inv_scalar, repeats, stride, stride)
                # Avoid the CANN SPR mask-merge bug for vector/cast branches.
                # Only the scalar valid-row flag is conditional; all row vector
                # instructions are identical and padded-row probabilities become zero.
                valid_row = t.Scalar("float32", name="valid_row")
                valid_row.set_as(1.0)
                with t.if_scope(row >= m):
                    valid_row.set_as(0.0)
                t.vec_muls(mask, s_f, s_f, valid_row, repeats, stride, stride)
                # 7: hi = Half(p); lo = Half((p - float(hi)) * 1024).
                t.h_cast(hi_h, s_f, "none")
                t.h_cast(aux_f, hi_h, "none")
                t.vec_muls(mask, aux_f, aux_f, -1.0, repeats, stride, stride)
                t.vec_add(mask, aux_f, s_f, aux_f, repeats, stride, stride, stride)
                t.vec_muls(mask, aux_f, aux_f, lo_scale, repeats, stride, stride)
                t.h_cast(lo_h, aux_f, "none")
                t.data_move(hi_gm[hi_off], hi_h, 0, store["nburst"], store["burst"],
                            store["src_stride"], store["dst_stride"])
                t.data_move(lo_gm[lo_off], lo_h, 0, store["nburst"], store["burst"],
                            store["src_stride"], store["dst_stride"])
    return {"scores": score_gm, "aux0": aux0_gm, "aux1": aux1_gm,
            "hi": hi_gm, "lo": lo_gm}


def _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                     compiled_now=False) -> dict:
    meta = json.loads(meta_path.read_text())
    if int(meta.get("blockDim", -1)) != plan["grid"]["cores"]:
        raise SoftmaxPackError(
            f"cached kernel {name} has blockDim {meta.get('blockDim')} but the plan needs "
            f"{plan['grid']['cores']} cores; refusing a mismatched binary")
    obj_sha = hashlib.sha256(obj.read_bytes()).hexdigest()
    coverage = row_element_coverage(plan)
    if not coverage["load_covers_row"] or not coverage["store_covers_row"]:
        raise SoftmaxPackError(f"DMA descriptors do not tile the row: {coverage}")
    spec = {
        "round": 97,
        "scope": "Float32 NZ score row -> true-4032 unmasked Float softmax -> "
                 "Half hi + Half lo NZ probability packing, one launch",
        "executed": False,
        "compiled_now": compiled_now,
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "numerical_tolerance": None,
        "kernel_name": name,
        "kernel_symbol": meta["kernelName"],
        "obj_block": int(meta["blockDim"]),
        "block_dim": int(meta["blockDim"]),
        "core_type": meta.get("coreType"),
        "obj_path": str(obj),
        "obj_sha256": obj_sha,
        "json_path": str(meta_path),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "GM_argument_count": 5,
        "kernel_arg_order": plan["kernel_arg_order"],
        "ABI": "five GM pointers, args retained by the caller",
        "input_dtype": "float32",
        "output_dtype": "float16",
        "output_hi_lo": True,
        "lo_rescale_inverse": plan["logical"]["lo_scale"],
        "score_scale": plan["logical"]["score_scale"],
        "score_scale_basis": "head_dim 72 ** -0.5 (original; not padded 128)",
        "reduction": {
            "order": "hardware vcmax + vcadd; not a Torch replay",
            "float_reduce_gated_by_intrinsic_check": True,
            "newton_steps_for_reciprocal": NEWTON_STEPS,
            "bit_identity_with_torch_softmax_claimed": False,
            "vreduce_signature_assumption":
                "vec_reduce_max/vec_reduce_add(mask, dst, src, work, repeat_times, "
                "src_rep_stride) with one call over all 64 repeats leaving the row "
                "result in dst[0]; confirmed for vec_reduce_add by the executed "
                "Round61/62 AdaLN norm, assumed identical for vec_reduce_max "
                "(same vreduce family, vcmax registered). Root build/probe is the "
                "authority; the author never compiled this kernel.",
            "partial_ub_slice_operands":
                "s_f[valid_keys] is used as a vector_dup destination so only the "
                "64-slot key tail is written; the project FA patch already passes "
                "offset tensor slices to vector_dup (scores[block, 0, 0]).",
        },
        "masking": {
            "motto": "true 4032 keys only, unmasked (no causal)",
            "tail_keys_zeroed": TAIL_KEYS,
            "tail_masked_before_max_and_exp": True,
            "tail_explicitly_zeroed_before_denominator": True,
            "negative_max_finite": True,
            "padded_query_rows_forced_zero": True,
            "padded_rows_uniform_softmax_forbidden": True,
        },
        "logical": plan["logical"],
        "padded": plan["padded"],
        "grid": plan["grid"],
        "head_stride_elems": plan["head_stride_elems"],
        "gm_shapes": plan["gm_shapes"],
        "gm_bytes": plan["gm_bytes"],
        "dma": plan["dma"],
        "dma_row_coverage": coverage,
        "vector": plan["vector"],
        "buffers": plan["buffers"],
        "math_steps": plan["math_steps"],
        "capability": capability,
        "limits": plan["limits"],
        "no_float_linear_weight": True,
        "no_teacher_or_bf16_gold_cast": True,
        "caches_no_pixel_patch_hidden_deepstack_or_conditioning": True,
        "aux_gm_operands_are_unused_placeholders": True,
        "layout_equals_pv_a_nz": True,
        "a2_only_features_used": False,
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec


def build_kernel(batch: int = BATCH, m: int = M_PAD, n: int = N_KEYS,
                 valid_keys: int = VALID_KEYS, out_dir=None) -> dict:
    """Build (or reuse) one softmax+probability-packing kernel and describe it."""
    plan = plan_kernel(batch=batch, m=m, n=n, valid_keys=valid_keys)
    from tbe import tik

    capability = capability_report()
    name = kernel_name_for(plan)
    out_root = Path(out_dir).resolve() if out_dir is not None else DEFAULT_OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / name
    obj = dest / (name + ".o")
    meta_path = dest / (name + ".json")
    spec_path = dest / "spec97.json"

    if obj.is_file() and meta_path.is_file():
        return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                                compiled_now=False)
    with (out_root / ".softmax_pack97.compile.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if obj.is_file() and meta_path.is_file():
            return _describe_cached(name, dest, obj, meta_path, spec_path, plan,
                                    capability, compiled_now=False)
        work = Path(tempfile.mkdtemp(prefix="softmax_pack97_build_", dir=out_root)).resolve()
        if work.parent != out_root or not work.name.startswith("softmax_pack97_build_"):
            raise SoftmaxPackError("unexpected build workspace")
        t = tik.Tik(disable_debug=True)
        gm = _emit_kernel(t, plan, tik)
        previous = os.getcwd()
        try:
            os.chdir(work)
            # All five GM operands are passed explicitly: the two Float auxiliaries
            # exist only to keep the proven five-pointer ABI and automatic
            # collection would prune them.
            t.BuildCCE(kernel_name=name,
                       inputs=[gm["scores"], gm["aux0"], gm["aux1"]],
                       outputs=[gm["hi"], gm["lo"]])
        finally:
            os.chdir(previous)
        built_obj = work / "kernel_meta" / (name + ".o")
        built_meta = work / "kernel_meta" / (name + ".json")
        if not built_obj.is_file() or not built_meta.is_file():
            raise SoftmaxPackError(
                f"BuildCCE produced no kernel_meta artefacts for {name}; the workspace "
                f"{work} is retained for diagnosis")
        if dest.exists():
            raise SoftmaxPackError(f"refusing to overwrite existing kernel directory {dest}")
        dest.mkdir(parents=True)
        import shutil

        shutil.copyfile(built_obj, obj)
        shutil.copyfile(built_meta, meta_path)
        shutil.rmtree(work, ignore_errors=True)
    return _describe_cached(name, dest, obj, meta_path, spec_path, plan, capability,
                            compiled_now=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the R97 softmax+pack kernel.")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--m", type=int, default=M_PAD)
    parser.add_argument("--out-dir", default=None)
    arguments = parser.parse_args()
    print(json.dumps(build_kernel(batch=arguments.batch, m=arguments.m,
                                  out_dir=arguments.out_dir), indent=2))