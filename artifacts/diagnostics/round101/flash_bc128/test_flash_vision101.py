#!/usr/bin/env python3
"""Stdlib-only contract tests for artifacts/diagnostics/round101/flash_bc128.

NO CANN, TIK, torch, torch_npu, NPU, GPU or model is imported or executed here.
Everything below is pure python: geometry/budget/DMA contract assertions, an
AST-level source discipline check, and a *recording TIK stub* that runs the real
``_emit_kernel`` and inspects the emitted instruction sequence (ABI order, static
instruction counts, loop nesting, phantom-mask placement, absence of the Round99
high-level sliced ``h_max``/large strided ``h_cast`` patterns).

These tests are structural only. They are not an accuracy result, not a speed
result, not a compiler result and not a tolerance of any kind; the root driver
compiles this builder and owns every real number.
"""
import ast
import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "flash_vision101.py"
_spec = importlib.util.spec_from_file_location("flash_vision101_under_test", SOURCE_PATH)
flash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flash)
TEXT = SOURCE_PATH.read_text()


# ---------------------------------------------------------------------------
# Symbolic integers: the emitter builds offsets like head*stride + kv*bc*16.
# ---------------------------------------------------------------------------


def _e(value):
    if isinstance(value, Sym):
        return value.expr
    if isinstance(value, StubHandle):
        return "handle(%s[%s])" % (value.tensor.name, _e(value.index))
    return repr(value)


class Sym:
    """A symbolic integer used as a dynamic loop counter."""

    __slots__ = ("expr",)

    def __init__(self, expr):
        self.expr = expr

    def __add__(self, other):
        return Sym("(%s+%s)" % (self.expr, _e(other)))

    def __radd__(self, other):
        return Sym("(%s+%s)" % (_e(other), self.expr))

    def __sub__(self, other):
        return Sym("(%s-%s)" % (self.expr, _e(other)))

    def __rsub__(self, other):
        return Sym("(%s-%s)" % (_e(other), self.expr))

    def __mul__(self, other):
        return Sym("(%s*%s)" % (self.expr, _e(other)))

    def __rmul__(self, other):
        return Sym("(%s*%s)" % (_e(other), self.expr))

    def __floordiv__(self, other):
        return Sym("(%s//%s)" % (self.expr, _e(other)))

    def __eq__(self, other):
        return Sym("(%s==%s)" % (self.expr, _e(other)))

    def __hash__(self):
        return object.__hash__(self)

    def __lt__(self, other):
        return Sym("(%s<%s)" % (self.expr, _e(other)))

    def __repr__(self):
        return "Sym(%s)" % self.expr


# ---------------------------------------------------------------------------
# Recording TIK stub
# ---------------------------------------------------------------------------


class StubHandle:
    """An indexed slice/element of a stub tensor."""

    __slots__ = ("tensor", "index")

    def __init__(self, tensor, index):
        self.tensor = tensor
        self.index = index

    def __getitem__(self, index):
        """Fold a nested flat offset so ``dst[start][k]`` reports one offset."""
        if isinstance(self.index, int) and isinstance(index, int):
            return StubHandle(self.tensor, self.index + index)
        return StubHandle(self.tensor, "%s+%s" % (_e(self.index), _e(index)))

    def __repr__(self):
        return "handle(%s[%s])" % (self.tensor.name, _e(self.index))


class StubTensor:
    __slots__ = ("stub", "dtype", "shape", "name", "scope", "origin")

    def __init__(self, stub, dtype, shape, name="", scope="", origin=None):
        self.stub = stub
        self.dtype = dtype
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        self.name = name
        self.scope = scope
        self.origin = origin
        if stub is not None:
            stub.tensors.append(self)

    def __repr__(self):
        return self.name

    def reshape(self, shape):
        self.stub._record("reshape", tensor=self.name,
                          shape=tuple(int(x) for x in shape))
        return StubTensor(self.stub, self.dtype, shape, name=self.name + ".rs",
                          scope=self.scope, origin=self)

    def reinterpret_cast_to(self, dtype):
        self.stub._record("reinterpret", tensor=self.name, dtype=dtype)
        return StubTensor(self.stub, dtype, self.shape, name=self.name + ".re",
                          scope=self.scope, origin=self)

    def __getitem__(self, index):
        kind = "flat"
        if isinstance(index, tuple):
            kind = "nz_slice" if any(isinstance(x, slice) for x in index) else "multi"
        self.stub._record("index", tensor=self.name, kind=kind)
        return StubHandle(self, index)


class StubScalar:
    __slots__ = ("stub", "dtype", "name")

    def __init__(self, stub, dtype, name=""):
        self.stub = stub
        self.dtype = dtype
        self.name = name

    def __repr__(self):
        return self.name

    def set_as(self, value):
        self.stub._record("scalar_set", scalar=self.name,
                          from_handle=isinstance(value, StubHandle))


class _StubCtx:
    def __init__(self, stub, kind, var=None):
        self.stub = stub
        self.kind = kind
        self.var = var

    def __enter__(self):
        self.stub.stack.append(self.kind)
        return Sym(self.kind + "_v") if self.var else None

    def __exit__(self, *_exc):
        self.stub.stack.pop()
        return False


class TikStub:
    """Minimal recording stand-in for ``tbe.tik``; never computes anything."""

    scope_gm = "gm"
    scope_cbuf = "cbuf"
    scope_cc = "cc"
    scope_ubuf = "ubuf"

    def __init__(self):
        self.records = []
        self.tensors = []
        self.stack = []
        self.build_cce = None
        self.loop_counter = 0
        self.counters = {"for_range": 0, "if_scope": 0, "new_stmt_scope": 0}

    # -- recording ---------------------------------------------------------
    def _record(self, op, **kw):
        entry = {"op": op, "path": tuple(self.stack)}
        entry.update(kw)
        self.records.append(entry)
        return entry

    def ops(self, name):
        return [r for r in self.records if r["op"] == name]

    def ops_in(self, name, marker):
        return [r for r in self.ops(name) if marker in r["path"]]

    # -- tensors / scalars -------------------------------------------------
    def Tensor(self, dtype, shape, name="", scope=""):
        return StubTensor(self, dtype, shape, name=name, scope=scope)

    def Scalar(self, dtype, name="", init_value=None):
        self._record("scalar_new", scalar=name, dtype=dtype)
        return StubScalar(self, dtype, name=name)

    # -- control flow ------------------------------------------------------
    def for_range(self, begint, endt, block_num=None):
        self.counters["for_range"] += 1
        self.loop_counter += 1
        kind = "loop%d" % self.loop_counter
        self._record("for_range", begint=begint, endt=endt, block_num=block_num,
                     label=kind)
        return _StubCtx(self, kind, var=True)

    def if_scope(self, cond):
        self.counters["if_scope"] += 1
        self._record("if_scope", cond=_e(cond), cond_is_symbolic=isinstance(cond, Sym))
        return _StubCtx(self, "if")

    def new_stmt_scope(self, disable_sync=False):
        self.counters["new_stmt_scope"] += 1
        self._record("new_stmt_scope", disable_sync=disable_sync)
        return _StubCtx(self, "stmt")

    # -- instructions ------------------------------------------------------
    def _ops(self, name, args):
        self._record(name, args=[_e(a) for a in args])

    def vector_dup(self, *a):
        self._ops("vector_dup", a)

    def vec_muls(self, *a):
        self._ops("vec_muls", a)

    def vec_add(self, *a):
        self._ops("vec_add", a)

    def vec_max(self, *a):
        self._ops("vec_max", a)

    def vec_or(self, *a):
        self._ops("vec_or", a)

    def vec_trans(self, *a):
        self._ops("vec_trans", a)

    def data_move(self, *a):
        self._ops("data_move", a)

    def matmul(self, *a, **kw):
        self._record("matmul", args=[_e(x) for x in a],
                     shape=list(a[3:6]), init_l1out=kw.get("init_l1out"))

    def tensor_mov(self, *a):
        self._ops("tensor_mov", a)

    def h_cast(self, *a):
        self._ops("h_cast", a)

    def h_reduce_max(self, *a):
        self._ops("h_reduce_max", a)

    def h_reduce_sum(self, *a):
        self._ops("h_reduce_sum", a)

    def h_sub(self, *a):
        self._ops("h_sub", a)

    def h_add(self, *a):
        self._ops("h_add", a)

    def h_mul(self, *a):
        self._ops("h_mul", a)

    def h_max(self, *a):
        self._ops("h_max", a)

    def h_exp(self, *a):
        self._ops("h_exp", a)

    def h_div(self, *a):
        self._ops("h_div", a)

    def BuildCCE(self, kernel_name, inputs, outputs):
        self.build_cce = {"kernel_name": kernel_name,
                          "inputs": [x.name for x in inputs],
                          "outputs": [x.name for x in outputs]}


def emit(plan=None):
    """Run the real emitter against the recording stub."""
    plan = plan or flash.plan_kernel()
    stub = TikStub()
    gm = flash._emit_kernel(stub, plan, stub)
    return plan, stub, gm


# ===========================================================================
# Geometry
# ===========================================================================


class GeometryContracts(unittest.TestCase):
    def test_real_h3_vision_geometry(self):
        self.assertEqual((flash.HEADS, flash.SEQ, flash.HEAD_DIM), (16, 4032, 72))
        self.assertEqual((flash.PAD_DIM, flash.MB, flash.BC), (80, 64, 128))
        self.assertEqual(flash.D1, 5)
        self.assertEqual(flash.KB1, 8)
        self.assertEqual(4032, 48 * 84)

    def test_padded_key_length_and_phantom_tail(self):
        lg = flash.plan_kernel()["logical"]
        self.assertEqual(lg["n"], 4032)
        self.assertEqual(lg["n_pad"], 4096)
        self.assertEqual(lg["tail_keys"], 64)
        self.assertEqual(lg["phantom_keys"], 64)
        self.assertEqual(lg["phantom_groups"], 4)
        self.assertEqual(lg["real_groups_last"], 4)
        self.assertEqual(lg["score_groups"], 8)

    def test_query_tiles_kv_tiles_tasks_cores_steps(self):
        gr = flash.plan_kernel()["grid"]
        self.assertEqual(gr["m_tiles"], 63)
        self.assertEqual(gr["kv_tiles"], 32)
        self.assertEqual(gr["tasks"], 1008)
        self.assertEqual(gr["cores"], 30)
        self.assertEqual(gr["steps"], math.ceil(1008 / 30))
        self.assertTrue(gr["dynamic_kv_loop"])

    def test_scale_is_the_original_head_dim_not_the_padded_width(self):
        lg = flash.plan_kernel()["logical"]
        self.assertAlmostEqual(lg["scale"], 72 ** -0.5, places=15)
        self.assertNotAlmostEqual(lg["scale"], 80 ** -0.5, places=6)
        self.assertNotAlmostEqual(lg["scale"], 128 ** -0.5, places=6)
        check = flash.scale_check(flash.plan_kernel())
        self.assertTrue(check["matches_original"])
        self.assertFalse(check["padded_width_scale_used"])

    def test_attention_is_unmasked_and_not_causal(self):
        lg = flash.plan_kernel()["logical"]
        self.assertTrue(lg["unmasked"])
        self.assertFalse(lg["causal"])
        self.assertNotIn("causal_mask", TEXT)

    def test_pad_keys_helper(self):
        self.assertEqual(flash.pad_keys(4032), 4096)
        self.assertEqual(flash.pad_keys(192), 256)
        self.assertEqual(flash.pad_keys(128), 128)
        self.assertEqual(flash.pad_keys(64), 128)

    def test_refuses_key_length_not_multiple_of_64(self):
        for n in (100, 192 + 1, 4030):
            with self.assertRaises(flash.FlashVisionError):
                flash.plan_kernel(2, 128, n)

    def test_refuses_wrong_tile_sizes_and_unaligned_query(self):
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(2, 128, 256, 72, mb=64, bc=64)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(2, 127, 256)

    def test_refuses_head_dim_that_leaves_a_padded_block(self):
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(2, 128, 256, 64)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(2, 128, 256, 96)

    def test_smoke_geometry_is_accepted_and_keeps_a_half_real_last_block(self):
        h, m, n, d = flash.SMOKE_GEOMETRY
        self.assertEqual((h, m, n, d), (2, 128, 192, 72))
        plan = flash.plan_kernel(h, m, n, d)
        self.assertEqual(plan["logical"]["n_pad"], 256)
        self.assertEqual(plan["grid"]["kv_tiles"], 2)
        self.assertEqual(plan["grid"]["tasks"], 4)
        self.assertEqual(plan["grid"]["cores"], 4)
        self.assertEqual(plan["logical"]["phantom_keys"], 64)
        self.assertEqual(plan["logical"]["phantom_groups"], 4)


# ===========================================================================
# Budget / memory reuse
# ===========================================================================


class BudgetContracts(unittest.TestCase):
    def test_ub_total_is_the_planned_190848_bytes(self):
        buf = flash.plan_kernel()["buffers"]
        self.assertEqual(buf["ub_named_total_bytes"], 184192)
        self.assertEqual(buf["ub_shift_extra_bytes"], 6656)
        self.assertEqual(buf["ub_total_bytes"], 190848)
        self.assertLessEqual(buf["ub_total_bytes"], flash.PINNED_UB_SAFE_BYTES)

    def test_informational_ub_aliases_are_never_summed_into_the_total(self):
        buf = flash.plan_kernel()["buffers"]
        aliases = buf["informational_ub_aliases_not_summed"]
        self.assertTrue(aliases)
        self.assertEqual(buf["ub_named_total_bytes"], sum(buf["ub_named"].values()))
        # the alias bytes are the same bytes already counted under reuse names
        self.assertNotIn(sum(buf[a] for a in aliases), buf["ub_named"].values())
        self.assertEqual(buf["ub_total_bytes"],
                         buf["ub_named_total_bytes"] + buf["ub_shift_extra_bytes"])
        self.assertEqual(flash._naive_alias_sum_guard(flash.plan_kernel()),
                         buf["ub_total_bytes"])

    def test_separate_low_residual_float_tile_is_gone(self):
        buf = flash.plan_kernel()["buffers"]
        self.assertNotIn("prob_lo_float", buf["ub_named"])
        self.assertEqual(buf["ub_named"]["prob_lo_half"], 8192 * 2)
        # the score Float tile is explicitly the exp tile and the low tile
        self.assertEqual(list(k for k in buf["ub_named"] if k.startswith("score_float")),
                         ["score_float_also_exp_also_lo"])
        self.assertEqual(buf["ub_named"]["score_float_also_exp_also_lo"], 8192 * 4)

    def test_l0c_and_l1_exact_sizes(self):
        buf = flash.plan_kernel()["buffers"]
        self.assertEqual(buf["l0c_score_bytes"], 32768)
        self.assertEqual(buf["l0c_pv_bytes"], 20480)
        self.assertEqual(buf["l0c_total_bytes"], 32768 + 2 * 20480)
        self.assertEqual(buf["l1_q_bytes"], 5 * 64 * 16 * 2)
        self.assertEqual(buf["l1_k_bytes"], 5 * 128 * 16 * 2)
        self.assertEqual(buf["l1_v_bytes"], 8 * 80 * 16 * 2)
        self.assertEqual(buf["l1_p_hi_bytes"], 8 * 64 * 16 * 2)
        self.assertEqual(buf["l1_p_lo_bytes"], 8 * 64 * 16 * 2)
        self.assertEqual(buf["l1_total_bytes"], 83968)

    def test_repeat_immediates_and_alignment(self):
        vec = flash.plan_kernel()["vector"]
        self.assertLessEqual(vec["group_repeats"], 255)
        self.assertLessEqual(vec["half_repeats"], 255)
        self.assertLessEqual(vec["broadcast_rep_stride"], 255)
        self.assertEqual(vec["broadcast_mask"] * 4 % 32, 0)
        self.assertEqual(vec["group_repeats"], 16)
        self.assertEqual(vec["half_repeats"], 8)

    def test_element_counts_match_nz_shapes(self):
        vec = flash.plan_kernel()["vector"]
        self.assertEqual(vec["score_elems"], 8 * 64 * 16)
        self.assertEqual(vec["pv_elems"], 5 * 64 * 16)
        self.assertEqual(vec["score_groups"], 8)
        self.assertEqual(vec["pv_groups"], 5)

    def test_budget_guard_refuses_an_oversized_plan(self):
        plan = flash.plan_kernel()
        plan["buffers"]["ub_total_bytes"] = flash.PINNED_UB_SAFE_BYTES + 1
        with self.assertRaises(flash.FlashVisionError):
            flash._check_budget(plan)

    def test_budget_guard_refuses_oversized_l0c_and_l1(self):
        plan = flash.plan_kernel()
        plan["buffers"]["l0c_total_bytes"] = 262145
        with self.assertRaises(flash.FlashVisionError):
            flash._check_budget(plan)
        plan = flash.plan_kernel()
        plan["buffers"]["l1_total_bytes"] = 1024 * 1024 + 1
        with self.assertRaises(flash.FlashVisionError):
            flash._check_budget(plan)


# ===========================================================================
# DMA descriptors
# ===========================================================================


class DmaContracts(unittest.TestCase):
    def setUp(self):
        self.plan = flash.plan_kernel()
        self.dma = self.plan["dma"]
        self.lg = self.plan["logical"]

    def test_q_load_is_five_blocks_of_64_rows_with_the_unpadded_query_stride(self):
        q = self.dma["q_load"]
        self.assertEqual((q["nburst"], q["burst"], q["dst_stride"]), (5, 64, 0))
        self.assertEqual(q["src_stride"], self.lg["m"] - 64)

    def test_k_load_is_five_blocks_of_128_rows_with_the_padded_key_stride(self):
        k = self.dma["k_load"]
        self.assertEqual((k["nburst"], k["burst"], k["dst_stride"]), (5, 128, 0))
        self.assertEqual(k["src_stride"], self.lg["n_pad"] - 128)

    def test_v_load_is_a_contiguous_run_of_eight_key_blocks_by_80_dim_rows(self):
        v = self.dma["v_load"]
        self.assertEqual((v["nburst"], v["burst"]), (8, 80))
        self.assertEqual((v["src_stride"], v["dst_stride"]), (0, 0))

    def test_p_tiles_are_eight_64_unit_blocks(self):
        p = self.dma["p_to_l1"]
        self.assertEqual((p["nburst"], p["burst"]), (8, 64))

    def test_output_store_is_half_so_no_float_double_stride(self):
        out = self.dma["out_store"]
        self.assertEqual((out["nburst"], out["burst"], out["src_stride"]), (5, 64, 0))
        self.assertEqual(out["dst_stride"], self.lg["m"] - 64)
        self.assertNotEqual(out["dst_stride"], 2 * (self.lg["m"] - 64))

    def test_l0c_to_ub_bursts_are_32_kib_score_and_20_kib_pv(self):
        self.assertEqual(self.dma["score_l0c_to_ub"]["burst_len_kb"], 32)
        self.assertEqual(self.dma["pv_l0c_to_ub"]["burst_len_kb"], 20)

    def test_descriptors_tile_their_tiles_exactly(self):
        # nburst*burst*16 Half elements must equal the tile they fill, and the
        # skipped stride must land on the next block start.
        q = self.dma["q_load"]
        self.assertEqual(q["nburst"] * (q["burst"] + q["src_stride"]), 5 * self.lg["m"])
        k = self.dma["k_load"]
        self.assertEqual(k["nburst"] * (k["burst"] + k["src_stride"]), 5 * self.lg["n_pad"])
        out = self.dma["out_store"]
        self.assertEqual(out["nburst"] * (out["burst"] + out["dst_stride"]), 5 * self.lg["m"])


# ===========================================================================
# Phantom key masking (the correctness core of this round)
# ===========================================================================


class PhantomMaskContracts(unittest.TestCase):
    def test_phantom_keys_and_groups_for_the_real_geometry(self):
        lg = flash.plan_kernel()["logical"]
        self.assertEqual(lg["phantom_keys"], 64)
        self.assertEqual(lg["phantom_groups"], 4)
        start = lg["score_groups"] - lg["phantom_groups"]
        self.assertEqual(start, 4)
        self.assertEqual(start * lg["mb"] * 16, 4096)   # first phantom element
        self.assertEqual(lg["phantom_groups"] * lg["mb"] * 16, 4096)

    def test_no_phantom_tail_when_the_key_length_fills_the_last_block(self):
        plan = flash.plan_kernel(2, 128, 256)
        self.assertEqual(plan["logical"]["n_pad"], 256)
        self.assertEqual(plan["logical"]["phantom_keys"], 0)
        self.assertEqual(plan["logical"]["phantom_groups"], 0)
        stub = TikStub()
        flash._emit_kernel(stub, plan, stub)
        # no phantom branch is emitted when there is nothing to mask
        self.assertEqual(stub.counters["if_scope"], 1)      # only the task guard

    def test_mask_helper_targets_only_the_trailing_phantom_groups(self):
        plan = flash.plan_kernel()
        stub = TikStub()
        tensor = StubTensor(stub, "float32", (plan["vector"]["score_elems"],),
                            name="s_f_flat")
        repeats = flash._mask_phantom_scores(stub, tensor, plan, flash.NEG_MAX)
        self.assertEqual(repeats, 4 * 16)                   # 4 groups x 16 repeats
        fills = [r for r in stub.records if r["op"] == "vector_dup"]
        self.assertEqual(len(fills), 64)
        self.assertTrue(all(r["args"][2] == repr(flash.NEG_MAX) for r in fills))
        # the first fill starts at the first phantom element, i.e. group 4
        self.assertIn("s_f_flat[4096]", fills[0]["args"][1])

    def test_mask_is_a_noop_when_no_phantom_groups_exist(self):
        plan = flash.plan_kernel(2, 128, 256)
        stub = TikStub()
        tensor = StubTensor(stub, "float32", (plan["vector"]["score_elems"],),
                            name="s_f_flat")
        self.assertEqual(flash._mask_phantom_scores(stub, tensor, plan, flash.NEG_MAX), 0)
        self.assertEqual([r for r in stub.records if r["op"] == "vector_dup"], [])

    def test_emitter_masks_neg_max_before_shift_and_zero_before_denominator(self):
        plan, stub, _gm = emit()
        ifs = [r for r in stub.records if r["op"] == "if_scope"]
        phantom_ifs = [r for r in ifs if "kv_tiles" not in r["cond"] and "==" in r["cond"]]
        self.assertEqual(len(phantom_ifs), 2)
        # both phantom branches are dynamic: they test the KV loop variable
        self.assertTrue(all(r["cond_is_symbolic"] for r in phantom_ifs))

        # both branches carry the SAME condition text, so address them by record
        # position instead of by value.
        positions = [i for i, r in enumerate(stub.records)
                     if r["op"] == "if_scope" and "==" in r["cond"]
                     and "kv_tiles" not in r["cond"]]
        self.assertEqual(len(positions), 2)

        body_path = tuple(stub.records[positions[0]]["path"]) + ("if",)

        def fills_between(start, stop):
            return [r for r in stub.records[start + 1:stop]
                    if r["op"] == "vector_dup" and tuple(r["path"]) == body_path]

        neg = fills_between(positions[0], positions[1])
        zero = fills_between(positions[1], len(stub.records))
        self.assertEqual(len(neg), 64)
        self.assertEqual(len(zero), 64)
        self.assertTrue(all(r["args"][2] == repr(flash.NEG_MAX) for r in neg))
        self.assertTrue(all(r["args"][2] == "0.0" for r in zero))

        # ordering inside the KV body: mask(-FLTMAX) -> ... exp ... -> mask(0) -> denom
        kv = [r for r in stub.records if "loop3" in r["path"]]
        ops = [r["op"] for r in kv]
        first_neg = ops.index("if_scope")
        exp_at = ops.index("h_exp", ops.index("h_cast", first_neg))
        last_if = len(ops) - 1 - ops[::-1].index("if_scope")
        denom_at = ops.index("h_reduce_sum")
        self.assertLess(first_neg, exp_at)
        self.assertLess(exp_at, last_if)
        self.assertLess(last_if, denom_at)

    def test_padded_v_alone_is_documented_as_insufficient(self):
        mask = flash.plan_kernel()["math"]["phantom_key_mask"]
        self.assertTrue(mask["padded_v_alone_is_not_sufficient"])
        self.assertIn("denominator", mask["why"])
        self.assertEqual(mask["before_shift_and_exp"], flash.NEG_MAX)
        self.assertEqual(mask["after_exp_before_denominator"], 0.0)


# ===========================================================================
# Math order
# ===========================================================================


class MathContracts(unittest.TestCase):
    def test_probability_split_order_and_in_place_low_residual(self):
        plan = flash.plan_kernel()
        split = plan["math"]["probability_split"]
        self.assertEqual(split[0], "hi = Half(exp)")
        self.assertIn("in place", split[1])
        self.assertEqual(split[2], "lo = Half(store_lo)")
        self.assertEqual(plan["math"]["residual_scale"], 1024.0)
        self.assertEqual(plan["math"]["low_downscale"], 1.0 / 1024.0)

    def test_low_residual_is_written_into_the_score_buffer(self):
        _plan, stub, _gm = emit()
        kv = [r for r in stub.records if "loop3" in r["path"]]
        ops = [r["op"] for r in kv]
        # hi cast, widen, sub in place on the score tile, scale, lo cast
        hi = ops.index("h_cast")
        widen = ops.index("h_cast", hi + 1)
        sub = ops.index("h_sub", widen + 1)
        scale = ops.index("vec_muls", sub + 1)
        lo = ops.index("h_cast", scale + 1)
        self.assertLess(hi, widen)
        self.assertLess(widen, sub)
        self.assertLess(sub, scale)
        self.assertLess(scale, lo)
        # the in-place subtraction writes the score tile from the widened hi tile
        sub_args = [r for r in kv if r["op"] == "h_sub"][-1]["args"]
        self.assertEqual(sub_args[0], sub_args[1])            # dst == src0
        self.assertIn("score_f_ub", sub_args[0])              # score buffer
        self.assertIn("p_hi_f", sub_args[2])                  # widened hi
        # exactly eight 1024-scaled group multiplies on the score tile
        scale_ops = [r for r in kv if r["op"] == "vec_muls" and r["args"][3] == "1024.0"]
        self.assertEqual(len(scale_ops), 8)
        self.assertTrue(all("score_f_ub" in r["args"][1] for r in scale_ops))

    def test_denominator_is_a_true_float_sum_never_half(self):
        plan = flash.plan_kernel()
        self.assertEqual(plan["math"]["denominator_dtype"], "float32")
        self.assertEqual(plan["math"]["accumulator_dtype"], "float32")
        self.assertEqual(plan["math"]["main_softmax_dtype"], "float32")
        _plan, stub, _gm = emit()
        kv = [r for r in stub.records if "loop3" in r["path"]]
        sums = [r for r in kv if r["op"] == "h_reduce_sum"]
        self.assertEqual(len(sums), 1)
        self.assertIn("fold_f_a", sums[0]["args"][1])         # Float fold buffer
        # seven pairwise Float adds build the 8-group fold
        adds = [r for r in kv if r["op"] == "vec_add" and "fold_f" in r["args"][1]]
        self.assertEqual(len(adds), 7)

    def test_denominator_fold_order_is_recorded_as_evidence(self):
        fold = flash.plan_kernel()["math"]["denominator_fold"]
        self.assertEqual(fold["groups"], 8)
        self.assertEqual(fold["order"].count("fold_f_a=fold_f_a+fold_f_b"), 3)
        self.assertTrue(fold["association_changes_low_bits_vs_round99"])
        self.assertIn("vcadd", fold["reduction"])

    def test_half_shift_is_only_an_estimate(self):
        math_plan = flash.plan_kernel()["math"]
        self.assertEqual(math_plan["shift_estimate_dtype"], "float16")
        self.assertFalse(math_plan["shift_is_exact"])
        self.assertEqual(math_plan["main_scores_dtype"], "float32")

    def test_state_is_initialised_without_a_first_block_branch(self):
        init = flash.plan_kernel()["math"]["init"]
        self.assertEqual(init["row_max"], flash.NEG_MAX)
        self.assertTrue(math.isfinite(init["row_max"]))
        self.assertEqual(init["output"], 0.0)
        self.assertEqual(init["denominator"], 0.0)
        _plan, stub, _gm = emit()
        # all three FP32 state fills happen before the KV loop opens
        pre_kv = [r for r in stub.records if r["op"] == "vector_dup"
                  and "loop3" not in r["path"] and "loop2" in r["path"]]
        self.assertGreaterEqual(len(pre_kv), 80 + 1 + 1)

    def test_final_narrowing_is_one_half_cast_of_a_float_quotient(self):
        _plan, stub, _gm = emit()
        kv = [r for r in stub.records if "loop3" in r["path"]]
        post_kv = [r for r in stub.records
                   if "loop3" not in r["path"] and "loop2" in r["path"]]
        out_ops = [r["op"] for r in post_kv]
        div = out_ops.index("h_div")
        cast = out_ops.index("h_cast", div)
        store = out_ops.index("data_move", cast)
        self.assertLess(div, cast)
        self.assertLess(cast, store)
        hv = [r for r in post_kv if r["op"] == "h_div"][0]["args"]
        self.assertIn("pv_hi_ub", hv[0])
        self.assertIn("out_acc_ub", hv[1])
        self.assertIn("pv_lo_ub", hv[2])
        cv = [r for r in post_kv if r["op"] == "h_cast"][0]["args"]
        self.assertIn("out_h", cv[0])
        self.assertIn("pv_hi_ub", cv[1])
        # no Half accumulation anywhere: exactly one cast after the KV loop
        self.assertEqual(len([r for r in post_kv if r["op"] == "h_cast"]), 1)
        self.assertFalse([r for r in kv if "out_h" in "".join(r.get("args", []))])


# ===========================================================================
# Emitted structure (recording stub over the real emitter)
# ===========================================================================


class EmissionContracts(unittest.TestCase):
    def setUp(self):
        self.plan, self.stub, self.gm = emit()

    def test_five_pointer_abi_and_build_order(self):
        self.assertEqual(list(self.gm), ["q", "k", "v", "aux", "out"])
        self.assertEqual(self.gm["q"].dtype, "float16")
        self.assertEqual(self.gm["k"].dtype, "float16")
        self.assertEqual(self.gm["v"].dtype, "float16")
        self.assertEqual(self.gm["aux"].dtype, "float32")
        self.assertEqual(self.gm["out"].dtype, "float16")
        self.assertEqual(list(flash.KERNEL_ARG_ORDER),
                         ["q_gm_float16", "k_gm_float16", "v_gm_float16",
                          "aux_gm_float32", "out_gm_float16"])
        self.assertEqual(self.gm["q"].shape, (16, 5, 4032, 16))
        self.assertEqual(self.gm["k"].shape, (16, 5, 4096, 16))
        self.assertEqual(self.gm["v"].shape, (16, 256, 80, 16))
        self.assertEqual(self.gm["out"].shape, (16, 5, 4032, 16))
        self.assertEqual(self.gm["aux"].shape, (16,))

    def test_three_loops_and_kv_loop_is_dynamic_not_unrolled(self):
        # core loop, task-step loop, KV loop -- and the KV loop must be a
        # for_range over kv_tiles, never a python range() unroll.
        loops = [r for r in self.stub.records if r["op"] == "for_range"]
        self.assertEqual(len(loops), 3)
        self.assertEqual(loops[0]["endt"], 30)          # 30 launched cores
        self.assertEqual(loops[1]["endt"], 34)          # 34 task steps
        self.assertEqual(loops[2]["endt"], 32)          # 32 KV tiles
        self.assertEqual(self.stub.counters["for_range"], 3)
        self.assertNotIn("range(kv_tiles)", TEXT)
        self.assertNotIn("range(gr[\"kv_tiles\"])", TEXT)

    def test_static_per_kv_instruction_counts(self):
        kv = [r for r in self.stub.records if "loop3" in r["path"]]
        counts = {}
        for r in kv:
            counts[r["op"]] = counts.get(r["op"], 0) + 1
        self.assertEqual(counts["matmul"], 3)           # 1 QK + 2 P@V
        self.assertEqual(counts["tensor_mov"], 3)       # 1 score + 2 P@V
        moves = [r for r in kv if r["op"] == "data_move"]
        # K, V, P_hi, P_lo plus the eight exact native shift-broadcast copies
        self.assertEqual(len(moves), 4 + 8)
        self.assertEqual(len([r for r in moves if "k_l1" in r["args"][0]]), 1)
        self.assertEqual(len([r for r in moves if "v_l1" in r["args"][0]]), 1)
        self.assertEqual(len([r for r in moves if "p_hi_l1" in r["args"][0]]), 1)
        self.assertEqual(len([r for r in moves if "p_lo_l1" in r["args"][0]]), 1)
        self.assertEqual(counts["h_reduce_max"], 1)     # Half shift estimate
        self.assertEqual(counts["h_reduce_sum"], 1)     # Float denominator
        self.assertEqual(counts["h_exp"], 2)            # rescale + score exp
        self.assertEqual(counts["vec_max"], 7)          # Half 8-group fold
        self.assertEqual(len([r for r in kv if r["op"] == "vec_add"
                              and "fold_f" in r["args"][1]]), 7)

    def test_matmul_shapes_are_the_bc128_geometry(self):
        kv = [r for r in self.stub.records if "loop3" in r["path"]]
        mats = [r for r in kv if r["op"] == "matmul"]
        # QK: m=64, k=80, n=128 (BC) with an initialised L0C
        qk = mats[0]
        self.assertEqual(qk["shape"], [64, 80, 128])     # m, k(5*16), n(BC)
        self.assertTrue(qk["init_l1out"])
        # P@V: m=64, k=128 (BC), n=80 for both halves
        for mat in mats[1:]:
            self.assertEqual(mat["shape"], [64, 128, 80])
            self.assertTrue(mat["init_l1out"])

    def test_no_high_level_hmax_on_a_sliced_nz_operand(self):
        # Round99's compile failure: h_max received NZ slices with a mismatched
        # inferred step. Every emitted h_max must be a plain 1-D row op.
        for rec in self.stub.ops("h_max"):
            for arg in rec["args"]:
                self.assertNotIn("nz_slice", arg)
                self.assertNotIn("[", arg)               # no indexed operand at all
        for rec in self.stub.ops("vec_max"):
            self.assertNotIn("hold", rec["args"][0])

    def test_no_large_strided_hcast(self):
        # Round99 also failed when h_cast's destination was a big strided view.
        for rec in self.stub.ops("h_cast"):
            dst, src = rec["args"][0], rec["args"][1]
            self.assertNotIn("rs[", dst)                 # not into a reshaped view
            self.assertNotIn("hold", dst)
            self.assertNotIn("hold", src)
        # exactly two wide whole-tile casts of the score tile exist: the Half
        # shift estimate (step 5) and the P_hi split (step 11). Both are flat
        # equal-length casts, never a strided view.
        wide = [r for r in self.stub.ops("h_cast")
                if "p_hi_h" in r["args"][0] and "score_f_ub" in r["args"][1]]
        self.assertEqual(len(wide), 2)

    def test_q_is_loaded_once_per_task_and_out_stored_once(self):
        q_moves = [r for r in self.stub.ops("data_move")
                   if "q_l1" in r["args"][0]]
        self.assertEqual(len(q_moves), 1)
        self.assertNotIn("loop3", q_moves[0]["path"])    # outside the KV loop
        out_moves = [r for r in self.stub.ops("data_move")
                     if "out_gm" in r["args"][0]]
        self.assertEqual(len(out_moves), 1)
        self.assertNotIn("loop3", out_moves[0]["path"])

    def test_sync_scopes_wrap_shared_l1_and_l0c_reuse(self):
        # K/V reload and the L1 P rewrite must both sit in a non-disabled sync
        # scope, and every new_stmt_scope in the emitter is disable_sync=False.
        scopes = self.stub.ops("new_stmt_scope")
        self.assertTrue(scopes)
        self.assertTrue(all(r["disable_sync"] is False for r in scopes))
        kv_scopes = [r for r in scopes if "loop3" in r["path"]]
        self.assertGreaterEqual(len(kv_scopes), 4)       # KV, score, PV, combine

    def test_exact_native_shift_broadcast_uses_the_proven_copy_pattern(self):
        ors = self.stub.ops("vec_or")
        trans = self.stub.ops("vec_trans")
        self.assertEqual(len(ors), 4)
        self.assertEqual(len(trans), 4)
        # one 1024-element Float holding tile, then 8 NZ group copies
        copies = [r for r in self.stub.ops("data_move")
                  if "p_hi_f" in r["args"][0]]
        self.assertEqual(len(copies), 8)

    def test_float_rescale_and_denominator_broadcasts_stay_float_scalar(self):
        # Only the Half-representable shift uses the bit-copy path; rescale and
        # denominator are expanded as Float scalars (64 rows x 5 pv groups).
        dup = [r for r in self.stub.records if r["op"] == "vector_dup"]
        rescale = [r for r in dup if "loop3" in r["path"]
                   and "pv_lo_ub" in r["args"][1]]
        self.assertEqual(len(rescale), 64)          # 5 pv groups per row, 64 rows
        denom = [r for r in dup if "loop3" not in r["path"]
                 and "loop2" in r["path"] and "pv_lo_ub" in r["args"][1]]
        self.assertEqual(len(denom), 64)
        # both reuse the same masked 16-value row slot with repeat stride 2*MB
        self.assertTrue(all(r["args"][3] == "5" and r["args"][4] == "1"
                            and r["args"][5] == "128" for r in rescale + denom))
        scalars = [r for r in self.stub.ops("scalar_set")]
        self.assertTrue(scalars)
        self.assertTrue(any(r["from_handle"] for r in scalars))


# ===========================================================================
# Source discipline / non-claims
# ===========================================================================


class SourceContracts(unittest.TestCase):
    def test_module_scope_imports_are_stdlib_only(self):
        tree = ast.parse(TEXT)
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        self.assertEqual(top, {"hashlib", "json", "os", "tempfile", "pathlib",
                               "__future__"})

    def test_cann_and_torch_imports_are_lazy(self):
        tree = ast.parse(TEXT)
        lazy = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Import):
                        lazy.update(a.name.split(".")[0] for a in child.names)
                    elif isinstance(child, ast.ImportFrom) and child.module:
                        lazy.add(child.module.split(".")[0])
        self.assertIn("tbe", lazy)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for banned in ("torch", "torch_npu", "numpy", "tensorflow", "mindspore"):
            self.assertNotIn(banned, imported)

    def test_no_forbidden_shortcut_in_the_source(self):
        flat = " ".join(TEXT.split())
        for banned in ("bfloat16", "FloatLinear", "np.clip", "precomputed_condition"):
            self.assertNotIn(banned, flat)
        # Strip docstrings, then prove the executable code never clamps scores and
        # never takes a tolerance: both only ever appear as refusal statements.
        tree = ast.parse(TEXT)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    node.body.pop(0)
        code = ast.unparse(tree)
        self.assertNotIn("clamp(", code)
        self.assertNotIn("tolerance", code.replace("numerical_tolerance", ""))
        self.assertNotIn("bfloat16", code)
        self.assertNotIn("torch", code)
        # the phantom mask and the head-dim scale must both be present as calls
        self.assertIn("_mask_phantom_scores(", code)
        self.assertIn("scale_for(", code)

    def test_receipts_claim_nothing(self):
        self.assertIn("SOURCE ONLY", TEXT)
        self.assertIn('"executed": False', TEXT)
        self.assertIn('"not_a_speed_claim": True', TEXT)
        self.assertIn('"full_encoder_precision_accepted": False', TEXT)
        self.assertIn('"whole_goal_complete": False', TEXT)
        self.assertIn('"numerical_tolerance": None', TEXT)

    def test_documented_assumptions_are_non_empty_and_specific(self):
        assumptions = flash.ASSUMPTIONS
        self.assertGreaterEqual(len(assumptions), 10)
        joined = " ".join(assumptions)
        for needle in ("h_reduce_max", "vec_max", "if_scope", "tensor_mov",
                       "-FLT_MAX", "vec_add"):
            self.assertIn(needle, joined)

    def test_host_protocol_rejects_missing_partial_negative_and_wrong_dim(self):
        for kwargs in ({}, {"q_absmax": 1.0}, {"max_abs_scaled_qk": -1.0},
                       {"q_absmax": -1.0, "k_absmax": 1.0},
                       {"max_abs_scaled_qk": 1.0, "d": 128}):
            with self.assertRaises(flash.FlashVisionError):
                flash.check_host_protocol(**kwargs)

    def test_host_protocol_accepts_the_bound_and_refuses_beyond_without_clamping(self):
        self.assertTrue(flash.check_host_protocol(flash.SHIFT_BOUND)["ok"])
        self.assertTrue(flash.check_host_protocol(0.0)["ok"])
        for bad in (flash.SHIFT_BOUND * 1.0001, float("inf"), float("nan")):
            with self.assertRaises(flash.FlashVisionError):
                flash.check_host_protocol(bad)
        self.assertFalse(flash.check_host_protocol(1.0)["clamped"])
        ok = flash.check_host_protocol(q_absmax=8.0, k_absmax=8.0)
        self.assertLess(ok["cauchy_bound"], flash.SHIFT_BOUND)


# ===========================================================================
# Capability fail-closed (fake platform module, no CANN import)
# ===========================================================================


class CapabilityContracts(unittest.TestCase):
    def _with_platform(self, table, supported):
        import sys
        from types import SimpleNamespace

        fake = SimpleNamespace(
            set_current_compile_soc_info=lambda soc: None,
            get_soc_spec=lambda key: table[key],
            intrinsic_check_support=lambda name, dtype: supported.get((name, dtype),
                                                                      False))
        saved = {k: sys.modules.get(k) for k in ("tbe", "tbe.common",
                                                 "tbe.common.platform")}
        tbe = SimpleNamespace(common=SimpleNamespace(platform=fake))
        sys.modules["tbe"] = tbe
        sys.modules["tbe.common"] = tbe.common
        sys.modules["tbe.common.platform"] = fake
        return saved

    def _restore(self, saved):
        import sys
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    TABLE = {"SHORT_SOC_VERSION": "Ascend910", "CORE_NUM": 32, "UB_SIZE": 262144,
             "L1_SIZE": 1048576, "L0C_SIZE": 262144}

    def _all_supported(self):
        supported = {}
        for intrinsic, dtype in flash.REQUIRED_INTRINSICS + flash.OPTIONAL_INTRINSICS:
            supported[(intrinsic, dtype)] = True
        supported[("Intrinsic_vcmax", "float32")] = False   # true on this part
        return supported

    def test_accepts_the_supported_target(self):
        saved = self._with_platform(self.TABLE, self._all_supported())
        try:
            report = flash.capability_report()
        finally:
            self._restore(saved)
        self.assertTrue(report["all_required_supported"])
        self.assertEqual(report["compile_soc"], "Ascend910")
        self.assertEqual(report["launched_usable_cores"], 30)

    def test_fails_closed_when_a_required_instruction_is_missing(self):
        supported = self._all_supported()
        supported[("Intrinsic_vcadd", "float32")] = False
        saved = self._with_platform(self.TABLE, supported)
        try:
            with self.assertRaises(flash.FlashVisionError) as ctx:
                flash.capability_report()
        finally:
            self._restore(saved)
        self.assertIn("vcadd", str(ctx.exception))

    def test_refuses_float_vcmax_as_a_required_intrinsic(self):
        # Float vcmax does not exist on this part; it must stay optional only.
        self.assertNotIn(("Intrinsic_vcmax", "float32"), flash.REQUIRED_INTRINSICS)
        self.assertIn(("Intrinsic_vcmax", "float32"), flash.OPTIONAL_INTRINSICS)

    def test_refuses_the_wrong_soc(self):
        table = dict(self.TABLE, SHORT_SOC_VERSION="Ascend910B")
        saved = self._with_platform(table, self._all_supported())
        try:
            with self.assertRaises(flash.FlashVisionError):
                flash.capability_report()
        finally:
            self._restore(saved)

    def test_refuses_a_capability_that_cannot_hold_the_plan(self):
        table = dict(self.TABLE, UB_SIZE=190847)
        saved = self._with_platform(table, self._all_supported())
        try:
            with self.assertRaises(flash.FlashVisionError):
                flash.capability_report()
        finally:
            self._restore(saved)

    def test_emitter_fails_closed_without_a_required_tik_method(self):
        class _Hiding:
            def __init__(self, inner, hidden):
                self._inner = inner
                self._hidden = hidden

            def __getattr__(self, name):
                if name == self._hidden:
                    raise AttributeError(name)
                return getattr(self._inner, name)

        plan = flash.plan_kernel()
        stub = TikStub()
        for missing in ("h_reduce_sum", "vec_max", "vec_muls", "tensor_mov"):
            with self.assertRaises(flash.FlashVisionError):
                flash._emit_kernel(_Hiding(stub, missing), plan, stub)


# ===========================================================================
# Manifest consistency (cannot drift from the live plan / files)
# ===========================================================================


class ManifestContracts(unittest.TestCase):
    def setUp(self):
        path = HERE / "manifest101.json"
        self.assertTrue(path.is_file(), "run make_manifest101.py first")
        self.manifest = json.loads(path.read_text())

    def test_manifest_hashes_match_the_files_on_disk(self):
        for name, record in self.manifest["files"].items():
            self.assertEqual(record["sha256"],
                             hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                             name)

    def test_manifest_records_the_immutable_upstream_source(self):
        upstream = self.manifest["upstream_start_source"]
        self.assertTrue(upstream["unchanged_on_disk"])
        self.assertEqual(upstream["actual_sha256"], upstream["expected_sha256"])
        self.assertEqual(len(upstream["expected_sha256"]), 64)
        # the pinned hash is the executed Round99 kernel, not a local edit
        up = HERE.parent.parent / "round99" / "flash" / "flash_vision_shift_temp_fixed99.py"
        self.assertEqual(hashlib.sha256(up.read_bytes()).hexdigest(),
                         upstream["expected_sha256"])

    def test_manifest_geometry_budget_and_api_match_the_live_plan(self):
        plan = flash.plan_kernel()
        geo, mem = self.manifest["geometry"], self.manifest["memory"]
        self.assertEqual(geo["kv_tiles"], plan["grid"]["kv_tiles"])
        self.assertEqual(geo["tasks"], plan["grid"]["tasks"])
        self.assertEqual(geo["cores"], plan["grid"]["cores"])
        self.assertEqual(geo["steps"], plan["grid"]["steps"])
        self.assertEqual(geo["phantom_keys_in_last_kv_tile"],
                         plan["logical"]["phantom_keys"])
        self.assertEqual(mem["ub_total_bytes"], plan["buffers"]["ub_total_bytes"])
        self.assertEqual(mem["l1_total_bytes"], plan["buffers"]["l1_total_bytes"])
        self.assertEqual(mem["l0c_total_bytes"], plan["buffers"]["l0c_total_bytes"])
        self.assertEqual(mem["ub_headroom_bytes"],
                         flash.PINNED_UB_SAFE_BYTES - plan["buffers"]["ub_total_bytes"])
        self.assertEqual(self.manifest["build_api"]["kernel_name"],
                         flash.kernel_name_for(plan))

    def test_manifest_carries_the_no_claim_fields(self):
        for field, value in (("executed", False), ("compiled_now", False),
                             ("not_a_speed_claim", True),
                             ("full_encoder_precision_accepted", False),
                             ("whole_goal_complete", False),
                             ("numerical_tolerance", None)):
            self.assertEqual(self.manifest[field], value, field)
        self.assertGreaterEqual(len(self.manifest["unverified_cann_behaviours_for_the_root_smoke"]),
                                len(flash.ASSUMPTIONS))


if __name__ == "__main__":
    unittest.main()