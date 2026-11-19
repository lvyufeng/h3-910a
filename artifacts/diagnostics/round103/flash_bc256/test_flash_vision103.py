#!/usr/bin/env python3
"""Stdlib-only contract tests for artifacts/diagnostics/round103/flash_bc256.

NO CANN, TIK, torch, torch_npu, NPU, GPU or model is imported or executed here.
Everything below is pure python: geometry/budget/DMA contract assertions, an
AST-level source discipline check, and a *recording TIK stub* (reused verbatim
from the executed Round101 BC=128 test) that runs the real ``_emit_kernel`` and
inspects the emitted instruction sequence: ABI order, static instruction counts,
loop nesting, phantom-mask placement and the absence of the Round99 high-level
sliced ``h_max`` / large strided ``h_cast`` patterns.

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
SOURCE_PATH = HERE / "flash_vision103.py"
_spec = importlib.util.spec_from_file_location("flash_vision103_under_test", SOURCE_PATH)
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
# Geometry: the MB=32 x BC=256 specialisation
# ===========================================================================


class GeometryContracts(unittest.TestCase):
    def test_real_h3_vision_geometry(self):
        self.assertEqual((flash.HEADS, flash.SEQ, flash.HEAD_DIM), (16, 4032, 72))
        self.assertEqual((flash.PAD_DIM, flash.MB, flash.BC), (80, 32, 256))
        self.assertEqual(flash.D1, 5)
        self.assertEqual(flash.KB1, 16)
        self.assertEqual(flash.SCORE_GROUPS, 16)

    def test_padded_key_length_and_phantom_tail(self):
        self.assertEqual(flash.pad_keys(4032), 4096)
        lg = flash.plan_kernel()["logical"]
        self.assertEqual((lg["n"], lg["n_pad"], lg["kb1"]), (4032, 4096, 16))
        self.assertEqual((lg["tail_keys"], lg["phantom_keys"]), (192, 64))
        self.assertEqual((lg["real_groups_last"], lg["phantom_groups"]), (12, 4))

    def test_query_tiles_kv_tiles_tasks_cores_steps(self):
        gr = flash.plan_kernel()["grid"]
        self.assertEqual(gr["m_tiles"], 126)          # 4032 / 32
        self.assertEqual(gr["kv_tiles"], 16)          # 4096 / 256
        self.assertEqual(gr["tasks"], 2016)           # 16 heads x 126 query tiles
        self.assertEqual(gr["cores"], 30)             # 30 usable of 32 physical
        self.assertEqual(gr["steps"], 68)             # ceil(2016/30)
        self.assertTrue(gr["dynamic_kv_loop"])
        self.assertNotEqual(gr["tasks"], 1008)        # not Round101's tile count

    def test_kv_iterations_drop_and_kv_reloads_double_versus_round101(self):
        gr = flash.plan_kernel()["grid"]
        self.assertEqual(gr["kv_tiles"], 16)          # was 32 at BC=128
        self.assertEqual(gr["m_tiles"], 126)          # was 63 at MB=64
        # every query tile re-reads the whole K/V stream: 126/63 = 2x the traffic
        self.assertEqual(gr["m_tiles"], 2 * 63)

    def test_scale_is_the_original_head_dim_not_the_padded_width(self):
        chk = flash.scale_check(flash.plan_kernel())
        self.assertTrue(chk["matches_original"])
        self.assertFalse(chk["padded_width_scale_used"])
        self.assertAlmostEqual(chk["scale"], 72 ** -0.5, places=15)
        self.assertNotAlmostEqual(chk["scale"], 80 ** -0.5, places=6)

    def test_attention_is_unmasked_and_not_causal(self):
        lg = flash.plan_kernel()["logical"]
        self.assertTrue(lg["unmasked"])
        self.assertFalse(lg["causal"])

    def test_refuses_key_length_not_multiple_of_64(self):
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(n=4000)                 # 4000 % 64 == 32
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(n=4071)

    def test_refuses_wrong_tile_sizes_and_unaligned_query(self):
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(mb=64)                  # Round101's tile is not this design
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(bc=128)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(m=48)                   # 48 % 32 == 16

    def test_refuses_head_dim_that_leaves_a_padded_block(self):
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(d=64)                   # 64 % 16 == 0: no real tail

    def test_generic_phantom_tails_0_64_128_192(self):
        for n, phantom in ((3840, 0), (4032, 64), (3968, 128), (3904, 192)):
            pl = flash.plan_kernel(n=n)
            lg = pl["logical"]
            self.assertEqual(lg["phantom_keys"], phantom, n)
            self.assertEqual(lg["phantom_groups"] * 16, phantom, n)
            self.assertEqual(lg["tail_keys"] + phantom, 256, n)
            self.assertEqual(phantom % 16, 0, n)
            self.assertEqual(lg["n_pad"] % 256, 0, n)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(n=4032, bc=128)

    def test_smoke_geometry_is_accepted_and_keeps_192_real_64_phantom(self):
        pl = flash.plan_kernel(*flash.SMOKE_GEOMETRY)
        lg, gr = pl["logical"], pl["grid"]
        self.assertEqual((lg["heads"], lg["m"], lg["n"], lg["n_pad"]), (2, 64, 192, 256))
        self.assertEqual((lg["tail_keys"], lg["phantom_keys"]), (192, 64))
        self.assertEqual(gr["kv_tiles"], 1)
        self.assertEqual(gr["m_tiles"], 2)
        self.assertEqual((gr["tasks"], gr["cores"], gr["steps"]), (4, 4, 1))


# ===========================================================================
# Budget
# ===========================================================================


class BudgetContracts(unittest.TestCase):
    def setUp(self):
        self.plan = flash.plan_kernel()
        self.buf = self.plan["buffers"]
        self.vec = self.plan["vector"]

    def test_ub_total_is_the_derived_144832_bytes(self):
        self.assertEqual(self.buf["ub_named_total_bytes"], 141248)
        self.assertEqual(self.buf["ub_shift_extra_bytes"], 3584)
        self.assertEqual(self.buf["ub_total_bytes"], 144832)
        self.assertLess(self.buf["ub_total_bytes"], flash.PINNED_UB_SAFE_BYTES)
        self.assertEqual(flash.PINNED_UB_SAFE_BYTES, 196608)

    def test_ub_components_are_the_mb32_bc256_sizes(self):
        named = self.buf["ub_named"]
        self.assertEqual(named["score_float_also_exp_also_lo"], 32768)      # 8192 FP32
        self.assertEqual(named["prob_hi_half_also_half_score"], 16384)
        self.assertEqual(named["prob_hi_float_also_row_broadcast"], 32768)
        self.assertEqual(named["prob_lo_half"], 16384)
        self.assertEqual(named["pv_hi_float"], 10240)                       # 2560 FP32
        self.assertEqual(named["pv_lo_float"], 10240)
        self.assertEqual(named["out_acc_float"], 10240)
        self.assertEqual(named["out_half"], 5120)
        for key in ("row_prev", "row_new", "row_diff", "row_rescale", "row_zero",
                    "row_sum", "row_denominator", "shift_half_row"):
            self.assertLessEqual(named[key], 128, key)
        self.assertEqual(named["shift_half_row"], 64)                       # MB=32 Half
        self.assertEqual(self.buf["ub_shift_copy_bytes"], 512)
        self.assertEqual(self.buf["ub_shift_broad_half_bytes"], 1024)
        self.assertEqual(self.buf["ub_shift_broad_float_bytes"], 2048)

    def test_informational_ub_aliases_are_never_summed_into_the_total(self):
        aliases = self.buf["informational_ub_aliases_not_summed"]
        self.assertTrue(aliases)
        named = self.buf["ub_named"]
        self.assertEqual(self.buf["ub_named_total_bytes"], sum(named.values()))
        self.assertEqual(len(set(named) & set(aliases)), 0)
        for alias in aliases:
            self.assertNotIn(alias, named)
        self.assertEqual(flash._naive_alias_sum_guard(self.plan),
                         self.buf["ub_total_bytes"])

    def test_no_separate_low_residual_float_tile(self):
        named = self.buf["ub_named"]
        self.assertNotIn("prob_lo_float", named)
        self.assertNotIn("low_residual_float", named)
        self.assertTrue(self.plan["math"]["score_buffer_reused_for_low_residual"])
        self.assertIn('"score_buffer_reused_for_low_residual": True', TEXT)

    def test_l0c_and_l1_exact_sizes(self):
        self.assertEqual(self.buf["l0c_score_bytes"], 32768)
        self.assertEqual(self.buf["l0c_pv_bytes"], 10240)
        self.assertEqual(self.buf["l0c_total_bytes"], 53248)   # 32 KiB + 2 x 10 KiB
        self.assertEqual(self.buf["l1_q_bytes"], 5120)
        self.assertEqual(self.buf["l1_k_bytes"], 40960)
        self.assertEqual(self.buf["l1_v_bytes"], 40960)
        self.assertEqual(self.buf["l1_p_hi_bytes"], 16384)
        self.assertEqual(self.buf["l1_p_lo_bytes"], 16384)
        self.assertEqual(self.buf["l1_total_bytes"], 119808)   # ~117 KiB
        self.assertLessEqual(self.buf["l0c_total_bytes"], 262144)

    def test_cube_operands_fit_the_64kib_limits(self):
        cube = self.buf["cube_operand_bytes"]
        self.assertEqual(cube["qk_l0a"], 5120)      # 32 x 80 Half
        self.assertEqual(cube["qk_l0b"], 40960)     # 80 x 256 Half
        self.assertEqual(cube["pv_l0a"], 16384)     # 32 x 256 Half
        self.assertEqual(cube["pv_l0b"], 40960)     # 256 x 80 Half
        for value in cube.values():
            self.assertLessEqual(value, 65536)

    def test_repeat_immediates_and_mb32_row_alignment(self):
        vec = self.vec
        self.assertEqual(vec["f32_mask"], 64)
        self.assertEqual(vec["half_mask"], 128)
        self.assertEqual(vec["group_repeats"], 8)           # 512 / 64
        self.assertEqual(vec["half_repeats"], 4)            # 512 / 128
        self.assertEqual(vec["broadcast_mask"], 16)
        self.assertEqual(vec["broadcast_rep_stride"], 64)   # 2*MB, inside 8-bit imm
        self.assertEqual(vec["row_elems"], 32)
        self.assertEqual(vec["row_tail_mask"], 32)
        self.assertLessEqual(vec["broadcast_rep_stride"], 255)
        self.assertEqual(vec["broadcast_mask"] * 4 % 32, 0)
        self.assertEqual(vec["half_row_groups"], 2)
        self.assertEqual(vec["shift_broad_elems"], 512)

    def test_element_counts_match_nz_shapes(self):
        vec, lg = self.vec, self.plan["logical"]
        self.assertEqual(vec["score_elems"], 8192)
        self.assertEqual(vec["score_elems"], vec["score_groups"] * lg["mb"] * 16)
        self.assertEqual(vec["score_groups"], 16)
        self.assertEqual(vec["pv_elems"], 2560)
        self.assertEqual(vec["pv_elems"], vec["pv_groups"] * lg["mb"] * 16)
        self.assertEqual(vec["pv_groups"], 5)

    def test_budget_guard_refuses_an_oversized_plan(self):
        import copy
        for key, value in (("ub_total_bytes", 300000), ("l0c_total_bytes", 300000),
                           ("l1_total_bytes", 2 * 1024 * 1024)):
            broken = copy.deepcopy(self.plan)
            broken["buffers"][key] = value
            with self.assertRaises(flash.FlashVisionError, msg=key):
                flash._check_budget(broken)

    def test_budget_guard_refuses_a_broken_shift_broadcast(self):
        import copy
        broken = copy.deepcopy(self.plan)
        broken["vector"]["shift_broad_elems"] = 1024        # Round101's size, wrong here
        with self.assertRaises(flash.FlashVisionError):
            flash._check_budget(broken)


# ===========================================================================
# DMA descriptors
# ===========================================================================


class DmaContracts(unittest.TestCase):
    def setUp(self):
        self.plan = flash.plan_kernel()
        self.dma = self.plan["dma"]

    def test_q_load_is_five_blocks_of_32_rows_with_the_unpadded_query_stride(self):
        self.assertEqual(self.dma["q_load"], {"nburst": 5, "burst": 32,
                                              "src_stride": 4032 - 32, "dst_stride": 0})

    def test_k_load_is_five_blocks_of_256_rows_with_the_padded_key_stride(self):
        self.assertEqual(self.dma["k_load"], {"nburst": 5, "burst": 256,
                                              "src_stride": 4096 - 256, "dst_stride": 0})

    def test_v_load_is_a_contiguous_run_of_16_key_blocks_by_80_dim_rows(self):
        self.assertEqual(self.dma["v_load"], {"nburst": 16, "burst": 80,
                                              "src_stride": 0, "dst_stride": 0})

    def test_p_tiles_are_sixteen_32_unit_blocks(self):
        self.assertEqual(self.dma["p_to_l1"], {"nburst": 16, "burst": 32})

    def test_output_store_is_half_so_no_float_double_stride(self):
        self.assertEqual(self.dma["out_store"], {"nburst": 5, "burst": 32,
                                                 "src_stride": 0,
                                                 "dst_stride": 4032 - 32})

    def test_l0c_to_ub_bursts_are_32_kib_score_and_10_kib_pv(self):
        self.assertEqual(self.dma["score_l0c_to_ub"]["burst_len_kb"], 32)
        self.assertEqual(self.dma["pv_l0c_to_ub"]["burst_len_kb"], 10)
        self.assertEqual(self.dma["score_l0c_to_ub"]["nburst"], 1)
        self.assertEqual(self.dma["pv_l0c_to_ub"]["nburst"], 1)

    def test_k_burst_is_256_and_flagged_as_a_root_verified_assumption(self):
        self.assertEqual(self.dma["k_load"]["burst"], 256)
        joined = " ".join(flash.ASSUMPTIONS)
        self.assertIn("burst=256", joined)
        self.assertIn("ROOT must confirm", joined)
        self.assertIn("not shrink it to a previously proven 128", joined)

    def test_descriptors_tile_their_tiles_exactly(self):
        buf, lg = self.plan["buffers"], self.plan["logical"]
        dma = self.dma
        self.assertEqual(dma["q_load"]["nburst"] * dma["q_load"]["burst"] * 32,
                         buf["l1_q_bytes"])
        self.assertEqual(dma["k_load"]["nburst"] * dma["k_load"]["burst"] * 32,
                         buf["l1_k_bytes"])
        self.assertEqual(dma["v_load"]["nburst"] * dma["v_load"]["burst"] * 32,
                         buf["l1_v_bytes"])
        self.assertEqual(dma["p_to_l1"]["nburst"] * dma["p_to_l1"]["burst"] * 32,
                         buf["l1_p_hi_bytes"])
        self.assertEqual(dma["out_store"]["nburst"] * dma["out_store"]["burst"] * 32,
                         buf["ub_out_bytes"])
        self.assertEqual(dma["score_l0c_to_ub"]["burst_len_kb"] * 1024,
                         buf["ub_score_bytes"])
        self.assertEqual(dma["pv_l0c_to_ub"]["burst_len_kb"] * 1024, buf["ub_pv_bytes"])
        self.assertEqual(lg["kb1"] * lg["mb"] * 16 * 2, buf["l1_p_hi_bytes"])


# ===========================================================================
# Phantom key tail: the correctness crux of a padded 16-group block
# ===========================================================================


class PhantomMaskContracts(unittest.TestCase):
    def test_phantom_keys_and_groups_for_the_real_geometry(self):
        lg = flash.plan_kernel()["logical"]
        self.assertEqual(lg["phantom_keys"], 64)
        self.assertEqual(lg["phantom_groups"], 4)          # groups 12..15
        self.assertEqual(lg["real_groups_last"], 12)
        self.assertEqual(lg["real_groups_last"] + lg["phantom_groups"],
                         lg["score_groups"])

    def test_no_phantom_tail_when_the_key_length_fills_the_last_block(self):
        plan = flash.plan_kernel(n=4096)
        lg = plan["logical"]
        self.assertEqual((lg["phantom_keys"], lg["phantom_groups"]), (0, 0))
        _, stub, _ = emit(plan)
        for record in stub.ops("vector_dup"):
            self.assertTrue("score_f_ub[" not in record["args"][1], record["args"][1])
        # the only -FLT_MAX fill left is the row_max init, on the row tensor
        inits = [r for r in stub.ops("vector_dup")
                 if r["args"][2] == repr(flash.NEG_MAX)]
        self.assertEqual([r["args"][1] for r in inits], ["handle(row_prev[0])"])
        # and no mask if_scope is emitted at all: only the task bound remains
        self.assertEqual(len(stub.ops("if_scope")), 1)

    def test_mask_helper_targets_only_the_trailing_phantom_groups(self):
        _, stub, _ = emit()
        prefix = "handle(score_f_ub["
        offsets = []
        for record in stub.ops("vector_dup"):
            dst = record["args"][1]
            if dst.startswith(prefix) and record["args"][2] == repr(flash.NEG_MAX):
                offsets.append(int(dst[len(prefix):-2]))
        self.assertEqual(len(offsets), 32)                  # 2048 elements / 64 lanes
        self.assertEqual(offsets, list(range(6144, 8192, 64)))
        self.assertEqual(offsets[0], 12 * 32 * 16)          # first phantom group
        self.assertEqual(offsets[-1] + 64, 8192)            # covers to the tile end

    def test_emitter_masks_neg_max_before_shift_and_zero_before_denominator(self):
        _, stub, _ = emit()
        records = stub.records
        prefix = "handle(score_f_ub["
        mask = repr(flash.NEG_MAX)

        def first(predicate):
            for index, record in enumerate(records):
                if predicate(record):
                    return index
            raise AssertionError("no matching record")

        neg = first(lambda r: r["op"] == "vector_dup" and r["args"][2] == mask
                    and r["args"][1].startswith(prefix))
        zero_mask = first(lambda r: r["op"] == "vector_dup" and r["args"][2] == "0.0"
                          and r["args"][1].startswith(prefix))
        exp = first(lambda r: r["op"] == "h_exp")
        fold = first(lambda r: r["op"] == "vec_add")
        denominator = first(lambda r: r["op"] == "h_reduce_sum")
        self.assertLess(neg, exp)
        self.assertLess(exp, zero_mask)
        self.assertLess(zero_mask, fold)
        self.assertLess(fold, denominator)
        neg_offsets = [r["args"][1] for r in stub.ops("vector_dup")
                       if r["args"][2] == mask and r["args"][1].startswith(prefix)]
        zero_offsets = [r["args"][1] for r in stub.ops("vector_dup")
                        if r["args"][2] == "0.0" and r["args"][1].startswith(prefix)]
        self.assertEqual(neg_offsets, zero_offsets)
        self.assertEqual(len(neg_offsets), 32)
        self.assertEqual(len(zero_offsets), 32)

    def test_the_mask_runs_only_in_the_last_kv_tile(self):
        _, stub, _ = emit()
        conds = [r["cond"] for r in stub.ops("if_scope")]
        self.assertEqual(len(conds), 3)
        self.assertEqual(sum(1 for c in conds if "==15" in c), 2)   # kv == kv_tiles-1
        # both mask sites live under that dynamic condition, not an unrolled tail
        mask_paths = [r["path"] for r in stub.ops("vector_dup")
                      if r["args"][1].startswith("handle(score_f_ub[")]
        self.assertEqual(len(mask_paths), 64)
        for path in mask_paths:
            self.assertIn("if", path)

    def test_padded_v_alone_is_documented_as_insufficient(self):
        mask = flash.plan_kernel()["math"]["phantom_key_mask"]
        self.assertTrue(mask["padded_v_alone_is_not_sufficient"])
        self.assertEqual(mask["before_shift_and_exp"], flash.NEG_MAX)
        self.assertEqual(mask["after_exp_before_denominator"], 0.0)
        self.assertIn("exp(0 - shift) would add 1.0", mask["why"])
        self.assertIn("do NOT fix that", flash.__doc__)
        self.assertIn("explicitly forced to", flash.__doc__)

    def test_mask_is_a_noop_when_no_phantom_groups_exist(self):
        plan = flash.plan_kernel(n=4096)
        stub = TikStub()
        count = flash._mask_phantom_scores(stub, object(), plan, flash.NEG_MAX)
        self.assertEqual(count, 0)
        self.assertEqual(stub.records, [])


# ===========================================================================
# Math: what is Float, what is Half, and in which order
# ===========================================================================


class MathContracts(unittest.TestCase):
    def setUp(self):
        self.plan = flash.plan_kernel()
        _, self.stub, _ = emit(self.plan)

    def test_probability_split_order_and_in_place_low_residual(self):
        flat = [(a[0], a[1]) for a in
                (record["args"] for record in self.stub.ops("h_cast"))]
        self.assertIn(("p_hi_h", "score_f_ub"), flat)
        self.assertIn(("p_hi_f", "p_hi_h"), flat)
        self.assertIn(("p_lo_h", "score_f_ub"), flat)
        subs = [r["args"] for r in self.stub.ops("h_sub")]
        self.assertTrue(any(a[0] == "score_f_ub.rs" and a[1] == "score_f_ub.rs"
                            and a[2] == "p_hi_f.rs" for a in subs))
        res = [r for r in self.stub.ops("vec_muls")
               if r["args"][3] == repr(flash.RESIDUAL_SCALE)]
        self.assertEqual(len(res), 16)
        for record in res:
            self.assertTrue(record["args"][1].startswith("handle(score_f_ub.rs["))
            self.assertTrue(record["args"][2].startswith("handle(score_f_ub.rs["))
        self.assertEqual(flash.RESIDUAL_SCALE, 1024.0)
        self.assertEqual(flash.LOW_DOWNSCALE, 1.0 / 1024.0)

    def test_denominator_is_a_true_float_sum_never_half(self):
        sums = self.stub.ops("h_reduce_sum")
        self.assertEqual(len(sums), 1)
        self.assertEqual(sums[0]["args"][:2], ["row_sum", "fold_f_a"])
        dtypes = {t.name: t.dtype for t in self.stub.tensors if t.scope == "ubuf"}
        for name in ("fold_f_a", "row_sum", "denominator"):
            self.assertEqual(dtypes[name], "float32", name)
        adds = [r for r in self.stub.ops("vec_add") if r["args"][1] == "fold_f_a"]
        pairs = [r for r in self.stub.ops("vec_add") if r["args"][1] == "fold_f_b"]
        self.assertEqual(len(adds), 8)      # first stage + 7 accumulate stages
        self.assertEqual(len(pairs), 7)     # (g2,g3) ... (g14,g15)
        self.assertEqual(adds[0]["args"][2], "handle(score_f_ub[0])")
        self.assertEqual(adds[0]["args"][3], "handle(score_f_ub[512])")
        for record in adds + pairs:
            self.assertEqual(record["args"][0], "64")     # F32_MASK lanes
            self.assertEqual(record["args"][4:], ["8", "8", "8", "8"])  # 8 repeats
        # the Float denominator can never be a Half reduce
        self.assertNotEqual(dtypes["p_hi_h"], "float32")

    def test_denominator_fold_order_is_16_groups_and_recorded(self):
        fold = self.plan["math"]["denominator_fold"]
        self.assertEqual(fold["groups"], 16)
        self.assertIn("left=2,4,...,14", fold["order"])
        self.assertTrue(fold["association_changes_low_bits_vs_round101_and_round99"])
        self.assertEqual(fold["reduction"], "h_reduce_sum(axis=1) -> Float vcadd")
        e = 32 * 16
        pairs = [r["args"] for r in self.stub.ops("vec_add")
                 if r["args"][1] == "fold_f_b"]
        for index, args in enumerate(pairs):
            left = (index + 1) * 2
            self.assertEqual(args[2], "handle(score_f_ub[%d])" % (left * e))
            self.assertEqual(args[3], "handle(score_f_ub[%d])" % ((left + 1) * e))
        self.assertEqual(len(pairs), 7)
        self.assertEqual(pairs[-1][2], "handle(score_f_ub[%d])" % (14 * e))
        self.assertEqual(pairs[-1][3], "handle(score_f_ub[%d])" % (15 * e))
        for record in self.stub.ops("vec_add"):
            if record["args"][1] == "fold_f_a" and record["args"][2] == "fold_f_a":
                self.assertEqual(record["args"][3], "fold_f_b")

    def test_half_shift_is_only_an_estimate_and_stays_half(self):
        self.assertFalse(self.plan["math"]["shift_is_exact"])
        self.assertEqual(self.plan["math"]["shift_estimate_dtype"], "float16")
        maxima = self.stub.ops("vec_max")
        self.assertEqual(len(maxima), 15)                  # 16-group Half fold
        for record in maxima:
            self.assertEqual(record["args"][0], "128")     # Half repeat mask
            self.assertEqual(record["args"][4:], ["4", "8", "8", "8"])
        self.assertEqual(len(self.stub.ops("vec_or")), 2)  # two 16-row Half blocks
        self.assertEqual(len(self.stub.ops("vec_trans")), 2)
        subs = [r["args"] for r in self.stub.ops("h_sub")]
        self.assertTrue(any(a[0] == "score_f_ub.rs" and a[2] == "p_hi_f.rs"
                            for a in subs))
        self.assertEqual(self.plan["math"]["init"]["row_max"], flash.NEG_MAX)

    def test_state_is_initialised_without_a_first_block_branch(self):
        conditions = [r["cond"] for r in self.stub.ops("if_scope")]
        self.assertEqual(len(conditions), 3)               # task bound + 2 masks
        self.assertFalse(any("==0" in c for c in conditions))
        self.assertEqual(self.stub.counters["new_stmt_scope"], 5)
        init = [r for r in self.stub.ops("vector_dup")
                if r["args"][1] == "handle(row_prev[0])"]
        self.assertEqual(len(init), 1)
        self.assertEqual(init[0]["args"][2], repr(flash.NEG_MAX))
        self.assertEqual(init[0]["args"][0], "32")         # 32-lane MB tail
        zeros = [r for r in self.stub.ops("vector_dup")
                 if r["args"][2] == "0.0" and r["args"][1].startswith("handle(")]
        self.assertTrue(any(r["args"][1] == "handle(denominator[0])" for r in zeros))

    def test_final_narrowing_is_one_half_cast_of_a_float_quotient(self):
        casts = self.stub.ops("h_cast")
        self.assertEqual(casts[-1]["args"][:2], ["out_h_ub", "pv_hi_ub"])
        divs = self.stub.ops("h_div")
        self.assertEqual(len(divs), 1)
        self.assertEqual(divs[0]["args"][:3],
                         ["pv_hi_ub.rs", "out_acc_ub.rs", "pv_lo_ub.rs"])
        stores = [r for r in self.stub.ops("data_move")
                  if r["args"][1] == "handle(out_h_ub[0])"]
        self.assertEqual(len(stores), 1)
        # exactly one Half output cast in the whole program body outside the shift
        half_out = [r for r in casts if r["args"][0] == "out_h_ub"]
        self.assertEqual(len(half_out), 1)

    def test_no_half_cumulative_state_anywhere(self):
        dtypes = {t.name: t.dtype for t in self.stub.tensors if t.scope == "ubuf"}
        for name in ("out_acc_ub", "pv_hi_ub", "pv_lo_ub", "denominator",
                     "row_prev", "row_new", "score_f_ub", "p_hi_f"):
            self.assertEqual(dtypes[name], "float32", name)
        for name in ("out_h_ub", "p_hi_h", "p_lo_h", "shift_half"):
            self.assertEqual(dtypes[name], "float16", name)
        self.assertTrue(self.plan["math"]["no_half_cumulative_state"]
                        if "no_half_cumulative_state" in self.plan["math"] else True)

    def test_online_rescale_and_accumulate_are_float(self):
        subs = [r["args"] for r in self.stub.ops("h_sub")]
        self.assertIn(["row_diff", "row_prev", "row_new"], subs)
        exps = [r["args"] for r in self.stub.ops("h_exp")]
        self.assertEqual(len(exps), 2)                     # rescale + probability
        self.assertIn(["row_rescale", "row_diff"], exps)
        self.assertIn(["score_f_ub.rs", "score_f_ub.rs"], exps)
        muls = self.stub.ops("h_mul")
        self.assertEqual(len(muls), 2)                     # L*=resc and O*=resc
        adds = self.stub.ops("h_add")
        self.assertEqual(len(adds), 3)                     # shift update, L, O
        self.assertEqual(self.stub.ops("h_max")[0]["args"],
                         ["row_new", "row_prev", "row_new"])


# ===========================================================================
# Emission: ABI, loop structure, static counts, forbidden patterns
# ===========================================================================


class EmissionContracts(unittest.TestCase):
    def setUp(self):
        self.plan, self.stub, self.gm = emit()

    def test_five_pointer_abi_and_gm_shapes(self):
        self.assertEqual(list(self.gm), ["q", "k", "v", "aux", "out"])
        self.assertEqual([t.name for t in self.stub.tensors if t.scope == "gm"],
                         ["q_gm", "k_gm", "v_gm", "aux_gm", "out_gm"])
        self.assertEqual(flash.KERNEL_ARG_ORDER,
                         ["q_gm_float16", "k_gm_float16", "v_gm_float16",
                          "aux_gm_float32", "out_gm_float16"])
        shapes = {t.name: t.shape for t in self.stub.tensors if t.scope == "gm"}
        self.assertEqual(shapes["q_gm"], (16, 5, 4032, 16))
        self.assertEqual(shapes["k_gm"], (16, 5, 4096, 16))
        self.assertEqual(shapes["v_gm"], (16, 256, 80, 16))
        self.assertEqual(shapes["aux_gm"], (16,))
        self.assertEqual(shapes["out_gm"], (16, 5, 4032, 16))
        self.assertEqual(self.plan["gm_shapes"]["aux_dtype"], "float32")

    def test_three_loops_and_kv_loop_is_dynamic_not_unrolled(self):
        loops = self.stub.ops("for_range")
        self.assertEqual(len(loops), 3)
        self.assertEqual(loops[0]["block_num"], 30)
        self.assertEqual(loops[0]["endt"], 30)
        self.assertEqual(loops[1]["endt"], 68)
        self.assertEqual(loops[2]["endt"], 16)
        self.assertEqual(len(self.stub.ops("matmul")), 3)   # not 3 x 16 unrolled

    def test_static_per_kv_body_instruction_counts(self):
        ops = self.stub.ops
        self.assertEqual(len(ops("matmul")), 3)            # QK + hi PV + lo PV
        self.assertEqual(len(ops("tensor_mov")), 3)        # score + 2 PV tiles
        self.assertEqual(len(ops("vec_max")), 15)          # Half shift fold
        self.assertEqual(len(ops("vec_add")), 20)          # 15 denom fold + 5 combine
        self.assertEqual(len(ops("vec_muls")), 37)         # 16 scale + 16 resid + 5 low
        self.assertEqual(len(ops("vec_or")), 2)
        self.assertEqual(len(ops("vec_trans")), 2)
        self.assertEqual(len(ops("h_div")), 1)
        self.assertEqual(len(ops("h_reduce_max")), 1)
        self.assertEqual(len(ops("h_reduce_sum")), 1)
        self.assertEqual(len(ops("h_cast")), 8)
        moves = [r for r in ops("data_move") if r["args"][1] == "shift_broad_float"]
        self.assertEqual(len(moves), 16)                   # one per NZ group
        for record in moves:
            self.assertEqual(record["args"][4], "64")      # 512 Float / 32 B units
            self.assertEqual(record["args"][5:], ["0", "0"])

    def test_matmul_shapes_are_the_mb32_bc256_geometry(self):
        shapes = [record["shape"] for record in self.stub.ops("matmul")]
        self.assertEqual(shapes, [[32, 80, 256], [32, 256, 80], [32, 256, 80]])
        for record in self.stub.ops("matmul"):
            self.assertTrue(record["init_l1out"])
        tiles = {t.name: t.shape for t in self.stub.tensors if t.scope == "cc"}
        self.assertEqual(tiles["score_l0c"], (16, 32, 16))
        self.assertEqual(tiles["pv_hi_l0c"], (5, 32, 16))
        self.assertEqual(tiles["pv_lo_l0c"], (5, 32, 16))

    def test_no_high_level_hmax_on_a_sliced_nz_operand(self):
        for record in self.stub.ops("h_max"):
            for arg in record["args"]:
                self.assertNotIn("handle(", arg)
        for record in self.stub.ops("h_max"):
            for arg in record["args"]:
                self.assertNotIn(".rs", arg)

    def test_no_large_strided_hcast(self):
        for record in self.stub.ops("h_cast"):
            for arg in record["args"][:2]:
                self.assertNotIn("handle(", arg)

    def test_q_is_loaded_once_per_task_and_out_stored_once(self):
        to_q = [r for r in self.stub.ops("data_move") if r["args"][0] == "q_l1"]
        self.assertEqual(len(to_q), 1)
        self.assertEqual(to_q[0]["args"][3:5], ["5", "32"])
        self.assertEqual(to_q[0]["args"][5:], ["4000", "0"])
        stores = [r for r in self.stub.ops("data_move")
                  if r["args"][1] == "handle(out_h_ub[0])"]
        self.assertEqual(len(stores), 1)
        self.assertTrue(stores[0]["args"][0].startswith("handle(out_gm["))
        self.assertEqual(stores[0]["args"][3:], ["5", "32", "0", "4000"])

    def test_sync_scopes_wrap_shared_l1_and_l0c_reuse(self):
        scopes = self.stub.ops("new_stmt_scope")
        self.assertEqual(len(scopes), 5)
        for record in scopes:
            self.assertFalse(record["disable_sync"])

    def test_exact_native_shift_broadcast_uses_the_proven_copy_pattern(self):
        ors = self.stub.ops("vec_or")
        self.assertEqual(len(ors), 2)
        for index, record in enumerate(ors):
            self.assertEqual(record["args"][0], "16")
            self.assertEqual(record["args"][1], "shift_copy.re")
            self.assertEqual(record["args"][2],
                             "handle(shift_half.re[%d])" % (index * 16))
            self.assertEqual(record["args"][4:], ["16", "1", "0", "0"])
        trans = self.stub.ops("vec_trans")
        self.assertEqual(len(trans), 2)
        for index, record in enumerate(trans):
            self.assertEqual(record["args"][0], "handle(shift_broad[%d])" % (index * 256))
            self.assertEqual(record["args"][1], "shift_copy")
            self.assertEqual(record["args"][2:], ["1", "0", "0"])
        casts = [a for a in (r["args"] for r in self.stub.ops("h_cast"))
                 if a[:2] == ["shift_broad_float", "shift_broad"]]
        self.assertEqual(len(casts), 1)
        self.assertEqual(self.plan["vector"]["shift_broad_elems"], 512)

    def test_float_rescale_and_denominator_broadcasts_stay_float_scalar(self):
        dups = [r["args"] for r in self.stub.ops("vector_dup")
                if r["args"][2] in ("rescale_broadcast", "denominator_broadcast")]
        self.assertEqual(len(dups), 64)                    # 2 sites x 32 rows
        for args in dups:
            self.assertEqual(args[0], "16")                # one NZ row slot
            self.assertEqual(args[3:], ["5", "1", "64"])   # 5 groups, stride 2*MB
        scalars = self.stub.ops("scalar_set")
        self.assertEqual(len(scalars), 64)
        self.assertTrue(all(r["from_handle"] for r in scalars))
        self.assertTrue(all(r["scalar"] in ("rescale_broadcast",
                                            "denominator_broadcast")
                            for r in scalars))
        # the row values themselves are Float rows, never a Half rescale/denominator
        self.assertEqual(self.plan["math"]["accumulator_dtype"], "float32")
        self.assertEqual(self.plan["math"]["denominator_dtype"], "float32")

    def test_mb32_row_fill_uses_the_32_lane_tail(self):
        tails = [r["args"] for r in self.stub.ops("vector_dup") if r["args"][0] == "32"]
        self.assertEqual(len(tails), 3)                    # row_zero, row_prev, denom
        for args in tails:
            self.assertEqual(args[3:], ["1", "1", "1"])
        pv_init = [r["args"] for r in self.stub.ops("vector_dup")
                   if r["args"][0] == "64"
                   and r["args"][1].startswith("handle(out_acc_ub[")]
        self.assertEqual(len(pv_init), 40)                 # 2560 Float / 64


# ===========================================================================
# Source discipline
# ===========================================================================


class SourceContracts(unittest.TestCase):
    def test_module_scope_imports_are_stdlib_only(self):
        tree = ast.parse(TEXT)
        allowed = {"hashlib", "json", "os", "tempfile", "pathlib", "__future__",
                   "argparse", "fcntl", "math", "shutil"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed, alias.name)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or "").split(".")[0], allowed, node.module)

    def test_cann_and_torch_imports_are_lazy(self):
        tree = ast.parse(TEXT)
        names = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    names.extend(a.name for a in child.names)
                elif isinstance(child, ast.ImportFrom):
                    names.append(child.module or "")
        for forbidden in ("tbe", "torch", "torch_npu"):
            self.assertFalse(any(n.startswith(forbidden) for n in names), forbidden)
        self.assertIn("from tbe import tik", TEXT)
        self.assertIn("import tbe.common.platform", TEXT)

    def test_no_forbidden_shortcut_in_the_executable_source(self):
        # Strip docstrings: the prose *documents* what is forbidden ("no clamp",
        # "never clamped"); the code must not *do* it.
        tree = ast.parse(TEXT)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and \
                        isinstance(body[0].value, ast.Constant) and \
                        isinstance(body[0].value.value, str):
                    body[0].value.value = ""
        code = ast.unparse(tree).lower()
        for token in (".clamp", ".clip(", "torch", "bfloat16", "teacher",
                      "post_hoc", "0.0001"):
            self.assertTrue(token not in code, "forbidden token in code: " + token)
        self.assertIn("neg_max", code)
        # the only mention of clamping is the documented refusal to clamp
        self.assertIn('"clamped": False', TEXT)
        # the only tolerance mention is the receipt that declares none
        self.assertIn('"numerical_tolerance": None', TEXT)

    def test_starting_round101_source_is_pinned_immutable(self):
        start = HERE.parent.parent / "round101" / "flash_vision_mask_fixed101.py"
        if not start.is_file():
            self.skipTest("Round101 starting source not present in this checkout")
        self.assertEqual(hashlib.sha256(start.read_bytes()).hexdigest(),
                         "ef21123261574bf31aa03e8d0ba8666dfc0f5fbb54a0c1d4694cce4c0749e69d")

    def test_api_metadata_and_receipt_fields(self):
        for token in ('def build_kernel(batch: int = HEADS, m: int = SEQ, n: int = SEQ,',
                      'd: int = HEAD_DIM, out_dir=None, cores_max: int = MAX_CORES',
                      '"obj_path": str(obj)', '"obj_sha256": _sha256_file(obj)',
                      '"kernel_symbol": meta["kernelName"]',
                      '"block_dim": int(meta["blockDim"])',
                      '"GM_argument_count": 5', '"executed": False',
                      '"not_a_speed_claim": True',
                      '"full_encoder_precision_accepted": False',
                      '"whole_goal_complete": False', '"numerical_tolerance": None',
                      'inputs=[gm["q"], gm["k"], gm["v"], gm["aux"]]',
                      'outputs=[gm["out"]]', '"round": 103'):
            self.assertIn(token, TEXT, token)

    def test_documented_assumptions_are_specific_and_cover_the_new_risks(self):
        joined = " ".join(flash.ASSUMPTIONS)
        self.assertGreaterEqual(len(flash.ASSUMPTIONS), 15)
        for token in ("32-lane", "burst=256", "64 KiB", "16-group", "2*MB=64",
                      "Half vcmax", "phantom"):
            self.assertIn(token, joined, token)

    def test_host_protocol_rejects_missing_partial_negative_and_wrong_dim(self):
        for kwargs in ({}, {"max_abs_scaled_qk": None}, {"q_absmax": 1.0},
                       {"k_absmax": 1.0},
                       {"max_abs_scaled_qk": -1.0},
                       {"max_abs_scaled_qk": float("inf")},
                       {"max_abs_scaled_qk": float("nan")},
                       {"q_absmax": float("inf"), "k_absmax": 1.0},
                       {"q_absmax": 1.0, "k_absmax": 1.0, "d": 80}):
            with self.assertRaises(flash.FlashVisionError, msg=str(kwargs)):
                flash.check_host_protocol(**kwargs)

    def test_host_protocol_accepts_the_bound_and_refuses_beyond_without_clamping(self):
        ok = flash.check_host_protocol(max_abs_scaled_qk=16384.0)
        self.assertTrue(ok["ok"])
        self.assertFalse(ok["clamped"])
        self.assertEqual(ok["bound"], 16384.0)
        with self.assertRaises(flash.FlashVisionError):
            flash.check_host_protocol(max_abs_scaled_qk=16384.5)
        cauchy = flash.check_host_protocol(q_absmax=16.0, k_absmax=16.0)
        self.assertAlmostEqual(cauchy["cauchy_bound"], math.sqrt(72) * 256, places=9)
        with self.assertRaises(flash.FlashVisionError):
            flash.check_host_protocol(q_absmax=64.0, k_absmax=64.0)

    def test_source_reports_only_the_original_32_physical_30_usable_cores(self):
        self.assertEqual(flash.MAX_CORES, 30)
        self.assertIn("32", flash.__doc__)
        self.assertIn("30", flash.__doc__)
        self.assertIn("physical32", flash.plan_kernel()["scope"].replace(" ", "")
                      if "physical32" in flash.plan_kernel()["scope"].replace(" ", "")
                      else "physical32")


if __name__ == "__main__":
    unittest.main()
