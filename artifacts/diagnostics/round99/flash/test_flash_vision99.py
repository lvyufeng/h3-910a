#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round99 streaming flash-attention source.

No torch, no torch_npu, no tbe/CANN, no NPU, no model, no SSH: every check runs on
pure python. ``_emit_kernel`` is exercised against a recording TIK stub so the
geometry, the DMA descriptors, the operand shapes, the op order and the "no
implicit broadcast" rule are all verified without compiling anything. These are
structural contracts, not numerical or speed results.
"""
import ast
import importlib.util
import json
import math
import struct
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "flash_vision99.py"
_spec = importlib.util.spec_from_file_location("flash_vision99_under_test", SOURCE_PATH)
flash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flash)
TEXT = SOURCE_PATH.read_text()
TREE = ast.parse(TEXT)


def _half_representable(value):
    try:
        struct.pack("<e", value)
        return True
    except (OverflowError, struct.error):
        return False


# ---------------------------------------------------------------------------
# A minimal recording TIK stub: enough surface for _emit_kernel
# ---------------------------------------------------------------------------


class Expr:
    """A symbolic emission-time value (loop variable or arithmetic on one)."""

    def __init__(self, text):
        self.text = text

    def _bin(self, other, op):
        return Expr(f"({self.text}{op}{getattr(other, 'text', other)})")

    def __add__(self, other):
        return self._bin(other, "+")

    def __radd__(self, other):
        return Expr(f"({other}+{self.text})")

    def __sub__(self, other):
        return self._bin(other, "-")

    def __rsub__(self, other):
        return Expr(f"({other}-{self.text})")

    def __mul__(self, other):
        return self._bin(other, "*")

    def __rmul__(self, other):
        return Expr(f"({other}*{self.text})")

    def __floordiv__(self, other):
        return self._bin(other, "//")

    def __lt__(self, other):
        return Expr(f"({self.text}<{getattr(other, 'text', other)})")

    def __gt__(self, other):
        return Expr(f"({self.text}>{getattr(other, 'text', other)})")


class Tensor:
    """Shape-carrying stub tensor; ``reshape`` must preserve the element count."""

    def __init__(self, dtype, shape, name="", scope=None):
        self.dtype = dtype
        self.shape = tuple(int(v) for v in shape)
        self.name = name
        self.scope = scope

    @property
    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def reshape(self, shape):
        new = Tensor(self.dtype, shape, self.name, self.scope)
        if new.numel != self.numel:
            raise AssertionError(
                f"reshape {self.shape}->{new.shape} changes the element count")
        return new

    def __getitem__(self, index):
        if isinstance(index, tuple):
            shape = tuple(dim for dim, key in zip(self.shape, index)
                          if not isinstance(key, int))
            return Tensor(self.dtype, shape, self.name, self.scope)
        if isinstance(index, (int, Expr)):
            return Ref(self, index)
        raise AssertionError("unexpected index form")


class Ref:
    """An indexed destination inside a tensor (offset in elements)."""

    def __init__(self, tensor, offset):
        self.tensor = tensor
        self.offset = offset
        self.shape = tensor.shape
        self.numel = tensor.numel
        self.dtype = tensor.dtype
        self.name = tensor.name


class Recorded:
    def __init__(self, op, scope_path, **kw):
        self.op = op
        self.scope_path = scope_path
        self.kw = kw

    def __getitem__(self, key):
        return self.kw[key]


class _Ctx:
    def __init__(self, tik, label, variable=False):
        self.tik = tik
        self.label = label
        self.variable = variable

    def __enter__(self):
        self.tik.scope.append(self.label)
        return Expr(self.label) if self.variable else None

    def __exit__(self, *exc):
        self.tik.scope.pop()
        return False


class FakeTik:
    """Records every emitted instruction with its operands."""

    def __init__(self):
        self.ops = []
        self.scope = []
        self.scope_gm, self.scope_cbuf, self.scope_cc, self.scope_ubuf = (
            "gm", "cbuf", "cc", "ubuf")

    # -- construction --------------------------------------------------------
    def Tensor(self, dtype, shape, name="", scope=None):
        return Tensor(dtype, shape, name, scope)

    def Scalar(self, dtype, name=""):
        tik = self

        class _Scalar:
            def __init__(self):
                self.dtype = dtype
                self.name = name
                self.shape = (1,)
                self.loads = 0

            def set_as(self, source):
                self.loads += 1
                tik.ops.append(Recorded("scalar_load", tuple(tik.scope),
                                        scalar=self, source=source))

        return _Scalar()

    # -- control flow --------------------------------------------------------
    def for_range(self, start, stop, block_num=None):
        return _Ctx(self, f"for{start}_{stop}", variable=True)

    def if_scope(self, condition):
        return _Ctx(self, "if")

    def new_stmt_scope(self, disable_sync=False):
        return _Ctx(self, "stmt")

    # -- ops -----------------------------------------------------------------
    def _record(self, op, **kw):
        self.ops.append(Recorded(op, tuple(self.scope), **kw))
        return self.ops[-1]

    def data_move(self, dst, src, sid, nburst, burst, src_stride, dst_stride):
        self._record("data_move", dst=dst, src=src, sid=sid, nburst=nburst,
                     burst=burst, src_stride=src_stride, dst_stride=dst_stride)

    def matmul(self, dst, a, b, m, k, n, init_l1out=True, bias=None):
        self._record("matmul", dst=dst, a=a, b=b, m=m, k=k, n=n,
                     init_l1out=init_l1out, bias=bias)

    def tensor_mov(self, dst, src, block_mode, nburst, burst_len, dst_stride,
                   src_stride):
        self._record("tensor_mov", dst=dst, src=src, block_mode=block_mode,
                     nburst=nburst, burst_len=burst_len, dst_stride=dst_stride,
                     src_stride=src_stride)

    def vec_muls(self, mask, dst, src, scalar, repeat_times, dst_stride, src_stride,
                 mask_mode="normal"):
        self._record("vec_muls", mask=mask, dst=dst, src=src, scalar=scalar,
                     repeat_times=repeat_times, dst_stride=dst_stride,
                     src_stride=src_stride)

    def vec_add(self, mask, dst, src0, src1, repeat_times, dst_stride,
                src0_stride, src1_stride):
        self._record("vec_add", mask=mask, dst=dst, src0=src0, src1=src1,
                     repeat_times=repeat_times, dst_stride=dst_stride,
                     src0_stride=src0_stride, src1_stride=src1_stride)

    def vector_dup(self, mask, dst, scalar, repeat_times, dst_blk_stride,
                   dst_rep_stride):
        self._record("vector_dup", mask=mask, dst=dst, scalar=scalar,
                     repeat_times=repeat_times, dst_blk_stride=dst_blk_stride,
                     dst_rep_stride=dst_rep_stride)

    def h_cast(self, dst, src, round_mode):
        self._record("h_cast", dst=dst, src=src, round_mode=round_mode)

    def h_reduce_max(self, dst, src, axis=None):
        self._record("h_reduce_max", dst=dst, src=src, axis=axis)

    def h_reduce_sum(self, dst, src, axis=None):
        self._record("h_reduce_sum", dst=dst, src=src, axis=axis)

    def _binary(self, op, dst, src0, src1):
        self._record(op, dst=dst, src0=src0, src1=src1)

    def h_sub(self, dst, src0, src1):
        self._binary("h_sub", dst, src0, src1)

    def h_add(self, dst, src0, src1):
        self._binary("h_add", dst, src0, src1)

    def h_mul(self, dst, src0, src1):
        self._binary("h_mul", dst, src0, src1)

    def h_max(self, dst, src0, src1):
        self._binary("h_max", dst, src0, src1)

    def h_div(self, dst, src0, src1):
        self._binary("h_div", dst, src0, src1)

    def h_exp(self, dst, src):
        self._record("h_exp", dst=dst, src=src)

    def BuildCCE(self, kernel_name, inputs, outputs):
        self._record("BuildCCE", kernel_name=kernel_name, inputs=inputs,
                     outputs=outputs)


class FakeTikModule:
    scope_gm, scope_cbuf, scope_cc, scope_ubuf = "gm", "cbuf", "cc", "ubuf"


def emit(plan=None):
    tik = FakeTik()
    plan = plan or flash.plan_kernel()
    gm = flash._emit_kernel(tik, plan, FakeTikModule())
    return tik, gm, plan


# ---------------------------------------------------------------------------
# Geometry / plan contracts
# ---------------------------------------------------------------------------


class GeometryContracts(unittest.TestCase):
    def test_real_h3_vision_geometry(self):
        plan = flash.plan_kernel()
        lg, gr = plan["logical"], plan["grid"]
        self.assertEqual((flash.HEADS, flash.SEQ, flash.HEAD_DIM), (16, 4032, 72))
        self.assertEqual((lg["d1"], lg["kb1"], flash.PAD_DIM), (5, 4, 80))
        self.assertEqual((flash.MB, flash.BC), (64, 64))
        self.assertEqual(4032 % 64, 0)
        self.assertEqual((gr["m_tiles"], gr["kv_tiles"]), (63, 63))
        self.assertEqual(gr["tasks"], 16 * 63)
        self.assertEqual((gr["cores"], gr["steps"]), (30, 34))
        self.assertGreaterEqual(gr["cores"] * gr["steps"], gr["tasks"])
        self.assertTrue(gr["dynamic_kv_loop"])

    def test_smoke_geometry_uses_the_same_source(self):
        plan = flash.plan_kernel(*flash.SMOKE_GEOMETRY)
        self.assertEqual(plan["grid"]["tasks"], 2 * 2)
        self.assertEqual(plan["grid"]["cores"], 4)
        self.assertEqual(plan["grid"]["steps"], 1)
        self.assertEqual(plan["grid"]["kv_tiles"], 2)
        self.assertLessEqual(plan["grid"]["cores"], flash.MAX_CORES)

    def test_scale_is_original_head_dim_never_padded_width(self):
        plan = flash.plan_kernel()
        self.assertAlmostEqual(plan["logical"]["scale"], 72 ** -0.5, places=15)
        self.assertNotAlmostEqual(plan["logical"]["scale"], 80 ** -0.5, places=6)
        self.assertNotAlmostEqual(plan["logical"]["scale"], 128 ** -0.5, places=6)
        check = flash.scale_check(plan)
        self.assertTrue(check["matches_original"])
        self.assertFalse(check["padded_width_scale_used"])

    def test_attention_is_unmasked_and_not_causal(self):
        plan = flash.plan_kernel()
        self.assertTrue(plan["logical"]["unmasked"])
        self.assertFalse(plan["logical"]["causal"])
        self.assertNotIn("attn_mask", TEXT)
        self.assertNotIn("causal_mask", TEXT)

    def test_head_dim_padding_guard(self):
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(16, 4032, 4032, 96)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(16, 4032, 4032, 64)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(16, 4032, 4032, 72, mb=32)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(16, 4000, 4032, 72)
        with self.assertRaises(flash.FlashVisionError):
            flash.plan_kernel(16, 4032, 4000, 72)


class LayoutContracts(unittest.TestCase):
    def test_gm_shapes_match_the_five_pointer_abi(self):
        shapes = flash.plan_kernel()["gm_shapes"]
        self.assertEqual(shapes["q"], [16, 5, 4032, 16])
        self.assertEqual(shapes["k"], [16, 5, 4032, 16])
        self.assertEqual(shapes["v"], [16, 4032 // 16, 80, 16])
        self.assertEqual(shapes["out"], [16, 5, 4032, 16])
        self.assertEqual(shapes["aux"], [16])
        self.assertEqual(shapes["aux_dtype"], "float32")
        self.assertEqual(flash.KERNEL_ARG_ORDER,
                         ["q_gm_float16", "k_gm_float16", "v_gm_float16",
                          "aux_gm_float32", "out_gm_float16"])

    def test_head_strides_are_dense_and_equal_for_q_k_out(self):
        plan = flash.plan_kernel()
        hs = plan["head_stride_elems"]
        self.assertEqual(hs["q"], 5 * 4032 * 16)
        self.assertEqual(hs["k"], 5 * 4032 * 16)
        self.assertEqual(hs["out"], 5 * 4032 * 16)
        self.assertEqual(hs["v"], 252 * 80 * 16)
        for name, stride in hs.items():
            self.assertEqual(plan["gm_elems"][name], 16 * stride)
            self.assertEqual(plan["gm_bytes"][name], 16 * stride * 2)

    def test_v_is_key_major_with_80_padded_dim_rows(self):
        plan = flash.plan_kernel()
        self.assertEqual(plan["gm_shapes"]["v"][1:], [252, 80, 16])
        self.assertEqual(plan["dma"]["v_load"]["nburst"], 4)
        self.assertEqual(plan["dma"]["v_load"]["burst"], 80)


class DmaContracts(unittest.TestCase):
    def test_q_and_k_load_five_blocks_of_64_rows(self):
        dma = flash.plan_kernel()["dma"]
        for key in ("q_load", "k_load"):
            self.assertEqual(dma[key]["nburst"], 5)
            self.assertEqual(dma[key]["burst"], 64)
            self.assertEqual(dma[key]["src_stride"], 4032 - 64)
            self.assertEqual(dma[key]["dst_stride"], 0)

    def test_p_tiles_copy_four_64_unit_blocks_with_a_proven_burst(self):
        dma = flash.plan_kernel()["dma"]
        self.assertEqual(dma["p_to_l1"]["nburst"], 4)
        self.assertEqual(dma["p_to_l1"]["burst"], 64)
        # No emitted burst may exceed the 128-unit bursts the executed Round85/96
        # kernels already use on this part.
        for name, row in dma.items():
            if isinstance(row, dict) and "burst" in row:
                self.assertLessEqual(row["burst"], 128, name)

    def test_output_store_is_half_so_no_float_double_stride(self):
        dma = flash.plan_kernel()["dma"]
        self.assertEqual(dma["out_store"]["nburst"], 5)
        self.assertEqual(dma["out_store"]["burst"], 64)
        self.assertEqual(dma["out_store"]["dst_stride"], 4032 - 64)
        self.assertNotEqual(dma["out_store"]["dst_stride"], 2 * (4032 - 64))

    def test_l0c_to_ub_burst_is_the_whole_tile_in_kb(self):
        dma = flash.plan_kernel()["dma"]
        self.assertEqual(dma["score_l0c_to_ub"]["burst_len_kb"], 16)
        self.assertEqual(dma["pv_l0c_to_ub"]["burst_len_kb"], 20)
        for key in ("score_l0c_to_ub", "pv_l0c_to_ub"):
            self.assertEqual(dma[key]["nburst"], 1)
            self.assertEqual(dma[key]["block_mode"], "m")


class BudgetContracts(unittest.TestCase):
    def test_buffers_fit_the_part(self):
        buf = flash.plan_kernel()["buffers"]
        self.assertLessEqual(buf["l1_total_bytes"], 512 * 1024)
        self.assertLessEqual(buf["l0c_total_bytes"], 256 * 1024)
        self.assertLessEqual(buf["ub_total_bytes"], flash.PINNED_UB_SAFE_BYTES)
        self.assertEqual(buf["l1_total_bytes"], 3 * 10240 + 2 * 8192)
        self.assertEqual(buf["l0c_total_bytes"], 16384 + 2 * 20480)
        self.assertLessEqual(buf["ub_total_bytes"], 160 * 1024)

    def test_repeat_immediates_and_alignment(self):
        vec = flash.plan_kernel()["vector"]
        self.assertEqual(vec["group_repeats"], 16)
        self.assertLessEqual(vec["group_repeats"], 255)
        self.assertEqual(vec["broadcast_mask"], 16)
        self.assertEqual(vec["broadcast_mask"] * 4 % 32, 0)
        self.assertEqual(vec["broadcast_rep_stride"], 128)
        self.assertLessEqual(vec["broadcast_rep_stride"], 255)
        self.assertEqual(vec["f32_mask"], 64)
        self.assertEqual(vec["f32_rep_stride"], 8)


class HostProtocolContracts(unittest.TestCase):
    def test_missing_partial_negative_or_wrong_dimension_evidence_fails_closed(self):
        for kwargs in ({}, {'q_absmax': 1.0}, {'max_abs_scaled_qk': -1.0},
                       {'q_absmax': -1.0, 'k_absmax': 1.0}, {'max_abs_scaled_qk': 1.0, 'd': 128}):
            with self.assertRaises(flash.FlashVisionError):
                flash.check_host_protocol(**kwargs)

    def test_bound_and_half_safety(self):
        self.assertEqual(flash.SHIFT_BOUND, 16384.0)
        self.assertEqual(flash.SHIFT_HALF_ULP_AT_BOUND, 8.0)
        self.assertAlmostEqual(flash.EXP_MAX_AT_BOUND, math.exp(8.0), places=9)
        self.assertTrue(_half_representable(flash.EXP_MAX_AT_BOUND))
        self.assertTrue(_half_representable(math.exp(-8.0)))
        self.assertGreater(flash.EXP_MAX_AT_BOUND, 1.0)

    def test_protocol_accepts_the_bound_and_refuses_beyond(self):
        self.assertTrue(flash.check_host_protocol(flash.SHIFT_BOUND)["ok"])
        self.assertTrue(flash.check_host_protocol(0.0)["ok"])
        for bad in (flash.SHIFT_BOUND * 1.0001, 65000.0, float("inf"), float("nan")):
            with self.assertRaises(flash.FlashVisionError):
                flash.check_host_protocol(bad)

    def test_cauchy_path(self):
        ok = flash.check_host_protocol(q_absmax=8.0, k_absmax=8.0)
        self.assertLess(ok["cauchy_bound"], flash.SHIFT_BOUND)
        with self.assertRaises(flash.FlashVisionError):
            flash.check_host_protocol(q_absmax=2000.0, k_absmax=2000.0)

    def test_no_clamp_anywhere_in_the_source(self):
        self.assertFalse(flash.check_host_protocol(1.0)["clamped"])
        self.assertNotIn("clamp(", TEXT)

    def test_plan_records_the_protocol(self):
        proto = flash.plan_kernel()["host_protocol"]
        self.assertEqual(proto["bound"], flash.SHIFT_BOUND)
        self.assertFalse(proto["clamped"])
        self.assertEqual(proto["enforced_by"], "check_host_protocol")


# ---------------------------------------------------------------------------
# Emitted program contracts (recording TIK stub)
# ---------------------------------------------------------------------------


class EmissionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tik, cls.gm, cls.plan = emit()

    def kv_ops(self):
        return [op for op in self.tik.ops
                if any(part.startswith("for0_63") for part in op.scope_path)]

    def test_returned_gm_operands_match_the_five_pointer_abi(self):
        self.assertEqual(sorted(self.gm), ["aux", "k", "out", "q", "v"])
        self.assertEqual([self.gm[n].name for n in ("q", "k", "v", "aux", "out")],
                         ["q_gm", "k_gm", "v_gm", "aux_gm", "out_gm"])
        self.assertEqual([self.gm[n].dtype for n in ("q", "k", "v", "aux", "out")],
                         ["float16", "float16", "float16", "float32", "float16"])
        self.assertEqual(self.gm["q"].shape, (16, 5, 4032, 16))
        self.assertEqual(self.gm["v"].shape, (16, 252, 80, 16))
        self.assertEqual(self.gm["aux"].numel, 16)
        # The build call must list four explicit inputs (so auto-collection cannot
        # prune the unused Float aux) and the single output.
        self.assertIn('inputs=[gm["q"], gm["k"], gm["v"], gm["aux"]]', TEXT)
        self.assertIn('outputs=[gm["out"]]', TEXT)

    def test_three_matmul_products_per_kv_block(self):
        matmuls = [op for op in self.kv_ops() if op.op == "matmul"]
        self.assertEqual(len(matmuls), 3)
        score, pv_hi, pv_lo = matmuls
        self.assertEqual((score["m"], score["k"], score["n"]), (64, 80, 64))
        self.assertEqual((pv_hi["m"], pv_hi["k"], pv_hi["n"]), (64, 64, 80))
        self.assertEqual((pv_lo["m"], pv_lo["k"], pv_lo["n"]), (64, 64, 80))
        for op in matmuls:
            self.assertTrue(op["init_l1out"])
            self.assertIsNone(op["bias"])
            self.assertEqual(op["a"].shape, (op["k"] // 16, op["m"], 16))
            self.assertEqual(op["b"].shape, (op["k"] // 16, op["n"], 16))
            self.assertEqual(op["dst"].shape, (op["n"] // 16, op["m"], 16))
            self.assertEqual(op["a"].dtype, "float16")
            self.assertEqual(op["dst"].dtype, "float32")

    def test_l0c_to_ub_moves_cover_the_whole_tile(self):
        moves = [op for op in self.tik.ops if op.op == "tensor_mov"]
        self.assertEqual(len(moves), 3)
        for op in moves:
            self.assertEqual(op["dst"].dtype, "float32")
            self.assertEqual(op["dst"].numel, op["burst_len"] * 1024 // 4)
            self.assertEqual(op["src"].numel, op["dst"].numel)

    def test_every_data_move_covers_exactly_its_tile(self):
        plan = self.plan
        tile_bytes = {"q_l1": plan["buffers"]["l1_q_bytes"],
                      "k_l1": plan["buffers"]["l1_k_bytes"],
                      "v_l1": plan["buffers"]["l1_v_bytes"],
                      "p_hi_l1": plan["buffers"]["l1_p_hi_bytes"],
                      "p_lo_l1": plan["buffers"]["l1_p_lo_bytes"]}
        moves = [op for op in self.tik.ops if op.op == "data_move"]
        self.assertTrue(moves)
        seen = {}
        for op in moves:
            covered = op["nburst"] * op["burst"] * 32
            name = op["dst"].name
            seen[name] = seen.get(name, 0) + 1
            if name in tile_bytes:
                self.assertEqual(covered, tile_bytes[name], name)
            elif name == "out_gm":
                # The store is a slice of the per-head GM extent, not the whole head.
                self.assertEqual(covered, 5 * 64 * 32)
                self.assertEqual(op["dst"].dtype, "float16")
                self.assertEqual(op["dst_stride"], 4032 - 64)
            else:
                self.fail(f"unexpected data_move destination {name}")
        # Six moves in the emitted body: Q once per task, K/V/P_hi/P_lo once per
        # dynamic KV block, and the final store.
        self.assertEqual(seen, {"q_l1": 1, "k_l1": 1, "v_l1": 1, "p_hi_l1": 1,
                                "p_lo_l1": 1, "out_gm": 1})

    def test_half_shift_uses_half_reduce_and_float_denominator_is_a_sum(self):
        reduces_max = [op for op in self.kv_ops() if op.op == "h_reduce_max"]
        reduces_sum = [op for op in self.kv_ops() if op.op == "h_reduce_sum"]
        self.assertEqual(len(reduces_max), 1)
        self.assertEqual(len(reduces_sum), 1)
        self.assertEqual(reduces_max[0]["src"].dtype, "float16")
        self.assertEqual(reduces_max[0]["dst"].dtype, "float16")
        self.assertEqual(reduces_max[0]["axis"], 1)
        self.assertEqual(reduces_sum[0]["src"].dtype, "float32")
        self.assertEqual(reduces_sum[0]["dst"].dtype, "float32")
        self.assertEqual(reduces_sum[0]["axis"], 1)
        self.assertEqual(reduces_max[0]["src"].shape, (64, 16))
        self.assertEqual(reduces_sum[0]["src"].shape, (64, 16))

    def test_exp_is_float(self):
        exps = [op for op in self.kv_ops() if op.op == "h_exp"]
        self.assertTrue(exps)
        for op in exps:
            self.assertEqual(op["dst"].dtype, "float32")
            self.assertEqual(op["src"].dtype, "float32")

    def test_no_implicit_broadcast_every_h_binary_op_is_same_shape(self):
        for op in self.tik.ops:
            if op.op in ("h_sub", "h_add", "h_mul", "h_max", "h_div"):
                shapes = {tuple(t.shape) for t in (op["dst"], op["src0"], op["src1"])}
                dtypes = {t.dtype for t in (op["dst"], op["src0"], op["src1"])}
                self.assertEqual(len(shapes), 1, f"{op.op} mixes shapes {shapes}")
                self.assertEqual(len(dtypes), 1)

    def test_final_narrowing_is_one_half_cast_of_a_float_quotient(self):
        self.assertFalse(any(op.op == "h_div" for op in self.kv_ops()))
        divs = [i for i, op in enumerate(self.tik.ops) if op.op == "h_div"]
        self.assertEqual(len(divs), 1)
        div = self.tik.ops[divs[0]]
        self.assertEqual((div["dst"].dtype, div["src0"].dtype, div["src1"].dtype),
                         ("float32", "float32", "float32"))
        casts_after = [op for op in self.tik.ops[divs[0]:]
                       if op.op == "h_cast" and op["dst"].dtype == "float16"]
        self.assertEqual(len(casts_after), 1)
        self.assertEqual(casts_after[0]["src"].dtype, "float32")
        self.assertEqual(casts_after[0]["src"].numel, 5 * 64 * 16)
        self.assertEqual(casts_after[0]["round_mode"], "none")

    def test_no_half_cumulative_accumulator(self):
        for op in self.tik.ops:
            if op.op == "h_add" and op["dst"].dtype == "float16":
                self.fail("a Half h_add would be a Half cumulative accumulator")
        half_names = {op["dst"].name for op in self.tik.ops
                      if op.op == "h_cast" and op["dst"].dtype == "float16"}
        self.assertEqual(half_names, {"p_hi_h", "p_lo_h", "out_h_ub"})
        reduce_names = {op["dst"].name for op in self.tik.ops
                        if op.op == "h_reduce_max"}
        self.assertEqual(reduce_names, {"shift_half"})

    def test_row_broadcast_expansion_tiles_the_nz_groups_exactly(self):
        dups = [op for op in self.tik.ops if op.op == "vector_dup"]
        broadcast = [op for op in dups
                     if isinstance(op["dst"], Ref) and op["dst_rep_stride"] == 128]
        self.assertEqual(len(broadcast), 3 * 64)
        for op in broadcast:
            self.assertEqual(op["mask"], 16)
            self.assertEqual(op["dst_blk_stride"], 1)
            end = (op["dst"].offset + (op["repeat_times"] - 1)
                   * op["dst_rep_stride"] * 8 + op["mask"])
            self.assertLessEqual(end, op["dst"].numel)
        # The last row of every expansion must land exactly on the tile end: that
        # is what proves the 2*MB repeat stride tiles the (groups, MB, 16) layout.
        last_row = [op for op in broadcast if op["dst"].offset == 63 * 16]
        self.assertEqual(len(last_row), 3)
        for op in last_row:
            end = (op["dst"].offset + (op["repeat_times"] - 1)
                   * op["dst_rep_stride"] * 8 + op["mask"])
            self.assertEqual(end, op["dst"].numel)
        # Two sites expand over the 4-group score tile, one over the 5-group PV tile.
        groups = sorted(op["repeat_times"] for op in last_row)
        self.assertEqual(groups, [4, 5, 5])

    def test_constant_fills_use_one_repeat_per_call(self):
        dups = [op for op in self.tik.ops if op.op == "vector_dup"]
        fills = [op for op in dups if op["repeat_times"] == 1]
        # row_zero + row_prev + denominator + output accumulator (80 repeats).
        self.assertEqual(len(fills), 3 + 80)
        for op in fills:
            self.assertEqual(op["mask"], 64)
            self.assertLessEqual(op["dst"].offset + op["mask"], op["dst"].numel)

    def test_kv_loop_is_dynamic_and_not_unrolled(self):
        kv_ops = self.kv_ops()
        self.assertTrue(kv_ops)
        self.assertLess(len(kv_ops), 400)

    def test_grouped_vector_ops_use_the_proven_slices(self):
        muls = [op for op in self.tik.ops if op.op == "vec_muls"]
        self.assertTrue(muls)
        for op in muls:
            self.assertEqual(op["mask"], 64)
            self.assertEqual(op["repeat_times"], 16)
            self.assertEqual(op["dst_stride"], 8)
            self.assertEqual(op["src_stride"], 8)
            self.assertEqual(op["dst"].shape, (64, 16))
        adds = [op for op in self.tik.ops if op.op == "vec_add"]
        self.assertTrue(adds)
        for op in adds:
            self.assertEqual(op["repeat_times"], 16)
            self.assertEqual(op["dst"].shape, (64, 16))


class SourceContracts(unittest.TestCase):
    def test_source_has_no_forbidden_shortcut(self):
        for forbidden in ("import torch", "torch_npu", "teacher_weight",
                          "float32_linear", "bfloat16", "torch.softmax",
                          "dist.all_reduce", "np.load", "requests"):
            self.assertNotIn(forbidden, TEXT)
        self.assertIn("SOURCE ONLY", TEXT)
        self.assertIn("no_clamp", TEXT)
        self.assertIn("no_implicit_broadcast", TEXT)

    def test_module_scope_imports_are_stdlib_only(self):
        top = set()
        for node in TREE.body:
            if isinstance(node, ast.Import):
                top.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        self.assertEqual(top, {"__future__", "hashlib", "json", "os", "tempfile",
                               "pathlib"})

    def test_cann_is_lazy(self):
        for node in TREE.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None) or ""
                self.assertFalse(module.startswith("tbe"))
        self.assertIn("from tbe import tik", TEXT)
        self.assertIn("import tbe.common.platform as platform", TEXT)

    def test_capability_probe_requires_the_right_intrinsics(self):
        required = {name for name, _dtype in flash.REQUIRED_INTRINSICS}
        self.assertIn("Intrinsic_vcmax", required)
        self.assertIn("Intrinsic_vcadd", required)
        self.assertIn(("Intrinsic_vcmax", "float16"), flash.REQUIRED_INTRINSICS)
        self.assertNotIn(("Intrinsic_vcmax", "float32"), flash.REQUIRED_INTRINSICS)
        self.assertIn(("Intrinsic_vcmax", "float32"), flash.OPTIONAL_INTRINSICS)

    def _with_fake_platform(self, runner, soc="Ascend910", supported=()):
        import sys
        from types import SimpleNamespace
        table = {"SHORT_SOC_VERSION": soc, "CORE_NUM": 32, "UB_SIZE": 262144,
                 "L1_SIZE": 1048576, "L0C_SIZE": 262144}
        fake_platform = SimpleNamespace(
            set_current_compile_soc_info=lambda value: None,
            get_soc_spec=lambda key: table[key],
            intrinsic_check_support=lambda name, dtype: (name, dtype) in supported)
        fake_tbe = SimpleNamespace(common=SimpleNamespace(platform=fake_platform))
        saved = {k: sys.modules.get(k)
                 for k in ("tbe", "tbe.common", "tbe.common.platform")}
        try:
            sys.modules["tbe"] = fake_tbe
            sys.modules["tbe.common"] = fake_tbe.common
            sys.modules["tbe.common.platform"] = fake_platform
            return runner()
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_capability_probe_fails_closed_on_a_missing_instruction(self):
        def runner():
            with self.assertRaises(flash.FlashVisionError):
                flash.capability_report()
        self._with_fake_platform(runner, supported=())

    def test_capability_probe_accepts_a_supported_target(self):
        supported = {("Intrinsic_vcmax", "float16"), ("Intrinsic_vcadd", "float32"),
                     ("Intrinsic_vmax", "float32"), ("Intrinsic_vadd", "float32"),
                     ("Intrinsic_vmul", "float32"), ("Intrinsic_vexp", "float32"),
                     ("Intrinsic_vdiv", "float32")}

        def runner():
            report = flash.capability_report()
            self.assertTrue(report["all_required_supported"])
            self.assertEqual(report["compile_soc"], "Ascend910")
            self.assertEqual(report["launched_usable_cores"], 30)
            self.assertFalse(report["intrinsic_checks"]["Intrinsic_vcmax:float32"])
            return report
        self._with_fake_platform(runner, supported=supported)

    def test_wrong_soc_is_refused(self):
        def runner():
            with self.assertRaises(flash.FlashVisionError):
                flash.capability_report()
        self._with_fake_platform(runner, soc="Ascend910B", supported=())

    def test_receipts_claim_nothing(self):
        for token in ('"executed": False', '"compiled_now": compiled_now',
                      '"not_a_speed_claim": True', '"whole_goal_complete": False',
                      '"full_encoder_precision_accepted": False',
                      '"numerical_tolerance": None'):
            self.assertIn(token, TEXT)

    def test_every_documented_assumption_names_a_verification(self):
        self.assertGreaterEqual(len(flash.ASSUMPTIONS), 6)
        self.assertEqual(list(flash.ASSUMPTIONS), flash.plan_kernel()["assumptions"])


class ManifestContracts(unittest.TestCase):
    def test_manifest_hashes_match_the_delivered_files(self):
        manifest_path = HERE / "manifest99.json"
        self.assertTrue(manifest_path.is_file(), "run make_manifest99.py")
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        self.assertEqual(sorted(files),
                         sorted(["flash_vision99.py", "test_flash_vision99.py",
                                 "README.md"]))
        for name, receipt in files.items():
            payload = (HERE / name).read_bytes()
            self.assertEqual(receipt["bytes"], len(payload), name)
            self.assertEqual(receipt["sha256"],
                             __import__("hashlib").sha256(payload).hexdigest(), name)

    def test_manifest_records_the_plan_and_stays_non_claiming(self):
        manifest = json.loads((HERE / "manifest99.json").read_text())
        plan = flash.plan_kernel()
        self.assertFalse(manifest["executed"])
        self.assertFalse(manifest["compiled_now"])
        self.assertTrue(manifest["not_a_speed_claim"])
        self.assertTrue(manifest["source_only"])
        self.assertFalse(manifest["full_encoder_precision_accepted"])
        self.assertFalse(manifest["whole_goal_complete"])
        self.assertIsNone(manifest["numerical_tolerance"])
        self.assertFalse(manifest["default_enabled"])
        self.assertEqual(manifest["geometry"]["real"]["tasks"],
                         plan["grid"]["tasks"])
        self.assertEqual(manifest["geometry"]["real"]["cores"],
                         plan["grid"]["cores"])
        self.assertEqual(manifest["host_protocol"]["bound"], flash.SHIFT_BOUND)
        self.assertEqual(manifest["assumptions"], plan["assumptions"])
        self.assertEqual(manifest["gm_abi"]["argument_order"],
                         list(flash.KERNEL_ARG_ORDER))
        self.assertEqual(manifest["gm_abi"]["count"], 5)
        self.assertTrue(manifest["gm_abi"]["aux_is_unused_float_placeholder"])

    def test_manifest_records_the_sdk_evidence_that_shaped_the_design(self):
        manifest = json.loads((HERE / "manifest99.json").read_text())
        evidence = manifest["evidence_read_only"]
        self.assertTrue(evidence["h_broadcast_absent"])
        self.assertFalse(evidence["float_vcmax_unsupported"] is False)
        self.assertTrue(evidence["half_vcmax_supported"])
        self.assertGreaterEqual(len(manifest["root_smoke_must_check"]), 5)

    def test_sdk_sheet_matches_the_code_constants(self):
        sheet_path = HERE.parent / "actual_sdk_highlevel99.json"
        self.assertTrue(sheet_path.is_file())
        sheet = json.loads(sheet_path.read_text())
        self.assertIsNone(sheet["api_signatures"]["h_broadcast"])
        self.assertFalse(sheet["intrinsic_checks"]["Intrinsic_vcmax:float32"])
        self.assertTrue(sheet["intrinsic_checks"]["Intrinsic_vcmax:float16"])
        self.assertEqual(sheet["compile_soc"], flash.PINNED_COMPILE_SOC)
        self.assertEqual(flash.PINNED_COMPILE_SOC, "Ascend910")


if __name__ == "__main__":
    unittest.main()
