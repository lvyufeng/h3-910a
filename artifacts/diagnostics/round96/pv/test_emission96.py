#!/usr/bin/env python3
"""Stdlib-only emission tests: run the Round96 TIK emitter against a recording stub.

The real emitter needs ``tbe.tik``; this stub provides exactly the surface
``_emit_kernel`` uses and records every emitted operation together with its
arguments. That lets the whole kernel body be reviewed as data - operand order,
accumulator count, K-loop init flags, L0C->UB order, the Float32 combine and the
Half store geometry - without any CANN, torch or NPU work.

These are structural contracts about emitted IR, not compiled kernels, not
numerical results and not speed claims.
"""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
PV_PATH = HERE / "pv_high_low96.py"


def _load_pv():
    spec = importlib.util.spec_from_file_location("pv_high_low96", PV_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pv_high_low96"] = module
    spec.loader.exec_module(module)
    return module


pv = _load_pv()


#: Concrete sample values for the two dynamic loop variables. The stub evaluates
#: every emitted address with them, so the kernel body can be checked numerically.
SAMPLE = {"core": 3, "step": 1}


class Expr:
    """Symbolic scalar carrying both a text form and a concrete sample value."""

    __slots__ = ("text", "value")

    def __init__(self, text, value=0):
        self.text, self.value = str(text), int(value)

    def _wrap(self, other):
        return other if isinstance(other, Expr) else Expr(other, other)

    def __add__(self, other):
        other = self._wrap(other)
        return Expr(f"({self.text}+{other.text})", self.value + other.value)

    def __radd__(self, other):
        other = self._wrap(other)
        return Expr(f"({other.text}+{self.text})", other.value + self.value)

    def __sub__(self, other):
        other = self._wrap(other)
        return Expr(f"({self.text}-{other.text})", self.value - other.value)

    def __mul__(self, other):
        other = self._wrap(other)
        return Expr(f"({self.text}*{other.text})", self.value * other.value)

    def __rmul__(self, other):
        other = self._wrap(other)
        return Expr(f"({other.text}*{self.text})", other.value * self.value)

    def __floordiv__(self, other):
        other = self._wrap(other)
        if other.value == 0:
            raise AssertionError("division by a zero sample value")
        return Expr(f"({self.text}//{other.text})", self.value // other.value)

    def __lt__(self, other):
        other = self._wrap(other)
        return Expr(f"({self.text}<{other.text})", int(self.value < other.value))

    def __repr__(self):
        return self.text


class Addr:
    __slots__ = ("tensor", "offset")

    def __init__(self, tensor, offset):
        self.tensor = tensor
        self.offset = offset


class Piece:
    __slots__ = ("tensor", "group")

    def __init__(self, tensor, group):
        self.tensor = tensor
        self.group = group


class Tensor:
    def __init__(self, dtype, shape, name, scope):
        self.dtype, self.shape, self.name, self.scope = dtype, tuple(shape), name, scope

    def __getitem__(self, key):
        if isinstance(key, tuple):
            group = key[0]
            if isinstance(group, int):
                return Piece(self, group)
            raise AssertionError(f"unexpected slice group {key!r}")
        return Addr(self, key)

    def __repr__(self):
        return f"Tensor({self.dtype},{self.shape},{self.name})"


class Recorder:
    """Records emitted ops; scopes and loops run their body exactly once."""

    def __init__(self):
        self.tensors = []
        self.ops = []
        self._loop_depth = 0

    # -- declarations ---------------------------------------------------
    def Tensor(self, dtype, shape, name, scope):  # noqa: N802 - mirrors the SDK
        tensor = Tensor(dtype, shape, name, scope)
        self.tensors.append(tensor)
        return tensor

    # -- control flow ---------------------------------------------------
    class _Scope:
        def __init__(self, kind, detail=None):
            self.kind, self.detail = kind, detail

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def for_range(self, start, stop, block_num=None):
        name = "core" if self._loop_depth == 0 else "step"
        self._loop_depth += 1
        self._last_loop = name
        self._loop_depth -= 1
        return self._LoopCtx(self, Expr(name, SAMPLE[name]))

    class _LoopCtx:
        def __init__(self, recorder, var):
            self.recorder, self.var = recorder, var

        def __enter__(self):
            self.recorder._loop_depth += 1
            return self.var

        def __exit__(self, *exc):
            self.recorder._loop_depth -= 1
            return False

    def if_scope(self, cond):
        return self._Scope("if", cond)

    def new_stmt_scope(self, disable_sync=False):
        return self._Scope("stmt", disable_sync)

    # -- instructions ---------------------------------------------------
    @staticmethod
    def _name(operand):
        if isinstance(operand, Addr):
            return operand.tensor.name
        return getattr(operand, "name", operand)

    def data_move(self, dst, src, sid, nburst, burst, src_stride, dst_stride):
        self.ops.append({"op": "data_move", "dst": self._name(dst),
                         "src": self._name(src),
                         "src_offset": getattr(src, "offset", None),
                         "dst_offset": getattr(dst, "offset", None),
                         "nburst": nburst, "burst": burst,
                         "src_stride": src_stride, "dst_stride": dst_stride})

    def matmul(self, dst, a, b, m, k, n, init_l1out=False):
        self.ops.append({"op": "matmul", "dst": dst.name, "a": a.name, "b": b.name,
                         "m": m, "k": k, "n": n, "init_l1out": init_l1out})

    def tensor_mov(self, dst, src, block_mode, nburst, burst_len_kb, src_stride,
                   dst_stride):
        self.ops.append({"op": "tensor_mov", "dst": dst.name, "src": src.name,
                         "block_mode": block_mode, "nburst": nburst,
                         "burst_len_kb": burst_len_kb, "src_stride": src_stride,
                         "dst_stride": dst_stride})

    def vec_muls(self, mask, dst, src, scalar, repeat_times, dst_rep_stride,
                 src_rep_stride):
        self.ops.append({"op": "vec_muls", "mask": mask, "dst": dst.tensor.name,
                         "dst_group": dst.group, "src": src.tensor.name,
                         "src_group": src.group, "scalar": scalar,
                         "repeat_times": repeat_times,
                         "dst_rep_stride": dst_rep_stride,
                         "src_rep_stride": src_rep_stride})

    def vec_add(self, mask, dst, src0, src1, repeat_times, dst_rep_stride,
                src0_rep_stride, src1_rep_stride):
        self.ops.append({"op": "vec_add", "mask": mask, "dst": dst.tensor.name,
                         "dst_group": dst.group, "src0": src0.tensor.name,
                         "src0_group": src0.group, "src1": src1.tensor.name,
                         "src1_group": src1.group, "repeat_times": repeat_times,
                         "dst_rep_stride": dst_rep_stride,
                         "src0_rep_stride": src0_rep_stride,
                         "src1_rep_stride": src1_rep_stride})

    def h_cast(self, dst, src, round_mode):
        self.ops.append({"op": "h_cast", "dst": dst.name, "src": src.name,
                         "round_mode": round_mode})

    # -- helpers --------------------------------------------------------
    def kinds(self, kind):
        return [op for op in self.ops if op["op"] == kind]


def _fake_tik():
    module = types.ModuleType("fake_tik")
    module.scope_gm, module.scope_cbuf = "gm", "cbuf"
    module.scope_cc, module.scope_ubuf = "cc", "ubuf"
    return module


def _emit(shape):
    plan = pv.plan_tiling(*shape)
    pv._check_budget(plan)
    recorder = Recorder()
    gm = pv._emit_kernel(recorder, plan, _fake_tik())
    return plan, recorder, gm


class EmittedOperandContracts(unittest.TestCase):
    def test_five_gm_operands_in_the_declared_abi_order(self):
        plan, rec, gm = _emit(pv.TARGET_SHAPES["pv_full_chunk"])
        self.assertEqual([t.name for t in rec.tensors if t.scope == "gm"],
                         ["hi_gm", "lo_gm", "v_gm", "aux_gm", "out_gm"])
        self.assertEqual(list(gm), ["hi", "lo", "v", "aux", "out"])
        self.assertEqual([gm[key].name for key in gm],
                         ["hi_gm", "lo_gm", "v_gm", "aux_gm", "out_gm"])
        gm_tensors = [t for t in rec.tensors if t.scope == "gm"]
        self.assertEqual([t.dtype for t in gm_tensors],
                         ["float16", "float16", "float16", "float32", "float16"])
        self.assertEqual(gm_tensors[0].shape, (16, 256, 512, 16))
        self.assertEqual(gm_tensors[1].shape, (16, 256, 512, 16))
        self.assertEqual(gm_tensors[2].shape, (16, 256, 128, 16))
        self.assertEqual(gm_tensors[3].shape, (16,))
        self.assertEqual(gm_tensors[4].shape, (16, 8, 512, 16))

    def test_two_float32_l0c_accumulators_and_three_ub_tiles(self):
        _plan, rec, _gm = _emit(pv.TARGET_SHAPES["pv_full_chunk"])
        l0c = [t for t in rec.tensors if t.scope == "cc"]
        ub = [t for t in rec.tensors if t.scope == "ubuf"]
        l1 = [t for t in rec.tensors if t.scope == "cbuf"]
        self.assertEqual([t.name for t in l0c], ["c_hi_l0c", "c_lo_l0c"])
        self.assertTrue(all(t.dtype == "float32" for t in l0c))
        self.assertTrue(all(t.shape == (8, 128, 16) for t in l0c))
        self.assertEqual([t.name for t in ub], ["u_hi_ub", "u_lo_ub", "u_h_ub"])
        self.assertEqual([t.dtype for t in ub], ["float32", "float32", "float16"])
        self.assertEqual([t.name for t in l1], ["a_hi_l1", "a_lo_l1", "b_l1"])
        self.assertTrue(all(t.dtype == "float16" for t in l1))
        self.assertTrue(all(t.shape == (8, 128, 16) for t in l1))


class EmittedBodyContracts(unittest.TestCase):
    def setUp(self):
        self.plan, self.rec, _gm = _emit(pv.TARGET_SHAPES["pv_full_chunk"])

    def test_every_k_block_loads_both_a_operands_once_and_issues_two_matmul(self):
        moves = [op for op in self.rec.kinds("data_move")
                 if op["dst"] in ("a_hi_l1", "a_lo_l1", "b_l1")]
        matmuls = self.rec.kinds("matmul")
        self.assertEqual(len(moves), 3 * 32)
        self.assertEqual(len(matmuls), 2 * 32)
        self.assertEqual(sum(1 for op in moves if op["dst"] == "a_hi_l1"), 32)
        self.assertEqual(sum(1 for op in moves if op["dst"] == "a_lo_l1"), 32)
        self.assertEqual(sum(1 for op in moves if op["dst"] == "b_l1"), 32)
        # The V tile is loaded once per K block and read by BOTH products.
        for index in range(32):
            block = matmuls[index * 2:index * 2 + 2]
            self.assertEqual([op["b"] for op in block], ["b_l1", "b_l1"])
            self.assertEqual([op["dst"] for op in block], ["c_hi_l0c", "c_lo_l0c"])

    def test_only_the_first_k_block_initialises_each_accumulator(self):
        matmuls = self.rec.kinds("matmul")
        flags = [op["init_l1out"] for op in matmuls]
        self.assertEqual(len(flags), 64)
        self.assertEqual(flags[:4], [True, True, False, False])
        self.assertEqual(sum(1 for flag in flags if flag), 2)
        self.assertTrue(all(isinstance(flag, bool) for flag in flags))
        for index in range(32):
            self.assertEqual(flags[2 * index], index == 0)
            self.assertEqual(flags[2 * index + 1], index == 0)

    def test_dma_geometry_matches_the_tiling_plan(self):
        plan = self.plan
        for op in self.rec.kinds("data_move"):
            if op["dst"] == "a_hi_l1":
                self.assertEqual((op["nburst"], op["burst"], op["src_stride"],
                                  op["dst_stride"]), (8, 128, 384, 0))
            elif op["dst"] == "b_l1":
                self.assertEqual((op["nburst"], op["burst"], op["src_stride"],
                                  op["dst_stride"]), (8, 128, 0, 0))
            elif op["dst"] == "out_gm" and op["src"] == "u_h_ub":
                self.assertEqual((op["nburst"], op["burst"], op["src_stride"],
                                  op["dst_stride"]), (8, 128, 0, 384))
                self.assertEqual(op["nburst"] * op["burst"], 1024)
                self.assertEqual(op["dst_stride"],
                                 plan["padded"]["m_pad"] - plan["tiles"]["mb"])
        self.assertEqual(self.rec.kinds("data_move")[-1]["dst"], "out_gm")

    def test_half_store_is_not_the_fp32_store(self):
        store = self.rec.kinds("data_move")[-1]
        self.assertEqual(store["burst"], 128)          # FP32 would be 256
        self.assertEqual(store["dst_stride"], 384)     # FP32 would be 768
        self.assertNotEqual(store["dst_stride"], 2 * (512 - 128))
        self.assertEqual(store["src"], "u_h_ub")
        self.assertEqual(store["nburst"], 8)

    def test_low_product_moves_to_ub_before_the_high_product(self):
        movs = self.rec.kinds("tensor_mov")
        self.assertEqual([op["src"] for op in movs], ["c_lo_l0c", "c_hi_l0c"])
        self.assertEqual([op["dst"] for op in movs], ["u_lo_ub", "u_hi_ub"])
        for op in movs:
            self.assertEqual(op["block_mode"], "m")
            self.assertEqual(op["nburst"], 1)
            self.assertEqual(op["burst_len_kb"], 64)
            self.assertEqual((op["src_stride"], op["dst_stride"]), (0, 0))

    def test_float_combine_scales_low_then_adds_into_high_and_casts_once(self):
        muls, adds = self.rec.kinds("vec_muls"), self.rec.kinds("vec_add")
        self.assertEqual(len(muls), 8)
        self.assertEqual(len(adds), 8)
        for index, op in enumerate(muls):
            self.assertEqual(op["dst"], "u_lo_ub")
            self.assertEqual(op["src"], "u_lo_ub")
            self.assertEqual((op["dst_group"], op["src_group"]), (index, index))
            self.assertEqual(op["dst_rep_stride"], 8)
            self.assertEqual(op["src_rep_stride"], 8)
            self.assertEqual((op["dst_rep_stride"], op["src_rep_stride"]),
                             (op["dst_rep_stride"], op["src_rep_stride"]))
            self.assertEqual(op["dst_rep_stride"],
                             self.plan["vector"]["rep_stride"])
            self.assertEqual(op["mask"], self.plan["vector"]["mask"])
            self.assertEqual(op["repeat_times"],
                             self.plan["vector"]["repeats_per_chunk"])
            self.assertEqual(op["scalar"], 1.0 / 1024.0)
        for index, op in enumerate(adds):
            self.assertEqual((op["dst"], op["src0"], op["src1"]),
                             ("u_hi_ub", "u_hi_ub", "u_lo_ub"))
            self.assertEqual((op["dst_group"], op["src0_group"], op["src1_group"]),
                             (index, index, index))
            self.assertEqual(op["repeat_times"],
                             self.plan["vector"]["repeats_per_chunk"])
        self.assertTrue(math.isclose(muls[0]["scalar"], 2.0 ** -10, rel_tol=0, abs_tol=0))
        casts = self.rec.kinds("h_cast")
        self.assertEqual(len(casts), 1)
        self.assertEqual(casts[0], {"op": "h_cast", "dst": "u_h_ub", "src": "u_hi_ub",
                                    "round_mode": "none"})
        order = [op["op"] for op in self.rec.ops]
        self.assertLess(order.index("tensor_mov"),
                        next(i for i, op in enumerate(self.rec.ops)
                             if op["op"] == "vec_muls"))
        last_mul = max(i for i, op in enumerate(self.rec.ops) if op["op"] == "vec_muls")
        first_add = min(i for i, op in enumerate(self.rec.ops) if op["op"] == "vec_add")
        cast_at = order.index("h_cast")
        self.assertLess(last_mul, first_add)
        self.assertEqual([op["dst_group"] for op in muls], list(range(8)))
        self.assertEqual([op["dst_group"] for op in adds], list(range(8)))
        self.assertLess(max(i for i, op in enumerate(self.rec.ops)
                            if op["op"] == "vec_add"), cast_at)
        self.assertLess(cast_at, len(order) - 1)
        self.assertEqual(self.rec.ops[-1]["op"], "data_move")

    def test_no_bias_no_float_weight_teacher_no_relu_in_the_body(self):
        self.assertFalse(hasattr(self.rec, "vec_relu"))
        names = {op.get("dst") for op in self.rec.ops}
        self.assertNotIn("bias_ub", names)
        matmuls = self.rec.kinds("matmul")
        self.assertTrue(all(op["m"] == 128 and op["k"] == 128 and op["n"] == 128
                            for op in matmuls))

    def test_emitted_offsets_evaluate_to_the_planned_addresses(self):
        """Evaluate the emitted address arithmetic with fixed core/step samples."""
        plan = self.plan
        hs, kb1, m_pad, n = (plan["head_stride_elems"], 8, 512, 128)
        task = SAMPLE["core"] + SAMPLE["step"] * plan["grid"]["cores"]
        per_head = plan["grid"]["per_head_tasks"]
        head, local = divmod(task, per_head)
        mt_idx, nt_idx = divmod(local, plan["grid"]["n_tiles"])
        base_m, base_n = mt_idx * plan["tiles"]["mb"], nt_idx * plan["tiles"]["nb"]
        self.assertTrue(0 <= task < plan["grid"]["tasks"])
        for kind, key, stride in (("a_hi_l1", "hi", hs["hi"]),
                                  ("a_lo_l1", "lo", hs["lo"])):
            offsets = [op["src_offset"] for op in self.rec.kinds("data_move")
                       if op["dst"] == kind]
            self.assertEqual(len(offsets), 32)
            for kt, offset in enumerate(offsets):
                expected = head * stride + (kt * kb1 * m_pad + base_m) * 16
                self.assertEqual(offset.value, expected, f"{kind} k-block {kt}")
            self.assertEqual(len({off.value for off in offsets}), 32)
        b_offsets = [op["src_offset"] for op in self.rec.kinds("data_move")
                     if op["dst"] == "b_l1"]
        for kt, offset in enumerate(b_offsets):
            expected = head * hs["b"] + (kt * kb1 * n + base_n) * 16
            self.assertEqual(offset.value, expected, f"b_l1 k-block {kt}")
        store = self.rec.kinds("data_move")[-1]
        self.assertEqual(store["dst_offset"].value,
                         head * hs["out"] + ((base_n // 16) * m_pad + base_m) * 16)
        self.assertEqual(hs["hi"] * 16, plan["gm_elems"]["hi"])

    def test_tail_geometry_emits_the_same_grid_with_448_rows(self):
        tail_plan, tail_rec, _gm = _emit(pv.TARGET_SHAPES["pv_tail_chunk"])
        self.assertEqual(tail_plan["padded"]["m_pad"], 512)
        self.assertEqual(tail_plan["padded"]["m_pad_rows"], 64)
        self.assertEqual(len(tail_rec.kinds("matmul")), 64)
        self.assertEqual(len(tail_rec.kinds("data_move")),
                         len(self.rec.kinds("data_move")))
        self.assertEqual(tail_rec.kinds("data_move")[-1]["dst_stride"], 384)

    def test_store_target_is_the_out_gm_and_carries_the_head_stride(self):
        store = self.rec.kinds("data_move")[-1]
        self.assertEqual(store["dst"], "out_gm")
        self.assertEqual(store["src"], "u_h_ub")
        out_stride = self.plan["head_stride_elems"]["out"]
        self.assertEqual(store["dst_offset"].value % out_stride, (0 * 512 + 128) * 16)
        self.assertGreaterEqual(store["dst_offset"].value, 0)


if __name__ == "__main__":
    unittest.main()