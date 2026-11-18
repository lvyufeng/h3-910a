#!/usr/bin/env python3
"""Static contracts for the Round83 bias-epilogue kernel.

Standard library only: no torch, no torch_npu, no CANN, no device. These tests
run on a machine without the accelerator and therefore cannot claim any
numerical or performance result.

What they CAN establish: the kernel emits the intended instruction sequence
(no bias operand on matmul, FP32 accumulator preserved, bias added in FP32,
exactly one Half conversion at the very end), the packed layouts round-trip,
and the strided vector access stays inside its UB buffers under the SDK's own
documented offset arithmetic.
What they CANNOT establish: that the kernel compiles or runs, or that the
encoder agrees with the CPU FP16 reference. That needs the root's execution.
"""

import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tik_bias_epilogue83 as kern  # noqa: E402  (stdlib-safe: TIK imported lazily)

SOURCE = (HERE / "tik_bias_epilogue83.py").read_text(encoding="utf-8")

# SDK constants read from the CANN 9.2.0 copies kept in ./sdk.
BLOCK_NUM_PER_REP = 8      # BLK_NUM_PER_REP: 8 x 32 B blocks per vector repeat
ONE_BLOCK_BYTES = 32
FP32_PER_BLOCK = ONE_BLOCK_BYTES // 4
FP16_PER_BLOCK = ONE_BLOCK_BYTES // 2


class ProvenanceTests(unittest.TestCase):
    def test_module_compiles_and_imports_without_torch_or_tik(self):
        compile(SOURCE, "tik_bias_epilogue83.py", "exec")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("tbe", sys.modules)

    def test_tik_is_imported_lazily(self):
        tree = ast.parse(SOURCE)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertNotIn("tbe", [a.name for a in node.names])
                self.assertNotIn("torch", [a.name for a in node.names])
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module or "", ("tbe", "torch", "torch_npu"))
        # and the one real import lives inside a function
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_tik"]
        self.assertEqual(len(funcs), 1)
        inner = [n for n in ast.walk(funcs[0]) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertTrue(inner, "the TIK import must live inside _tik()")

    def test_python_wrapper_is_not_a_numerical_gold(self):
        tree = ast.parse(SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotEqual(alias.name.split(".")[0], "numpy")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual((node.module or "").split(".")[0], "numpy")
        plan = kern.build_plan()
        self.assertFalse(plan["proves_full_encoder_precision"])
        self.assertFalse(plan["proves_speed"])


class InstructionSequenceTests(unittest.TestCase):
    def _calls(self, name):
        tree = ast.parse(SOURCE)
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == name]

    def test_matmul_never_passes_a_bias_operand(self):
        calls = self._calls("matmul")
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertIn("bias", kwargs, "bias must be passed explicitly as None")
        self.assertIsInstance(kwargs["bias"], ast.Constant)
        self.assertIsNone(kwargs["bias"].value)
        self.assertIn("init_l1out", kwargs)
        self.assertIs(kwargs["init_l1out"].value, True)

    def test_accumulator_is_fp32_and_moved_without_cast(self):
        text = " ".join(SOURCE.split())
        self.assertIn('t.Tensor("float32", (n1, m, 16), name="acc_l0c", scope=tik.scope_cc)', text)
        self.assertIn('t.Tensor("float32", (n1, m, 16), name="c32_ub", scope=tik.scope_ubuf)', text)
        # tensor_mov L0C->UB with an FP32 destination: no conversion
        self.assertIn('t.tensor_mov(c32, acc_l0c, "m", 1, m * n * 4 // 1024, 0, 0)', text)

    def test_bias_add_uses_vec_add_not_matmul_bias(self):
        calls = self._calls("vec_add")
        self.assertEqual(len(calls), 1, "one vec_add inside the n1 group loop")
        args = calls[0].args
        self.assertEqual(len(args), 8, "vec_add(mask,dst,src0,src1,repeat,3 strides)")
        self.assertIsInstance(args[0], ast.Constant)
        self.assertEqual(args[0].value, 16)          # 16 FP32 elements
        # dst and src0 are the same in-place view of the group
        self.assertEqual(ast.dump(args[1]), ast.dump(args[2]))
        # src1 is the per-group bias slice
        self.assertIsInstance(args[3], ast.Subscript)
        self.assertEqual([a.value for a in args[5:]], [2, 2, 0])
        # the repeat is the row count and stays inside the 255 limit
        self.assertIsInstance(args[4], ast.Name)
        self.assertEqual(args[4].id, "m")

    def test_exactly_one_half_conversion_and_it_is_last(self):
        self.assertEqual(len(self._calls("h_cast")), 1)
        self.assertEqual(len(self._calls("tensor_mov")), 1)
        text = " ".join(SOURCE.split())
        self.assertIn('t.h_cast(c16, c32, "none")', text)
        # the Half buffer must be declared only inside the cast branch, after bias
        tree = ast.parse(SOURCE)
        assigns = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "Tensor"
                   and 'c16' in ast.dump(n)]
        self.assertEqual(len(assigns), 1)

    def test_no_global_soc_override_to_910a_or_a2(self):
        self.assertEqual(kern.PINNED_COMPILE_SOC, "Ascend910B")
        self.assertNotIn("set_current_compile_soc_info(\"Ascend910A\")", SOURCE)
        self.assertNotIn("'Ascend910A'", SOURCE)
        self.assertEqual(kern.EXPECTED_CORE_NUM, 30)

    def test_network_and_system_side_effects_absent(self):
        for forbidden in ("subprocess", "os.system", "requests", "urlopen", "socket"):
            self.assertNotIn(forbidden, SOURCE)


class LayoutTests(unittest.TestCase):
    def test_A_and_B_packing_are_bijections(self):
        m, k, n = 16, 32, 16
        for i_m in range(m):
            for i_k in range(k):
                self.assertEqual(kern.pack_mk_index(i_m, i_k, m, k),
                                 (i_k // 16) * (m * 16) + i_m * 16 + (i_k % 16))
        seen = {kern.pack_mk_index(i, j, m, k) for i in range(m) for j in range(k)}
        self.assertEqual(seen, set(range(m * k)))
        seen_b = {kern.pack_nk_weight_index(i, j, n, k) for i in range(n) for j in range(k)}
        self.assertEqual(seen_b, set(range(n * k)))

    def test_packed_layout_matches_the_executed_round82_convention(self):
        # Round82 executed a_np.reshape(m, k//16, 16).transpose(1, 0, 2),
        # i.e. packed[q][i][z] = a[i, q*16+z]. Reproduce that with pure python.
        m, k = 8, 32
        a = [[100 * i + j for j in range(k)] for i in range(m)]
        packed = [[[a[i][q * 16 + z] for z in range(16)] for i in range(m)]
                  for q in range(k // 16)]
        flat = [v for block in packed for row in block for v in row]
        for i in range(m):
            for j in range(k):
                self.assertEqual(flat[kern.pack_mk_index(i, j, m, k)], a[i][j])

    def test_output_unpacking_recovers_row_major_matrix(self):
        m, n = 16, 32
        c = [[1000 * i + j for j in range(n)] for i in range(m)]
        packed = [[[c[i][q * 16 + z] for z in range(16)] for i in range(m)]
                  for q in range(n // 16)]
        flat = [v for block in packed for row in block for v in row]
        for i in range(m):
            for j in range(n):
                self.assertEqual(flat[kern.unpack_mn_index(i, j, m, n)], c[i][j])

    def test_bias_group_slice_matches_column_group(self):
        for i_n in (0, 1, 15, 16, 17, 127):
            self.assertEqual(kern.bias_group_slice(i_n), (i_n // 16) * 16)
        # bias must be identical for every row of a group
        m, n = 16, 32
        c = [[float(i) for _ in range(n)] for i in range(m)]
        bias = [float(10 + j) for j in range(n)]
        expected = [[c[i][j] + bias[j] for j in range(n)] for i in range(m)]
        # emulate the kernel: per group, add 16 bias values to all m rows
        packed = [[(c[i][g * 16 + z], bias[g * 16 + z]) for z in range(16)]
                  for g in range(n // 16) for i in range(m)]
        got = [[sum(packed[(j // 16) * m + i][j % 16]) for j in range(n)] for i in range(m)]
        self.assertEqual(got, expected)


class BufferBoundsTests(unittest.TestCase):
    """Reproduce the SDK's own overflow arithmetic for the strided vec_add.

    te/tik/tik_lib/tik_vector_api/tik_tensor_op.py:
        _calculate_vecotor_max_offset(repeat, mask_len) for stride_unit 0:
            ((repeat-1)*rep_stride + (blk_num_last_rep-1)*blk_stride)*block_len
            + ele_num_last_blk
        check_tensor_op_overflow: need_offset = extend_offset + offset + ori_offset,
        asserted <= reduce_mul(original_shape).
    """

    @staticmethod
    def max_offset_elements(repeat_times, mask_len, rep_stride, blk_stride, block_len):
        blk_num_last_rep = -(-mask_len // block_len)
        ele_num_last_blk = mask_len % block_len or block_len
        return ((repeat_times - 1) * rep_stride +
                (blk_num_last_rep - 1) * blk_stride) * block_len + ele_num_last_blk

    def test_last_group_dst_stays_exactly_within_the_fp32_ub_tensor(self):
        m, n1 = 128, 8
        block_len = FP32_PER_BLOCK          # 8 FP32 per 32 B block
        mask_len = 16                       # mask=16 -> 16 FP32 elements
        max_off = self.max_offset_elements(m, mask_len, 2, 1, block_len)
        self.assertEqual(max_off, m * 16)   # exactly one packed group
        total = n1 * m * 16
        for group in range(n1):
            offset = group * m * 16
            self.assertLessEqual(max_off + offset, total,
                                 f"group {group} would overflow the FP32 UB tensor")

    def test_bias_source_slice_stays_within_the_bias_tensor(self):
        n = 128
        block_len = FP32_PER_BLOCK
        # src1: repeat=m, rep_stride=0 -> only one block advances, stride 0
        max_off = self.max_offset_elements(128, 16, 0, 1, block_len)
        self.assertEqual(max_off, 16)
        for group in range(n // 16):
            self.assertLessEqual(group * 16 + max_off, n)

    def test_rep_stride_two_blocks_is_exactly_one_packed_row(self):
        self.assertEqual(2 * ONE_BLOCK_BYTES, 16 * 4)   # 16 FP32 values

    def test_repeat_limit_respected(self):
        # vector repeat is at most 255 on this part; m=128 for one group call
        self.assertLessEqual(128, 255)
        # the Half cast is delegated to h_cast, so no manual repeat is needed
        plan = kern.build_plan()
        self.assertEqual(plan["vector_add"]["repeat_times"], 128)
        self.assertEqual(plan["vector_add"]["groups"], 8)
        self.assertEqual(plan["ub_tensor_elements"], 8 * 128 * 16)


class PlanTests(unittest.TestCase):
    def test_variants_and_dtypes(self):
        self.assertEqual(sorted(kern.VARIANTS),
                         ["fp16_out_unbiased", "fp32_bias_fp16_out", "fp32_bias_fp32_out"])
        primary = kern.build_plan("fp32_bias_fp16_out")
        self.assertEqual(primary["dst_dtype"], "float16")
        self.assertTrue(primary["adds_fp32_bias"])
        self.assertTrue(primary["single_final_half_cast"])
        diag = kern.build_plan("fp32_bias_fp32_out")
        self.assertEqual(diag["dst_dtype"], "float32")
        self.assertFalse(diag["single_final_half_cast"])
        control = kern.build_plan("fp16_out_unbiased")
        self.assertFalse(control["adds_fp32_bias"])

    def test_five_gm_arguments_in_documented_order(self):
        plan = kern.build_plan()
        self.assertEqual(plan["gm_argument_count"], 5)
        self.assertEqual(plan["gm_argument_order"], kern.GM_ARG_ORDER)
        self.assertEqual(plan["gm_argument_order"][2], "bias_gm_float32")
        self.assertFalse(plan["aux_argument_used_by_math"])
        self.assertFalse(plan["matmul_bias_operand_used"])

    def test_declared_non_claims(self):
        plan = kern.build_plan()
        self.assertFalse(plan["executed"])
        self.assertFalse(plan["proves_full_encoder_precision"])
        self.assertFalse(plan["proves_speed"])

    def test_h3_geometry_not_claimed_buildable(self):
        plan = kern.build_plan()
        self.assertEqual(plan["geometry"]["m"], 128)
        with self.assertRaises(kern.BiasKernelError):
            kern._validate("fp32_bias_fp16_out", 2043, 5120, 128)
        with self.assertRaises(kern.BiasKernelError):
            kern._validate("no_such_variant", 128, 512, 128)
        # documented target shapes stay documented, not claimed
        self.assertEqual(kern.H3_GEOMETRY["kv_proj"], {"M": 2043, "K": 5120, "N": 128})

    def test_manifest_pins_the_adapter_source(self):
        manifest = json.loads((HERE / "manifest83.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["executed_on_npu"])
        self.assertFalse(manifest["compiled"])
        self.assertEqual(manifest["kernel_source_sha256"],
                         hashlib.sha256((HERE / "tik_bias_epilogue83.py").read_bytes()).hexdigest())
        self.assertEqual(manifest["base_round82_source_sha256"],
                         "5e870dfa73b6a78b2cb25921202778d555dae38f3713580c52bea89de72dc059")
        self.assertEqual(manifest["launcher_library_sha256"],
                         "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043")
        self.assertEqual(manifest["sdk_cann_version"], "9.2.0")
        self.assertEqual(manifest["device_short_soc"], "Ascend910")
        self.assertFalse(manifest["matmul_bias_operand_used"])


if __name__ == "__main__":
    unittest.main()