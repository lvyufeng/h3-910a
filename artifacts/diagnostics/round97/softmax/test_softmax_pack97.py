#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round97 softmax+probability-pack source.

No torch, no torch_npu, no tbe/CANN, no NPU, no model, no SSH: every check here is
pure python against the planning/validation layer, so this file runs anywhere.
These tests prove geometry, NZ row arithmetic, DMA descriptors, tail/pad handling,
the five-pointer wire and the capability fail-closed path. They are NOT a
numerical or speed result and they do not compile or run the kernel.
"""
import ast
import hashlib
import importlib.util
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "softmax_pack_padded_fixed97.py"
_spec = importlib.util.spec_from_file_location("softmax_pack97_under_test", SOURCE_PATH)
softmax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(softmax)
TEXT = SOURCE_PATH.read_text()
TREE = ast.parse(TEXT)


def _half(value: float) -> float:
    """Round a python float to IEEE-754 binary16 with stdlib ``struct`` only."""
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


class GeometryContracts(unittest.TestCase):
    def test_fixed_vision_geometry_and_true_key_count(self):
        plan = softmax.plan_kernel()
        lg, pd = plan["logical"], plan["padded"]
        self.assertEqual(lg["batch"], 16)
        self.assertEqual(lg["m"], 512)
        self.assertEqual(lg["n"], 4096)
        self.assertEqual(lg["valid_keys"], 4032)
        self.assertEqual(pd["n1"], 256)
        self.assertEqual(pd["block"], 16)
        self.assertEqual(pd["tail_keys"], 64)
        self.assertEqual(pd["n1"] * pd["block"], 4096)
        # Attention runs over one image's raw 48x84 patch grid, i.e. 4032 keys,
        # BEFORE the 2x2 merger. So valid_keys is 4032, not 4032/4 and not 8064.
        self.assertEqual(4032, 48 * 84)
        self.assertEqual(lg["valid_keys"], 48 * 84)
        self.assertNotEqual(lg["valid_keys"], 2 * (48 * 84) // 4)   # not post-merger

    def test_original_head_scale_not_padded_128(self):
        scale = softmax.score_scale()
        self.assertAlmostEqual(scale, 72 ** -0.5, places=15)
        self.assertAlmostEqual(scale, 0.11785113019775793, places=15)
        self.assertNotAlmostEqual(scale, 128 ** -0.5, places=6)
        plan = softmax.plan_kernel()
        self.assertEqual(plan["logical"]["head_dim"], 72)
        self.assertEqual(plan["logical"]["score_scale"], scale)

    def test_lo_rescale_is_1024_and_pv_reconstruction_is_consistent(self):
        plan = softmax.plan_kernel()
        self.assertEqual(plan["logical"]["lo_scale"], 1024.0)
        # hi + lo/1024 reconstructs p far more tightly than a single Half can.
        for p in (0.0, 1e-8, 1 / 4032, 0.25, 0.5, 1.0):
            hi = _half(p)
            lo = _half((p - hi) * 1024.0)
            self.assertLessEqual(abs((hi + lo / 1024.0) - p), 2 ** -24 + 1e-12)
            # a single Half alone would be worse than the two-leaf split
            self.assertLessEqual(abs(hi - p), 2 ** -11 * max(abs(p), 2 ** -14) + 1e-12)

    def test_m_choices_and_endpoint_pad(self):
        for m, zero_rows in ((512, 0), (448, 64)):
            plan = softmax.plan_kernel(m=m)
            self.assertEqual(plan["padded"]["m_pad"], 512)
            self.assertEqual(plan["padded"]["zero_rows"], zero_rows)
        with self.assertRaises(softmax.SoftmaxPackError):
            softmax.plan_kernel(m=384)
        with self.assertRaises(softmax.SoftmaxPackError):
            softmax.plan_kernel(n=4032)
        with self.assertRaises(softmax.SoftmaxPackError):
            softmax.plan_kernel(valid_keys=4096)


class TaskGridContracts(unittest.TestCase):
    def test_flatten_is_bijective_over_all_heads_and_rows(self):
        plan = softmax.plan_kernel()
        total = plan["grid"]["tasks"]
        self.assertEqual(total, 16 * 512)
        seen = set()
        for task in range(total):
            head, row = softmax.flatten_task(task, plan)
            self.assertTrue(0 <= head < 16)
            self.assertTrue(0 <= row < 512)
            self.assertNotIn((head, row), seen)
            seen.add((head, row))
        self.assertEqual(len(seen), total)

    def test_grid_covers_every_task_with_30_cores(self):
        plan = softmax.plan_kernel()
        gr = plan["grid"]
        self.assertEqual(gr["cores"], 30)
        self.assertEqual(gr["steps"], -(-gr["tasks"] // 30))
        self.assertEqual(gr["steps"], 274)
        # core + step*cores must reach every task exactly once and no core overruns
        assigned = [core + step * 30
                    for step in range(gr["steps"]) for core in range(30)]
        self.assertEqual(sorted(t for t in assigned if t < gr["tasks"]), list(range(gr["tasks"])))
        self.assertTrue(all(t < gr["tasks"] + 30 for t in assigned))
        live_in_last = sum(1 for t in assigned if t < gr["tasks"]) - (gr["steps"] - 1) * 30
        self.assertEqual(live_in_last, gr["last_step_live"])

    def test_flatten_rejects_out_of_range_tasks(self):
        plan = softmax.plan_kernel()
        for bad in (-1, plan["grid"]["tasks"], 999999):
            with self.assertRaises(softmax.SoftmaxPackError):
                softmax.flatten_task(bad, plan)


class OffsetContracts(unittest.TestCase):
    def test_row_offsets_are_per_head_and_16_aligned(self):
        plan = softmax.plan_kernel()
        hs = plan["head_stride_elems"]["score_f32"]
        self.assertEqual(hs, 256 * 512 * 16)
        # Layout is [head, n1=256, m_pad, block=16]: the DMA *base* of one query
        # row is head*hs + row*16, and the n1 stride is carried by src_stride.
        first = softmax.row_offsets(plan, 0, 0)
        self.assertEqual(first["score_f32"], 0)
        self.assertEqual(softmax.row_offsets(plan, 0, 1)["score_f32"], 16)
        self.assertEqual(softmax.row_offsets(plan, 1, 0)["score_f32"], hs)
        self.assertEqual(softmax.row_offsets(plan, 15, 511)["score_f32"],
                         15 * hs + 511 * 16)
        for row in (0, 1, 447, 511):
            offsets = softmax.row_offsets(plan, 3, row)
            for key in ("score_f32", "hi_half", "lo_half"):
                self.assertEqual(offsets[key] % 16, 0, key)

    def test_heads_never_overlap_and_whole_tensor_is_covered(self):
        plan = softmax.plan_kernel()
        hs = plan["head_stride_elems"]["score_f32"]
        n1, m_pad = plan["padded"]["n1"], plan["padded"]["m_pad"]
        self.assertEqual(plan["gm_bytes"]["score_f32"], 16 * hs * 4)
        self.assertEqual(plan["gm_bytes"]["hi_half"], 16 * hs * 2)
        self.assertEqual(plan["gm_bytes"]["lo_half"], 16 * hs * 2)
        # Every (head, n1, row) NZ block of 16 is visited exactly once: the block
        # index head*(n1*m_pad) + n1_idx*m_pad + row is a bijection onto the tensor.
        self.assertEqual(16 * n1 * m_pad * 16, 16 * hs)
        highest = max(softmax.row_offsets(plan, head, row)["score_f32"]
                      for head in (0, 15) for row in (0, m_pad - 1))
        highest += (n1 - 1) * (m_pad * 16) + 15
        self.assertEqual(highest, 16 * hs - 1)   # last element of the whole tensor

    def test_padded_row_predicate_matches_the_endpoint_pad(self):
        plan = softmax.plan_kernel(m=448)
        self.assertFalse(softmax.is_padded_row(plan, 0))
        self.assertFalse(softmax.is_padded_row(plan, 447))
        self.assertTrue(softmax.is_padded_row(plan, 448))
        self.assertTrue(softmax.is_padded_row(plan, 511))
        full = softmax.plan_kernel(m=512)
        self.assertFalse(any(softmax.is_padded_row(full, row) for row in range(512)))


class DmaContracts(unittest.TestCase):
    def test_score_load_is_two_fp32_blocks_with_double_stride(self):
        plan = softmax.plan_kernel()
        load = plan["dma"]["score_load"]
        self.assertEqual(load["nburst"], 256)
        self.assertEqual(load["burst"], 2)          # 16 FP32 = 64 B = 2 x 32 B
        self.assertEqual(load["src_stride"], 2 * (512 - 1))
        self.assertEqual(load["dst_stride"], 0)     # packed contiguous UB row
        self.assertEqual(load["nburst"] * load["burst"] * 8, 4096)

    def test_prob_store_is_one_half_block_and_must_not_use_the_float_factor(self):
        plan = softmax.plan_kernel()
        store = plan["dma"]["prob_store"]
        self.assertEqual(store["nburst"], 256)
        self.assertEqual(store["burst"], 1)         # 16 Half = 32 B = 1 unit
        self.assertEqual(store["dst_stride"], 512 - 1)
        self.assertNotEqual(store["dst_stride"], 2 * (512 - 1))
        self.assertEqual(store["nburst"] * store["burst"] * 16, 4096)

    def test_descriptors_tile_the_whole_row_with_no_gap_or_overlap(self):
        for m in (512, 448):
            plan = softmax.plan_kernel(m=m)
            coverage = softmax.row_element_coverage(plan)
            self.assertTrue(coverage["load_covers_row"])
            self.assertTrue(coverage["store_covers_row"])
            self.assertEqual(coverage["load_units_per_n1_step"], 1024)
            self.assertEqual(coverage["store_units_per_n1_step"], 512)
            self.assertEqual(coverage["load_elements"], 4096)
            self.assertEqual(coverage["store_elements"], 4096)
            # 256 n1 steps of one full row stride == the whole head stride
            self.assertEqual(256 * (512 * 16 * 4 // 32),
                             plan["head_stride_elems"]["score_f32"] * 4 // 32)
            self.assertEqual(256 * (512 * 16 * 2 // 32),
                             plan["head_stride_elems"]["hi_half"] * 2 // 32)


class MathOrderContracts(unittest.TestCase):
    def test_math_step_order_masks_tail_then_exp_then_rezeros_before_sum(self):
        plan = softmax.plan_kernel()
        steps = plan["math_steps"]
        self.assertEqual(steps, list(softmax.MATH_STEPS))
        i_mask = steps.index("mask_key_tail")
        i_max = steps.index("reduce_row_max")
        i_exp = steps.index("exp_row")
        i_zero = steps.index("zero_key_tail")
        i_sum = steps.index("reduce_row_sum")
        self.assertLess(i_mask, i_max)
        self.assertLess(i_max, i_exp)
        self.assertLess(i_exp, i_zero)      # explicit tail zero BEFORE the denominator
        self.assertLess(i_zero, i_sum)
        self.assertLess(i_sum, steps.index("normalise"))
        self.assertLess(steps.index("normalise"), steps.index("half_hi"))
        self.assertLess(steps.index("half_hi"), steps.index("widen_hi"))
        self.assertLess(steps.index("widen_hi"), steps.index("residual_scaled"))
        self.assertLess(steps.index("residual_scaled"), steps.index("half_lo"))

    def test_negative_max_is_finite_so_the_mask_is_not_a_nan(self):
        plan = softmax.plan_kernel()
        value = plan["vector"]["negative_max"]
        self.assertLess(value, -3.4e38)
        self.assertGreater(value, -3.4028235e38)
        self.assertFalse(value != value)     # not NaN
        self.assertNotEqual(value, float("-inf"))

    def test_source_emits_the_reviewed_order_and_no_causal_or_teacher(self):
        calls = [node.func.attr for node in ast.walk(TREE)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for required in ("vec_max", "vec_reduce_add", "vec_exp", "vec_rec",
                         "h_cast", "vector_dup", "data_move", "vec_muls", "vec_add"):
            self.assertIn(required, calls)
        lowered = TEXT.lower()
        # No causal mask, no BF16 path, no teacher/gold cast, no Torch, no collective.
        for forbidden in ("causal_mask", "bfloat16", "float32_linear", "teacher_weight",
                          "torch.", "dist.all_reduce", "recompute_mask", "recompute_"):
            self.assertNotIn(forbidden, lowered)
        # ...and the absence is documented rather than merely unimplemented.
        self.assertIn("no causal", lowered)
        self.assertIn("no bf16", lowered)
        # the tail must be masked with the finite -FLT_MAX value and re-zeroed
        self.assertIn("negative_max", TEXT)
        self.assertGreaterEqual(TEXT.count("s_f[valid_keys]"), 2)


class WireContracts(unittest.TestCase):
    def test_five_gm_pointer_wire_and_dtypes(self):
        plan = softmax.plan_kernel()
        self.assertEqual(len(plan["kernel_arg_order"]), 5)
        self.assertEqual(plan["kernel_arg_order"],
                         ["scores_gm_float32", "aux0_gm_float32", "aux1_gm_float32",
                          "hi_gm_float16", "lo_gm_float16"])
        shapes = plan["gm_shapes"]
        self.assertEqual(shapes["score_f32"], [16, 256, 512, 16])
        self.assertEqual(shapes["hi_half"], shapes["score_f32"])
        self.assertEqual(shapes["lo_half"], shapes["score_f32"])
        self.assertEqual(shapes["aux0_f32"], [16])
        self.assertIn('inputs=[gm["scores"], gm["aux0"], gm["aux1"]]', TEXT)
        self.assertIn('outputs=[gm["hi"], gm["lo"]]', TEXT)
        self.assertIn("five GM pointers", TEXT)

    def test_layout_is_the_pv_a_nz_layout(self):
        plan = softmax.plan_kernel()
        # score C_NZ and PV's A_NZ share H16, N1=256, Mpad512, block16
        self.assertEqual(plan["gm_shapes"]["score_f32"][1:], [256, 512, 16])
        self.assertTrue(plan["gm_shapes"]["hi_half"][1:] == [256, 512, 16])
        self.assertIn("(head, n1=256, m_pad, 16)", TEXT)


class BudgetContracts(unittest.TestCase):
    def test_ub_budget_is_under_the_safe_bound(self):
        for m in (512, 448):
            plan = softmax.plan_kernel(m=m)
            buf = plan["buffers"]
            self.assertEqual(buf["s_f"], 4096 * 4)
            self.assertEqual(buf["aux_f"], 4096 * 4)
            self.assertEqual(buf["zero_f"], 4096 * 4)
            self.assertEqual(buf["hi_h"], 4096 * 2)
            self.assertEqual(buf["lo_h"], 4096 * 2)
            declared = sum(value for key, value in buf.items()
                           if isinstance(value, int)
                           and key not in ("ub_total_bytes", "ub_safe_bytes"))
            self.assertEqual(buf["ub_total_bytes"], declared)
            self.assertLessEqual(buf["ub_total_bytes"], 196608)
            self.assertLess(buf["ub_total_bytes"], 128 * 1024)

    def test_repeat_and_reduce_limits(self):
        plan = softmax.plan_kernel()
        vec = plan["vector"]
        self.assertEqual(vec["mask"], 64)
        self.assertEqual(vec["repeats"], 64)
        self.assertEqual(vec["rep_stride"], 8)
        self.assertLessEqual(vec["repeats"], 255)
        self.assertLessEqual(vec["repeats"], 4095)
        self.assertEqual(vec["reduce_work"], 64)
        self.assertEqual(vec["newton_steps"], 2)
        self.assertEqual(vec["masked_tail_keys"], 64)
        self.assertEqual(softmax.SCALAR_MASK, 8)

    def test_vector_signatures_match_actual_910_API(self):
        for node in ast.walk(TREE):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute): continue
            if node.func.attr in ('vec_muls', 'vec_adds'):
                self.assertEqual(len(node.args), 7)
            if node.func.attr in ('vec_mul', 'vec_add', 'vec_max'):
                self.assertEqual(len(node.args), 8)
        self.assertIn('t.vector_dup(mask, zero_f, 0.0, repeats, 1, stride)', TEXT)

    def test_query_padding_uses_scalar_flag_not_vector_cast_branch(self):
        self.assertIn('valid_row.set_as(0.0)', TEXT)
        self.assertIn('t.vec_muls(mask, s_f, s_f, valid_row, repeats, stride, stride)', TEXT)
        self.assertNotIn('with t.else_scope():', TEXT)

    def test_eight_wide_scratch_ops_never_use_the_64_wide_row_mask(self):
        """A 64 mask on an 8-slot tensor would address past that buffer."""
        eight_wide = ("neg_f", "den_f", "inv_f", "newton_f")
        # Elementwise ops whose operands are all 8 wide must pass SCALAR_MASK (8).
        for op in ("vec_muls", "vec_adds", "vec_add", "vec_mul", "vec_rec",
                   "vector_dup"):
            for tensor in eight_wide:
                self.assertNotRegex(
                    TEXT, r"t\.%s\(mask,\s*%s\b" % (op, tensor),
                    "%s must not take the 64-wide row mask on %s" % (op, tensor))
                self.assertNotRegex(
                    TEXT, r"t\.%s\(mask,\s*\w+,\s*%s\b" % (op, tensor),
                    "%s must not take the 64-wide row mask on %s" % (op, tensor))
        # The reciprocal chain is explicitly 8 wide.
        self.assertIn("t.vec_rec(SCALAR_MASK, inv_f, den_f", TEXT)
        for tensor in eight_wide:
            self.assertTrue("SCALAR_MASK, " + tensor in TEXT,
                            "%s must be driven with the 8-wide mask" % tensor)
        # vreduce is the documented exception: its mask describes the 64-wide
        # *source* row, while the 8-slot tensor is only the aligned destination.
        self.assertIn('t.vec_max(mask, max_work, max_work, s_f[group * 64]', TEXT)
        self.assertNotIn('t.vec_reduce_max(', TEXT)
        self.assertIn("t.vec_reduce_add(mask, sum_t, s_f", TEXT)
        self.assertIn('max_scalar.set_as(max_work[0])', TEXT)


class CapabilityContracts(unittest.TestCase):
    def test_capability_report_fails_closed_when_float_reduce_is_unsupported(self):
        import sys
        fake_platform = SimpleNamespace(
            set_current_compile_soc_info=lambda soc: None,
            get_soc_spec=lambda key: {"SHORT_SOC_VERSION": "Ascend910",
                                      "CORE_NUM": 30, "UB_SIZE": 262144}[key],
            intrinsic_check_support=lambda name, dtype: False)
        fake_tbe = SimpleNamespace(common=SimpleNamespace(platform=fake_platform))
        saved = {k: sys.modules.get(k) for k in ("tbe", "tbe.common", "tbe.common.platform")}
        try:
            sys.modules["tbe"] = fake_tbe
            sys.modules["tbe.common"] = fake_tbe.common
            sys.modules["tbe.common.platform"] = fake_platform
            with self.assertRaises(softmax.SoftmaxPackError) as ctx:
                softmax.capability_report()
            self.assertIn("vector reduce is not supported", str(ctx.exception))
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_capability_report_accepts_a_supported_float_reduce(self):
        import sys
        fake_platform = SimpleNamespace(
            set_current_compile_soc_info=lambda soc: None,
            get_soc_spec=lambda key: {"SHORT_SOC_VERSION": "Ascend910",
                                      "CORE_NUM": 32, "UB_SIZE": 262144}[key],
            intrinsic_check_support=lambda name, dtype: dtype in ("float32", "f32"))
        fake_tbe = SimpleNamespace(common=SimpleNamespace(platform=fake_platform))
        saved = {k: sys.modules.get(k) for k in ("tbe", "tbe.common", "tbe.common.platform")}
        try:
            sys.modules["tbe"] = fake_tbe
            sys.modules["tbe.common"] = fake_tbe.common
            sys.modules["tbe.common.platform"] = fake_platform
            report = softmax.capability_report()
            self.assertTrue(report["float_vreduce_supported"])
            self.assertEqual(report["core_num"], 32)
            self.assertEqual(report['launched_usable_cores'], 30)
            self.assertEqual(report["compile_soc"], "Ascend910")
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_wrong_soc_or_core_count_is_refused(self):
        import sys
        for bad in ({"SHORT_SOC_VERSION": "Ascend910B", "CORE_NUM": 30, "UB_SIZE": 262144},
                    {"SHORT_SOC_VERSION": "Ascend910", "CORE_NUM": 29, "UB_SIZE": 262144}):
            fake_platform = SimpleNamespace(
                set_current_compile_soc_info=lambda soc: None,
                get_soc_spec=lambda key, table=bad: table[key],
                intrinsic_check_support=lambda name, dtype: True)
            fake_tbe = SimpleNamespace(common=SimpleNamespace(platform=fake_platform))
            saved = {k: sys.modules.get(k) for k in ("tbe", "tbe.common",
                                                     "tbe.common.platform")}
            try:
                sys.modules["tbe"] = fake_tbe
                sys.modules["tbe.common"] = fake_tbe.common
                sys.modules["tbe.common.platform"] = fake_platform
                with self.assertRaises(softmax.SoftmaxPackError):
                    softmax.capability_report()
            finally:
                for key, value in saved.items():
                    if value is None:
                        sys.modules.pop(key, None)
                    else:
                        sys.modules[key] = value


class ManifestContracts(unittest.TestCase):
    def test_manifest_hashes_match_the_delivered_files(self):
        manifest_path = HERE / "manifest97.json"
        self.assertTrue(manifest_path.is_file(), "run make_manifest97.py")
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        self.assertEqual(sorted(files), sorted(["softmax_pack97.py",
                                                "softmax_pack_fixed97.py", "softmax_pack_padded_fixed97.py",
                                                "test_softmax_pack97.py", "README.md"]))
        for name, receipt in files.items():
            payload = (HERE / name).read_bytes()
            self.assertEqual(receipt["bytes"], len(payload), name)
            self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest(), name)

    def test_manifest_stays_non_claiming_and_records_the_assumption(self):
        manifest = json.loads((HERE / "manifest97.json").read_text())
        self.assertFalse(manifest["executed"])
        self.assertFalse(manifest["compiled_now"])
        self.assertTrue(manifest["not_a_speed_claim"])
        self.assertFalse(manifest["full_encoder_precision_accepted"])
        self.assertFalse(manifest["whole_goal_complete"])
        self.assertIsNone(manifest["numerical_tolerance"])
        assumed = [item for item in manifest["api_evidence"]
                   if item["need"] == "vec_reduce_max call shape"]
        self.assertEqual(len(assumed), 1)
        self.assertIn("ASSUMPTION", assumed[0]["evidence"])
        self.assertEqual(manifest["masking"]["valid_keys"], 4032)
        self.assertEqual(manifest["masking"]["tail_keys"], 64)
        self.assertFalse(manifest["masking"]["causal"])


class SourceAndNonClaimContracts(unittest.TestCase):
    def test_module_scope_is_stdlib_only_and_cann_is_lazy(self):
        top = set()
        for node in TREE.body:
            if isinstance(node, ast.Import):
                top.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        self.assertLessEqual(top, {"__future__", "hashlib", "json", "os", "tempfile",
                                   "pathlib", "argparse"})
        self.assertNotIn("import tbe", "\n".join(TEXT.splitlines()[:60]))
        self.assertIn("from tbe import tik", TEXT)

    def test_docstring_and_receipts_claim_no_precision_or_speed_result(self):
        self.assertIn("SOURCE ONLY", TEXT)
        self.assertIn('"executed": False', TEXT)
        self.assertIn('"not_a_speed_claim": True', TEXT)
        self.assertIn('"full_encoder_precision_accepted": False', TEXT)
        self.assertIn('"whole_goal_complete": False', TEXT)
        self.assertIn('"numerical_tolerance": None', TEXT)
        self.assertIn("bit_identity_with_torch_softmax_claimed", TEXT)

    def test_aux_operands_documented_as_unused_placeholders(self):
        self.assertIn("aux_gm_operands_are_unused_placeholders", TEXT)
        self.assertIn("keep the proven five-pointer ABI", TEXT)
        self.assertIn("collection would prune them", TEXT)


if __name__ == "__main__":
    unittest.main()