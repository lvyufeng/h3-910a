#!/usr/bin/env python3
"""Stdlib-only contract and numerical-stability tests for erf_gelu90.

No torch, no TIK, no CANN, no NPU: the kernel is source only. These tests pin
the plan/emission contract, the tensor-naming hazard that broke Round58's build,
the Adler/AS polynomial stability, and the adapter's per-instance restore. They
contain no accuracy acceptance: the hardware comparison belongs to the parent
driver, which runs the unchanged CPU FP16 reference.
"""

import ast
import importlib.util
import types
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("erf_gelu90_under_test", HERE / "erf_gelu90.py")
erf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(erf)


class GeometryTests(unittest.TestCase):
    def test_real_fixture_is_the_frozen_merger_activation(self):
        self.assertEqual(erf.REAL_NUMEL, 2016 * 4608)
        self.assertEqual(erf.REAL_NUMEL, 9289728)
        self.assertEqual(erf.REAL_NUMEL % erf.TILE_ELEMS, 0)

    def test_smoke_geometry_uses_two_cores_and_the_real_one_all_thirty(self):
        smoke = erf.plan_tiling(erf.SMOKE_NUMEL)
        self.assertEqual(smoke["tasks"], 2)
        self.assertEqual(smoke["cores"], 2)
        self.assertEqual(smoke["steps"], 1)
        real = erf.plan_tiling(erf.REAL_NUMEL)
        self.assertEqual(real["tasks"], 2268)
        self.assertEqual(real["cores"], 30)
        self.assertEqual(real["steps"], 76)
        self.assertEqual(real["physical_cores"], 30)
        self.assertEqual(real["live_tiles_in_last_step"], 2268 - 75 * 30)

    def test_tile_is_not_a_multiple_is_refused(self):
        for bad in (erf.TILE_ELEMS + 1, 0, -4096, 8192 + 64, True, 1.5):
            with self.assertRaises(erf.ErfGeluError):
                erf.plan_tiling(bad)

    def test_ub_budget_and_buffer_sizes(self):
        plan = erf.plan_tiling(erf.REAL_NUMEL)
        buffers = plan["buffers"]
        self.assertLessEqual(buffers["ub_bytes"], erf.PINNED_UB_SAFE_BYTES)
        self.assertEqual(buffers["half_tile_bytes"], erf.TILE_ELEMS * 2)
        self.assertEqual(buffers["float_tile_bytes"], erf.TILE_ELEMS * 4)
        self.assertEqual(buffers["ub_bytes"],
                         len(erf.HALF_BUFFERS) * erf.TILE_ELEMS * 2
                         + len(erf.FLOAT_BUFFERS) * erf.TILE_ELEMS * 4)
        for name in erf.HALF_BUFFERS + erf.FLOAT_BUFFERS:
            self.assertNotIn(name, erf.RESERVED_TENSOR_NAMES)

    def test_dma_splits_into_the_proven_burst_ceiling(self):
        plan = erf.plan_tiling(erf.REAL_NUMEL)
        dma = plan["dma"]
        self.assertEqual(dma["burst"], erf.MAX_DMA_BURST_BLOCKS)
        self.assertEqual(dma["burst"] * 32, 4096)
        self.assertEqual(dma["nburst"] * dma["burst"] * 32, dma["gm_bytes_per_tile"])
        self.assertEqual(dma["src_stride"], 0)
        self.assertEqual(dma["dst_stride"], 0)

    def test_vector_masks_match_the_32_byte_block_rule(self):
        plan = erf.plan_tiling(erf.REAL_NUMEL)
        vector = plan["vector"]
        self.assertEqual(vector["mask"], 64)
        self.assertEqual(vector["repeat_times"], erf.TILE_ELEMS // 64)
        self.assertEqual(vector["mask"] * 4 // 32, vector["rep_stride"])
        self.assertEqual(vector["rep_stride"], 8)

    def test_no_a2_only_feature_or_bf16_is_planned(self):
        plan = erf.plan_tiling(erf.REAL_NUMEL)
        self.assertFalse(plan["bf16_used"])
        self.assertFalse(plan["a2_only_features_used"])
        self.assertFalse(plan["core_dim_32_assumed"])
        self.assertEqual(erf.PINNED_CORE_NUM, 30)


class EmissionTests(unittest.TestCase):
    def setUp(self):
        self.plan = erf.plan_tiling(erf.REAL_NUMEL)
        self.steps = erf.emission_steps(self.plan)

    def test_exactly_one_final_narrowing_and_one_widening(self):
        casts = [s for s in self.steps if s["op"] == "h_cast"]
        self.assertEqual(len(casts), 2)
        widening, narrowing = casts
        self.assertEqual((widening["dst"], widening["src"]), ("x_f", "x_h"))
        self.assertEqual((narrowing["dst"], narrowing["src"]), ("c_h", "x_f"))
        self.assertEqual(narrowing["round_mode"], "none")
        self.assertEqual(len(erf.narrowing_steps(self.plan)), 1)

    def test_the_narrowing_is_the_last_arithmetic_step(self):
        kinds = [s["op"] for s in self.steps]
        self.assertEqual(kinds[-2:], ["h_cast", "data_move"])
        self.assertEqual(kinds[:2], ["data_move", "h_cast"])
        for step in self.steps[2:-2]:
            self.assertTrue(step["op"].startswith("vec_"), step)

    def test_no_half_math_after_widening(self):
        casts = [s for s in self.steps if s["op"] == "h_cast"]
        widening_index = self.steps.index(casts[0])
        narrowing_index = self.steps.index(casts[1])
        for step in self.steps[widening_index:narrowing_index]:
            if not step["op"].startswith("vec_"):
                continue
            self.assertNotIn(step["dst"], erf.HALF_BUFFERS, step)
            self.assertNotIn(step.get("src"), erf.HALF_BUFFERS, step)

    def test_every_vector_step_uses_the_float_repeat_geometry(self):
        for step in self.steps:
            if not step["op"].startswith("vec_"):
                continue
            self.assertEqual(step["mask"], 64)
            self.assertEqual(step["repeat_times"], erf.TILE_ELEMS // 64)
            self.assertEqual(step["dst_rep_stride"], 8)
            self.assertEqual(step["dst"], step["dst"])
            self.assertIn(step["dst"], erf.FLOAT_BUFFERS)
            for role in ("src", "src0", "src1"):
                value = step.get(role)
                if value is not None:
                    self.assertIn(value, erf.FLOAT_BUFFERS)

    def test_only_reviewed_ops_and_no_tanh_or_sigmoid(self):
        allowed = {"data_move", "h_cast", "vec_abs", "vec_muls", "vec_adds", "vec_mul",
                   "vec_add", "vec_rec", "vec_exp", "vec_relu"}
        for step in self.steps:
            self.assertIn(step["op"], allowed)
            for role in ("dst", "src", "src0", "src1"):
                name = step.get(role)
                if name is not None:
                    self.assertNotIn(name, erf.RESERVED_TENSOR_NAMES)
        text = (HERE / "erf_gelu90.py").read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("tanh", "sigmoid"):
                self.fail("kernel source must keep the original erf semantics")

    def test_polynomial_follows_the_pinned_abramowitz_stegun_order(self):
        scalars = [s.get("scalar") for s in self.steps if "scalar" in s]
        for coefficient in (erf.ERF_A1, erf.ERF_A2, erf.ERF_A3, erf.ERF_A4, erf.ERF_A5,
                            erf.ERF_P, 1.0, -1.0, 2.0, erf.HALF_SQRT2, 0.5):
            self.assertIn(coefficient, scalars)
        # Horner starts from a5 and ends by multiplying the inner sum by t once.
        first_horner = next(i for i, s in enumerate(self.steps)
                            if s.get("scalar") == erf.ERF_A5)
        self.assertEqual(self.steps[first_horner]["src"], "t_f")
        self.assertEqual(self.steps[first_horner]["dst"], "p_f")
        tail = self.steps[first_horner:]
        self.assertEqual([s.get("scalar") for s in tail if s["op"].startswith("vec_")
                          and "scalar" in s][:5],
                         [erf.ERF_A5, erf.ERF_A4, erf.ERF_A3, erf.ERF_A2, erf.ERF_A1])

    def test_two_newton_steps_and_a_relu_guard_are_emitted(self):
        self.assertEqual(erf.NEWTON_STEPS, 2)
        reciprocal_index = next(i for i, s in enumerate(self.steps) if s["op"] == "vec_rec")
        newton = [s for s in self.steps[reciprocal_index:]
                  if s["op"] in ("vec_mul",)]
        self.assertGreaterEqual(len(newton), 2 * erf.NEWTON_STEPS)
        self.assertIn("vec_exp", [s["op"] for s in self.steps])
        relu = [i for i, s in enumerate(self.steps) if s["op"] == "vec_relu"]
        self.assertEqual(len(relu), 1)
        exp_index = next(i for i, s in enumerate(self.steps) if s["op"] == "vec_exp")
        self.assertGreater(relu[0], exp_index)

    def test_every_instruction_carries_complete_operands(self):
        # Regression: an earlier draft passed scalar-op sources through ``src0``
        # while _emit_kernel reads ``src``, which would have failed at build time
        # on ``ub[None]``. Every emission step must now name real buffers.
        buffers = set(erf.HALF_BUFFERS) | set(erf.FLOAT_BUFFERS)
        for step in self.steps:
            op = step["op"]
            if op == "data_move":
                # GM->UB names a UB destination, UB->GM names a UB source.
                ub_role = "dst" if step["direction"] == "gm_to_ub" else "src"
                self.assertIn(step[ub_role], buffers, step)
                continue
            self.assertIn(step["dst"], buffers, step)
            if op in ("vec_muls", "vec_adds"):
                self.assertIsNotNone(step.get("scalar"), step)
                self.assertIn(step.get("src"), buffers, step)
            elif op in ("vec_mul", "vec_add"):
                self.assertIn(step.get("src0"), buffers, step)
                self.assertIn(step.get("src1"), buffers, step)
            else:
                self.assertIn(step.get("src"), buffers, step)

    def test_denominator_cannot_vanish_because_it_starts_at_one(self):
        plan = self.plan
        self.assertEqual(plan["vector"]["mask"] % 64, 0)
        # d = 1 + p*z with z >= 0 and p > 0, so d >= 1 in exact arithmetic.
        self.assertGreater(erf.ERF_P, 0.0)
        d = [s for s in self.steps if s.get("scalar") == 1.0]
        self.assertTrue(d)


class EmissionDryRunTests(unittest.TestCase):
    """Walk _emit_kernel against a recording stub: no TIK, no CANN, no NPU.

    This is the guard for the failure mode an earlier draft actually had: a
    scalar-op source recorded under the wrong role reached ``_emit_kernel`` as
    ``None`` and would have failed only at build time on ``ub[None]``.
    """

    class FakeTensor:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, item):
            return EmissionDryRunTests.FakeTensor("%s[%s]" % (self.name, item))

    class FakeTik:
        def __init__(self):
            self.calls = []

        def Tensor(self, dtype, shape, name=None, scope=None):
            return EmissionDryRunTests.FakeTensor(name)

        def for_range(self, begin, end, block_num=None):
            class Ctx:
                def __enter__(self):
                    return 0          # a real TIK loop variable is an int

                def __exit__(self, *exc):
                    return False
            return Ctx()

        def if_scope(self, condition):
            class Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, *exc):
                    return False
            return Ctx()

        def __getattr__(self, item):
            def record(*args, **kwargs):
                normalized = []
                for value in args:
                    normalized.append(value.name
                                      if isinstance(value, EmissionDryRunTests.FakeTensor)
                                      else value)
                self.calls.append((item,) + tuple(normalized))
            return record

    def test_dry_run_emits_a_complete_well_formed_instruction_stream(self):
        tik_stub = types.SimpleNamespace(scope_gm="gm", scope_ubuf="ub")
        fake = self.FakeTik()
        plan = erf.plan_tiling(erf.REAL_NUMEL)
        gm = erf._emit_kernel(fake, plan, tik_stub)
        self.assertEqual([gm[k].name for k in ("x", "aux1", "aux2", "aux3", "c")],
                         ["x_gm", "aux1_gm", "aux2_gm", "aux3_gm", "c_gm"])
        ops = [call[0] for call in fake.calls
               if call[0].startswith(("vec_", "h_cast", "data_move"))]
        self.assertEqual(len(ops), 37)
        self.assertEqual(ops.count("h_cast"), 2)
        self.assertEqual(ops.count("data_move"), 2)
        self.assertEqual(ops[:2], ["data_move", "h_cast"])
        self.assertEqual(ops[-2:], ["h_cast", "data_move"])
        for call in fake.calls:
            self.assertNotIn("NoneType", call, call)
        for call in fake.calls:
            if call[0] in ("vec_muls", "vec_adds"):
                self.assertIsInstance(call[4], float, call)
                self.assertIsInstance(call[2], str, call)
                self.assertIsInstance(call[3], str, call)
            elif call[0] in ("vec_mul", "vec_add"):
                self.assertTrue(all(isinstance(call[i], str) for i in (2, 3, 4)), call)
            elif call[0] == "h_cast":
                self.assertEqual(call[3], "none", call)
            elif call[0] == "data_move":
                # data_move(dst, src, sid, nburst, burst, src_stride, dst_stride)
                self.assertEqual((call[4], call[5]), (plan["dma"]["nburst"],
                                                     plan["dma"]["burst"]), call)
        # The single narrowing cast is the Float->Half one, immediately before GM.
        casts = [call for call in fake.calls if call[0] == "h_cast"]
        self.assertEqual(casts[-1][1:3], ("c_h", "x_f"))


class MathStabilityTests(unittest.TestCase):
    def test_emulation_tracks_the_double_precision_erf_gelu(self):
        grid = [-65504.0, -1000.0, -100.0, -30.0, -10.0, -6.0, -3.0, -1.0, -0.5,
                -0.1, -1e-3, -1e-6, 0.0, 1e-6, 0.3, 1.0, 3.0, 6.0, 10.0, 30.0,
                100.0, 1000.0, 65504.0]
        worst = 0.0
        for value in grid:
            worst = max(worst, abs(erf.scalar_emulation(value) - erf.scalar_reference(value)))
        # Bounded by Float32 rounding of the pipeline, not by the 1.5e-7 polynomial.
        self.assertLess(worst, 5e-6, worst)

    def test_emulation_is_finite_bounded_and_signed_over_a_dense_sweep(self):
        # The exact erf-GELU is NOT monotone: it dips to a shallow minimum near
        # x ~= -0.75 before rising, so a monotonicity assertion would be wrong.
        # What must hold is finiteness, sign correctness and agreement with the
        # reference across the whole sweep.
        values = [(i - 3000) * 0.01 for i in range(6001)]
        worst = 0.0
        reference_minimum = min(erf.scalar_reference(v) for v in values)
        for value in values:
            current = erf.scalar_emulation(value)
            self.assertTrue(math.isfinite(current), value)
            self.assertLessEqual(current, max(value, 0.0) + 1e-5, value)
            self.assertGreaterEqual(current, reference_minimum - 1e-5, value)
            if value > 1e-3:
                self.assertGreater(current, 0.0, value)
            elif value < -1e-3:
                self.assertLessEqual(current, 0.0, value)
            worst = max(worst, abs(current - erf.scalar_reference(value)))
        self.assertLess(worst, 5e-6, worst)
        self.assertLess(reference_minimum, -0.16)   # the real shallow dip

    def test_zero_and_small_arguments_are_exact(self):
        self.assertEqual(erf.scalar_emulation(0.0), 0.0)
        self.assertAlmostEqual(erf.scalar_emulation(1e-6), 5.0000039e-7, delta=1e-12)
        self.assertEqual(erf.scalar_erf_abs(0.0), 0.0)

    def test_far_negative_tail_returns_exactly_zero(self):
        for value in (-65504.0, -1000.0, -100.0, -30.0):
            self.assertEqual(erf.scalar_emulation(value), 0.0, value)
        # And the positive tail is the identity, as the true erf-GELU is.
        self.assertEqual(erf.scalar_emulation(65504.0), 65504.0)
        self.assertEqual(erf.scalar_emulation(100.0), 100.0)

    def test_negative_zero_is_numerically_zero_but_the_bit_differs(self):
        emulated = erf.scalar_emulation(-0.0)
        self.assertEqual(emulated, 0.0)
        reference = erf.scalar_reference(-0.0)
        self.assertEqual(reference, 0.0)
        # Recorded honestly: this kernel yields +0.0 where the CPU yields -0.0.
        self.assertEqual(math.copysign(1.0, emulated), 1.0)
        self.assertEqual(math.copysign(1.0, reference), -1.0)

    def test_erf_intermediate_stays_within_zero_and_one(self):
        values = [(i - 2000) * 0.02 for i in range(4001)] + [65504.0, -65504.0]
        for value in values:
            erf_abs = erf.scalar_erf_abs(value)
            self.assertGreaterEqual(erf_abs, 0.0, value)
            self.assertLessEqual(erf_abs, 1.0000001, value)

    def test_newton_steps_make_the_result_insensitive_to_the_reciprocal_seed(self):
        values = [(i - 2000) * 0.02 for i in range(4001)]
        baseline = max(abs(erf.scalar_emulation(v) - erf.scalar_reference(v))
                       for v in values)
        for seed_error in (2 ** -12, -(2 ** -12), 2 ** -8, -(2 ** -8)):
            degraded = max(abs(erf.scalar_emulation(v, rec_initial_relative_error=seed_error)
                               - erf.scalar_reference(v)) for v in values)
            self.assertLessEqual(degraded, baseline + 1e-9, (seed_error, degraded))

    def test_half_output_agrees_with_a_float32_exact_erf_gelu(self):
        import random

        random.seed(90)
        sample = [erf._f32(random.gauss(0.0, 1.0)) for _ in range(20000)]
        mismatch = 0
        worst = 0.0
        for value in sample:
            emitted = erf._f16(erf.scalar_emulation(value))
            reference = erf.scalar_half_reference(value)
            worst = max(worst, abs(emitted - reference))
            mismatch += emitted != reference
        # Differences are at most one Half ULP at a rounding boundary; this is a
        # prediction from the emulation, never an acceptance result.
        self.assertLess(worst, 0.002, worst)
        self.assertLess(mismatch / len(sample), 0.01, mismatch)


class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.source = (HERE / "erf_gelu90.py").read_text()
        self.tree = ast.parse(self.source)

    def test_module_imports_are_stdlib_only(self):
        stdlib = {"hashlib", "json", "math", "os", "tempfile", "pathlib", "__future__",
                  "struct"}
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], stdlib)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or "").split(".")[0], stdlib)

    def test_no_torch_tik_or_npu_import_at_module_scope(self):
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.dump(node)
                for forbidden in ("torch", "tik", "tbe", "npu", "acl"):
                    self.assertNotIn("'" + forbidden, text.lower(), ast.unparse(node))

    def test_no_remote_shell_or_build_side_effects_in_source(self):
        # The module documents "no SSH" in prose; what must be absent is any
        # executable path that could actually shell out or spawn a build.
        imported = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        for forbidden in ("subprocess", "shutil", "socket", "paramiko"):
            self.assertNotIn(forbidden, imported)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute):
                if node.attr in ("system", "popen", "spawn", "spawnl", "execv", "call"):
                    owner = ast.unparse(node.value)
                    if owner in ("os", "subprocess", "shutil"):
                        self.fail("source must not invoke %s.%s" % (owner, node.attr))

    def test_pins_are_the_reviewed_collaborators(self):
        self.assertEqual(erf.REFERENCE_RUNTIME_MODULE_SHA256,
                         "c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6")
        self.assertEqual(erf.REFERENCE_RUNTIME_LIBRARY_SHA256,
                         "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043")
        self.assertEqual(erf.REFERENCE_CANDIDATE_SHA256,
                         "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78")
        self.assertEqual(erf.PINNED_COMPILE_SOC, "Ascend910B")
        self.assertEqual(erf.PINNED_CORE_NUM, 30)

    def test_build_is_locked_and_refuses_to_overwrite(self):
        for needle in ("fcntl.flock", "tempfile.mkdtemp(prefix=\"erf_gelu90_build_\"",
                       "refusing to overwrite existing kernel directory",
                       "is retained for diagnosis"):
            self.assertIn(needle, self.source)

    def test_spec_repair_is_guarded_by_resolved_absolute_paths(self):
        body = self.source[self.source.index("def _write_spec_once"):]
        self.assertIn("resolve(strict=True).parent", body)
        self.assertIn("if resolved != intended", body)
        self.assertIn("tmp.resolve(strict=False).parent != intended.parent", body)
        self.assertIn("return False", body)
        self.assertIn("os.replace(tmp, resolved)", body)

    def test_build_never_claims_execution_or_accuracy(self):
        self.assertIn('"executed": False', self.source)
        self.assertIn('"numerical_accuracy_claim": None', self.source)
        self.assertIn('"full_encoder_precision_accepted": False', self.source)
        self.assertIn('"whole_goal_complete": False', self.source)

    def test_kernel_metadata_carries_the_launcher_contract(self):
        for needle in ('"GM_argument_count": 5', '"dst_dtype": "float16"',
                       '"input_dtype": "float16"', '"kernel_symbol"',
                       '"obj_path"', '"obj_sha256"', '"block_dim"'):
            self.assertIn(needle, self.source)

    def test_only_the_four_erf_merger_activations_are_named(self):
        self.assertEqual(erf.MERGER_GELU_NAMES,
                         ("merger.act_fn", "deepstack_merger_list.0.act_fn",
                          "deepstack_merger_list.1.act_fn",
                          "deepstack_merger_list.2.act_fn"))
        self.assertEqual(erf.EXPECTED_TANH_GELU_MODULES, 27)
        self.assertEqual(erf.EXPECTED_ERF_GELU_MODULES, 4)
        self.assertEqual(erf.EXPECTED_TOTAL_GELU_MODULES, 31)


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.source = (HERE / "erf_gelu90.py").read_text()

    def test_adapter_api_is_the_root_contract(self):
        for method in ("def prepare(", "def set_enabled(", "def snapshot(", "def uninstall("):
            self.assertIn(method, self.source)
        self.assertIn("class ErfGeluAdapter90", self.source)
        self.assertIn("def install(", self.source)

    def test_patch_is_per_instance_bound_method_and_default_disabled(self):
        self.assertIn("from types import MethodType", self.source)
        self.assertIn("MethodType(self._make_forward(name), module)", self.source)
        self.assertIn("self.enabled = False", self.source)
        self.assertIn("if not self.enabled:", self.source)
        self.assertIn('"per_instance_forward_only": True', self.source)
        self.assertIn('"patches_torch_global": False', self.source)
        self.assertIn('"patches_torch_functional": False', self.source)
        self.assertIn('"patches_other_activations": False', self.source)
        self.assertIn('"patches_biased_linears": False', self.source)

    def test_restore_is_identity_guarded_and_drops_the_instance_attribute(self):
        self.assertIn('getattr(current, "__self__", None) is module', self.source)
        self.assertIn('getattr(current, "__func__", None) is wrapper.__func__', self.source)
        self.assertIn("delattr(module, \"forward\")", self.source)
        self.assertIn("was not restored blindly", self.source)

    def test_operand_guard_is_half_npu_and_fixed_geometry(self):
        self.assertIn('if input_.dtype is not torch.float16:', self.source)
        self.assertIn('input_.device.type not in ("npu", "privateuseone")', self.source)
        self.assertIn("int(input_.shape[1]) != REAL_HIDDEN", self.source)
        self.assertIn("if input_.numel() != numel:", self.source)

    def test_erf_mode_and_tanh_inventory_are_enforced(self):
        self.assertIn('if approximate != "none":', self.source)
        self.assertIn("_check_vision_inventory", self.source)
        self.assertIn("EXPECTED_TANH_GELU_MODULES", self.source)

    def test_launcher_call_keeps_the_five_pointer_abi(self):
        self.assertIn("[flat, self._aux[0], self._aux[1], self._aux[2], out]", self.source)
        self.assertIn("synchronize=True", self.source)

    def test_no_float_weight_or_teacher_path_exists(self):
        for forbidden in ("module.weight.float()", "weight.float()", "bfloat16"):
            self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()