#!/usr/bin/env python3
"""Stdlib-only tests for the Round87 vision FP32-bias kernel source.

No torch, TIK, CANN, NPU or network is used or needed here; this file runs on a
plain laptop. It verifies the plan/tiling/DMA arithmetic, the layout index maps,
the kernel-emission ORDER (bias in FP32 before the single Half cast, no hardware
bias operand), the source pins and the cache contract. It does NOT build or run
a kernel: no numeric or speed claim is made by this suite.
"""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VISION = HERE / "vision_bias87.py"

sys.path.insert(0, str(HERE))
v = importlib.import_module("vision_bias87")


class ModuleScopeTests(unittest.TestCase):
    def test_module_scope_imports_are_stdlib_only(self):
        tree = ast.parse(VISION.read_text())
        allowed = {"__future__", "hashlib", "json", "os", "pathlib", "tempfile", "time"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed, alias.name)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or "").split(".")[0], allowed, node.module)

    def test_tik_and_torch_are_lazy(self):
        source = VISION.read_text()
        tree = ast.parse(source)
        module_level_calls = {n.func.id for n in ast.walk(tree)
                              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertNotIn("_tik", module_level_calls)
        # TIK/torch only appear inside functions.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    names = ([a.name for a in child.names] if isinstance(child, ast.Import)
                             else [child.module or ""])
                    for name in names:
                        self.assertNotIn(name.split(".")[0], {"torch", "tbe", "torch_npu"})


class GeometryTests(unittest.TestCase):
    def test_six_real_vision_geometries_resolve_to_thirty_cores(self):
        rows = {row["name"]: row for row in v.planned_geometries()}
        self.assertEqual(set(rows), {"qkv", "proj", "mlp1", "mlp2", "merger1", "merger2"})
        expected = {
            # name: (m, k, n, m_pad, k_tiles, tasks, steps)
            "qkv": (8064, 1152, 3456, 8064, 9, 1701, 57),
            "proj": (8064, 1152, 1152, 8064, 9, 567, 19),
            "mlp1": (8064, 1152, 4352, 8064, 9, 2142, 72),
            "mlp2": (8064, 4352, 1152, 8064, 34, 567, 19),
            "merger1": (2016, 4608, 4608, 2048, 36, 576, 20),
            "merger2": (2016, 4608, 5120, 2048, 36, 640, 22),
        }
        for name, (m, k, n, m_pad, k_tiles, tasks, steps) in expected.items():
            row = rows[name]
            self.assertEqual((row["m"], row["k"], row["n"]), (m, k, n), name)
            self.assertEqual(row["m_pad"], m_pad, name)
            self.assertEqual(row["k_tiles"], k_tiles, name)
            self.assertEqual(row["tasks"], tasks, name)
            self.assertEqual(row["steps"], steps, name)
            self.assertEqual(row["cores"], 30, name)
            self.assertTrue(row["kernel_name"].startswith("vision_bias87_m"), name)

    def test_real_vision_mlp_width_is_4304_padded_to_4352(self):
        self.assertEqual(v.VISION_INTERMEDIATE_4304, 4304)
        self.assertEqual(v.VISION_INTERMEDIATE_PADDED_4352, 4352)
        self.assertEqual(v.pad_n(4304), 4352)
        self.assertEqual(v.pad_k(4304), 4352)
        rows = {row["name"]: row for row in v.planned_geometries()}
        self.assertEqual(rows["mlp1"]["n"], 4352)
        self.assertEqual(rows["mlp1"]["n_logical"], 4304)
        self.assertEqual(rows["mlp2"]["k"], 4352)
        self.assertEqual(rows["mlp2"]["k_logical"], 4304)
        # The Round84 stale MLP width must never appear as a kernel operand.
        for row in v.planned_geometries():
            self.assertNotIn(3456, (row["k"], row["n"]) if row["name"] != "qkv" else (0,))
            self.assertNotIn(27648, (row["k"], row["n"], row["k_logical"], row["n_logical"]))

    def test_unpadded_k_or_n_is_refused_with_the_padding_hint(self):
        with self.assertRaises(v.VisionBiasError) as ctx:
            v.plan_tiling(8064, 4304, 1152)
        self.assertIn("4352", str(ctx.exception))
        with self.assertRaises(v.VisionBiasError):
            v.plan_tiling(8064, 1152, 4304)
        with self.assertRaises(v.VisionBiasError):
            v.plan_tiling(8064, 1200, 1152)

    def test_m_padding_rounds_up_and_is_visible(self):
        self.assertEqual(v.pad_m(2016), 2048)
        self.assertEqual(v.pad_m(8064), 8064)
        plan = v.plan_tiling(2016, 4608, 4608)
        self.assertEqual(plan["padded"]["m_pad_rows"], 32)
        self.assertTrue(plan["epilogue"]["padded_output_rows_are_bias_not_zero"])

    def test_bad_scalars_are_rejected(self):
        for bad in (0, -1, 1.5, True, "128"):
            with self.assertRaises(v.VisionBiasError):
                v.plan_tiling(bad, 128, 128)
        with self.assertRaises(v.VisionBiasError):
            v.plan_tiling(128, 256, 128, mb=100)


class MultiKTests(unittest.TestCase):
    def test_multi_k_geometry_keeps_one_static_accumulator(self):
        for m, k, n, k_tiles in ((128, 256, 128, 2), (8064, 1152, 1152, 9),
                                 (8064, 4352, 1152, 34), (2016, 4608, 5120, 36)):
            plan = v.plan_tiling(m, k, n)
            acc = plan["accumulation"]
            self.assertEqual(acc["k_tiles"], k_tiles, (m, k, n))
            self.assertTrue(acc["single_l0c_across_all_k_blocks"])
            self.assertTrue(acc["init_first_k_block_only"])
            self.assertFalse(acc["half_partial"])
            self.assertFalse(acc["fp32_ub_partial_sum"])

    def test_two_k_block_smoke_geometry_is_legal(self):
        m, k, n = v.SUGGESTED_SMOKE_GEOMETRY
        plan = v.plan_tiling(m, k, n)
        self.assertEqual(plan["accumulation"]["k_tiles"], 2)
        self.assertEqual(plan["grid"]["tasks"], 1)
        self.assertEqual(plan["padded"]["m_pad"], 128)
        self.assertLessEqual(plan["buffers"]["ub_bytes"], v.PINNED_UB_SAFE_BYTES)


class DmaTests(unittest.TestCase):
    def test_half_output_store_stride_is_not_doubled(self):
        plan = v.plan_tiling(2016, 4608, 5120)
        store = plan["dma"]["c_store"]
        m_pad, mb = plan["padded"]["m_pad"], plan["tiles"]["mb"]
        self.assertEqual(store["dst_stride"], m_pad - mb)
        self.assertEqual(store["dst_stride"], store["dst_stride"])
        self.assertNotEqual(store["dst_stride"], 2 * (m_pad - mb))
        self.assertEqual(store["fp32_output_stride_would_be"], 2 * (m_pad - mb))
        self.assertEqual(store["units_bytes"], 32)
        self.assertEqual(store["half_row_units"], 1)

    def test_input_strides_follow_the_verified_round83_formulas(self):
        for m, k, n in ((8064, 1152, 3456), (8064, 4352, 1152), (2016, 4608, 5120)):
            plan = v.plan_tiling(m, k, n)
            mb = plan["tiles"]["mb"]
            self.assertEqual(plan["dma"]["a_load"]["src_stride"], plan["padded"]["m_pad"] - mb)
            self.assertEqual(plan["dma"]["b_load"]["src_stride"], n - mb)
            self.assertEqual(plan["dma"]["a_load"]["nburst"], plan["tiles"]["kb"] // 16)
            self.assertEqual(plan["dma"]["a_load"]["burst"], mb)
            self.assertEqual(plan["dma"]["b_load"]["burst"], mb)
            self.assertEqual(plan["dma"]["c_store"]["nburst"], plan["tiles"]["nb"] // 16)

    def test_bias_and_l0c_moves_match_the_executed_micro_kernel(self):
        plan = v.plan_tiling(8064, 1152, 1152)
        nb, mb = plan["tiles"]["nb"], plan["tiles"]["mb"]
        self.assertEqual(plan["dma"]["bias_load"]["nburst"], 1)
        self.assertEqual(plan["dma"]["bias_load"]["burst"], nb * 4 // 32)     # 32 B units
        self.assertEqual(plan["dma"]["l0c_to_ub"]["block_mode"], "m")
        self.assertEqual(plan["dma"]["l0c_to_ub"]["nburst"], 1)
        self.assertEqual(plan["dma"]["l0c_to_ub"]["burst_len_kb"], mb * nb * 4 // 1024)

    def test_buffers_fit_the_pinned_910a_generation_limits(self):
        for row in v.planned_geometries():
            plan = v.plan_tiling(row["m"], row["k"], row["n"])
            buf = plan["buffers"]
            self.assertLessEqual(buf["l1_a_bytes"] + buf["l1_b_bytes"], v.PINNED_L1_SIZE)
            self.assertLessEqual(buf["l0c_bytes"], v.PINNED_L0C_SIZE)
            self.assertLessEqual(buf["ub_bytes"], v.PINNED_UB_SAFE_BYTES)
            # c32 + c16 + bias must be the whole UB footprint.
            self.assertEqual(
                buf["ub_bytes"],
                buf["ub_c32_bytes"] + buf["ub_c16_bytes"] + buf["ub_bias_bytes"])


class LayoutTests(unittest.TestCase):
    def test_pack_a_index_map_is_bijective_and_matches_the_transpose_order(self):
        m, k = 128, 256
        seen = {}
        for mi in range(0, m, 5):
            for ki in range(k):
                key = v.pack_a_index(mi, ki)
                self.assertNotIn(key, seen)
                seen[key] = (mi, ki)
        self.assertEqual(len(seen), len(range(0, m, 5)) * k)
        # packed_a[k1, m, k0] fits the documented GM shape; the full map is a
        # bijection because (m, k) -> (k1, m, k0) is a reshape+transpose.
        full = {v.pack_a_index(mi, ki) for mi in range(16) for ki in range(k)}
        self.assertEqual(len(full), 16 * k)
        self.assertEqual(len(full), (k // 16) * 16 * 16)
        for (k1, mi, k0) in seen:
            self.assertLess(k1, k // 16)
            self.assertGreaterEqual(k1, 0)
            self.assertLess(mi, m)
            self.assertLess(k0, 16)

    def test_pack_w_index_map_is_bijective(self):
        # Full enumeration on a small geometry (cheap), then the exact index range
        # and element count of the real 1152x4352 weight are checked arithmetically.
        n, k = 64, 64
        keys = {v.pack_w_index(ni, ki) for ni in range(n) for ki in range(k)}
        self.assertEqual(len(keys), n * k)
        for (k1, ni, k0) in keys:
            self.assertTrue(0 <= k1 < k // 16 and 0 <= ni < n and 0 <= k0 < 16)
        real_n, real_k = 1152, 4352
        self.assertEqual(len({v.pack_w_index(ni, ki)
                              for ni in range(0, real_n, 97)
                              for ki in range(0, real_k, 83)}),
                         len(range(0, real_n, 97)) * len(range(0, real_k, 83)))
        self.assertEqual((real_k // 16) * real_n * 16, real_n * real_k)

    def test_output_index_map_matches_unpack_c(self):
        m, n = 2016, 5120
        keys = {v.c_index(mi, ni) for mi in range(0, m, 7) for ni in range(0, n, 5)}
        self.assertEqual(len(keys), len(range(0, m, 7)) * len(range(0, n, 5)))
        for (n1, mi, n0) in keys:
            self.assertLess(n1, n // 16)
            self.assertLess(n0, 16)

    def test_host_layouts_and_kernel_arg_order_are_documented(self):
        layouts = v.host_layouts()
        self.assertEqual(set(layouts), {"a", "b", "bias", "c"})
        self.assertEqual(layouts["a"]["gm"], "(K1, M_pad, 16) float16")
        self.assertEqual(layouts["bias"]["gm"], "(N,) float32")
        self.assertEqual(layouts["c"]["gm"], "(N1, M_pad, 16) float16")
        self.assertEqual(v.GM_ARG_ORDER,
                         ("a_gm_float16", "b_gm_float16", "bias_gm_float32",
                          "aux_gm_float16", "c_gm_float16"))


class EmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = VISION.read_text()
        cls.tree = ast.parse(cls.source)
        cls.emit = next(n for n in cls.tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "_emit_kernel")

    def _calls(self, node):
        out = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                out.append((child.func.attr, getattr(child, "lineno", 0)))
        return out

    def test_exactly_one_half_cast_and_it_is_last_in_the_epilogue(self):
        calls = self._calls(self.emit)
        h_cast = [line for name, line in calls if name == "h_cast"]
        self.assertEqual(len(h_cast), 1, "the kernel must narrow exactly once")
        vec_add = [line for name, line in calls if name == "vec_add"]
        tensor_mov = [line for name, line in calls if name == "tensor_mov"]
        self.assertEqual(len(vec_add), 1, "one vec_add call site inside the group loop")
        self.assertLess(tensor_mov[0], vec_add[0])
        self.assertGreater(h_cast[0], vec_add[0])

    def test_h_cast_is_fp32_to_fp16_with_round_mode_none(self):
        for node in ast.walk(self.emit):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "h_cast":
                self.assertEqual(len(node.args), 3)
                self.assertEqual(ast.literal_eval(node.args[2]), "none")

    def test_bias_is_added_in_fp32_before_any_narrowing(self):
        source = ast.get_source_segment(self.source, self.emit)
        self.assertIn("c32[group, :, :]", source)
        self.assertIn("biasub[group * 16]", source)
        self.assertIn("mb, 2, 2, 0", source)
        # The vec_add dst/src are the FP32 UB tensor, not the Half one.
        self.assertNotIn("c16[group", source)

    def test_no_hardware_bias_operand_is_used(self):
        for node in ast.walk(self.emit):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "matmul":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                self.assertIn("bias", kwargs)
                self.assertIsNone(ast.literal_eval(kwargs["bias"]))
                self.assertIn("init_l1out", kwargs)

    def test_k_loop_is_static_and_only_the_first_block_initialises(self):
        source = ast.get_source_segment(self.source, self.emit)
        self.assertIn("for kt in range(k_tiles):", source)
        self.assertIn("init_l1out=(kt == 0)", source)
        # The outer tile step is the only dynamic loop.
        self.assertIn("with t.for_range(0, steps) as step:", source)
        self.assertNotIn("for step in range(", source)

    def test_single_l0c_tensor_outside_the_k_loop(self):
        source = ast.get_source_segment(self.source, self.emit)
        self.assertEqual(source.count('name="c_l0c"'), 1)
        self.assertEqual(source.count('scope=tik.scope_cc'), 1)
        # c_l0c is declared before the K loop and is not re-created per block.
        self.assertLess(source.index('name="c_l0c"'), source.index("for kt in range(k_tiles)"))

    def test_aux_operand_is_passed_explicitly_to_keep_the_five_gm_abi(self):
        source = self.source
        self.assertIn("inputs=[gm[\"a\"], gm[\"b\"], gm[\"bias\"], gm[\"aux\"]]", source)
        self.assertIn("outputs=[gm[\"c\"]]", source)
        self.assertEqual(v.GM_ARG_ORDER.__len__(), 5)

    def test_emission_has_no_path_or_delete_operations(self):
        forbidden = ("rmtree", "os.remove", "os.unlink", "os.rename", "os.rmdir")
        for token in forbidden:
            self.assertNotIn(token, self.source)


class CacheContractTests(unittest.TestCase):
    def test_build_uses_a_lock_and_refuses_to_overwrite_outputs(self):
        source = VISION.read_text()
        self.assertIn('".vision87.compile.lock"', source)
        self.assertIn("fcntl.flock", source)
        self.assertIn('refusing to overwrite existing kernel directory', source)
        self.assertIn("tempfile.mkdtemp(prefix=\"vision_bias87_build_\"", source)
        self.assertIn("shutil.copyfile", source)

    def test_cache_reuse_validates_block_dim(self):
        source = VISION.read_text()
        self.assertIn('meta.get("blockDim", -1)) != plan["grid"]["cores"]', source)
        self.assertIn("refusing to reuse a mismatched binary", source)

    def test_spec_file_is_repaired_atomically_and_never_moved_outside(self):
        source = VISION.read_text()
        self.assertIn('.json.tmp87', source)
        self.assertIn("os.replace(tmp, resolved_spec)", source)
        self.assertIn("if resolved_spec != intended", source)
        self.assertIn("tmp.resolve(strict=False).parent != intended.parent", source)

    def test_failed_build_retains_its_workspace_instead_of_deleting_it(self):
        source = VISION.read_text()
        self.assertIn("is retained for diagnosis", source)

    def test_spec_epoch_and_non_claim_flags_are_present(self):
        source = VISION.read_text()
        for token in ('"built_unix_epoch"', '"cache_source_epoch"', '"executed": False',
                      '"full_encoder_precision_accepted": False',
                      '"whole_goal_complete": False', '"not_a_speed_claim": True'):
            self.assertIn(token, source)


class SourcePinTests(unittest.TestCase):
    def test_reference_pins_are_the_reviewed_hashes(self):
        pins = v.reference_pins()
        self.assertEqual(pins["reference_base_sha256"],
                         "78896adadaf47801dd92d69bdfce394d2628f7885be643dd568647c8b4c8c2ee")
        self.assertEqual(pins["reference_bias_epilogue_sha256"],
                         "2f6a4470f9abc5f43028f69ddd6e012c29c2a7907bb0f6527667063bf81c8306")
        self.assertEqual(pins["reference_executed_bias_micro_sha256"],
                         "124ba029a518e26b9b79051275a83144e702ebf016defe64085d8b3fa05ee03a")
        self.assertEqual(pins["pinned_core_num"], 30)
        self.assertEqual(pins["pinned_compile_soc"], "Ascend910B")
        self.assertEqual(pins["kernel_prefix"], "vision_bias87_")

    def test_every_pinned_reference_source_is_present_and_matching(self):
        status = v.reference_status()
        self.assertEqual(status["missing"], [])
        self.assertTrue(status["all_pinned_sources_present_and_matching"], status)
        self.assertEqual(sorted(status["verified"]),
                         sorted(v.REFERENCE_SOURCES))
        self.assertTrue(v.check_reference_sources(strict=True)["all_pinned_sources_present_and_matching"])

    def test_a_modified_reference_copy_fails_closed(self):
        original = v._reference_candidates
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "blocked_fp32_gemm83.py"
            tampered.write_text("# tampered copy\n")
            v._reference_candidates = lambda: [tampered]
            try:
                status = v.reference_status()
                name = "blocked_fp32_gemm83.py"
                self.assertFalse(status["sources"][name]["matches_pin"])
                with self.assertRaises(v.VisionBiasError) as ctx:
                    v.check_reference_sources(strict=True)
                self.assertIn("no longer matches the reviewed evidence", str(ctx.exception))
                # strict=False is only for diagnosis and must not raise.
                self.assertFalse(v.check_reference_sources(strict=False)
                                 ["all_pinned_sources_present_and_matching"])
            finally:
                v._reference_candidates = original

    def test_missing_reference_copy_does_not_raise_but_is_reported(self):
        original = v._reference_candidates
        with tempfile.TemporaryDirectory() as tmp:
            v._reference_candidates = lambda: [Path(tmp) / "absent.py"]
            try:
                status = v.reference_status()
                self.assertEqual(len(status["missing"]), len(v.REFERENCE_SOURCES))
                self.assertFalse(status["all_pinned_sources_present_and_matching"])
                check = v.check_reference_sources(strict=True)
                self.assertFalse(check["all_pinned_sources_present_and_matching"])
            finally:
                v._reference_candidates = original


class NonClaimTests(unittest.TestCase):
    def test_module_claims_no_precision_or_speed_result(self):
        source = VISION.read_text()
        self.assertIn("SOURCE ONLY", source)
        self.assertIn("not_a_speed_claim", source)
        self.assertIn("full_encoder_precision_accepted", source)
        self.assertNotIn("proves_full_encoder_precision\": True", source)
        # No fabricated accuracy tolerance constant.
        for token in ("atol=", "rtol=", "tolerance =", "PASS_THRESHOLD"):
            self.assertNotIn(token, source)

    def test_reference_pins_resolve_to_the_committed_round83_files(self):
        base = REPO / "artifacts/diagnostics/round83/blocked/blocked_fp32_gemm83.py"
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(),
                         v.REFERENCE_BASE_SHA256)
        epilogue = REPO / "artifacts/diagnostics/round83/blocked_bias_epilogue83.py"
        self.assertEqual(hashlib.sha256(epilogue.read_bytes()).hexdigest(),
                         v.REFERENCE_BIAS_EPILOGUE_SHA256)
        micro = REPO / "artifacts/diagnostics/round83/bias/tik_bias_epilogue83.py"
        self.assertEqual(hashlib.sha256(micro.read_bytes()).hexdigest(),
                         v.REFERENCE_EXECUTED_BIAS_MICRO_SHA256)
        runtime = REPO / "artifacts/diagnostics/round83/runtime5gm83.py"
        self.assertEqual(hashlib.sha256(runtime.read_bytes()).hexdigest(),
                         v.REFERENCE_RUNTIME_MODULE_SHA256)


class ManifestTests(unittest.TestCase):
    def test_manifest_matches_the_module_and_is_non_claiming(self):
        manifest_path = HERE / "manifest87.json"
        self.assertTrue(manifest_path.is_file(), "manifest87.json must be checked in")
        manifest = json.loads(manifest_path.read_text())
        recorded = manifest["files"]["vision_bias87.py"]
        self.assertEqual(recorded["sha256"],
                         hashlib.sha256(VISION.read_bytes()).hexdigest())
        self.assertEqual(recorded["bytes"], VISION.stat().st_size)
        self.assertFalse(manifest["compiled"])
        self.assertFalse(manifest["executed_on_npu"])
        self.assertFalse(manifest["proves_precision"])
        self.assertFalse(manifest["proves_speed"])
        self.assertFalse(manifest["no_production_files_modified"] is False)
        self.assertEqual(len(manifest["planned_geometries"]), 6)

    def test_manifest_planned_geometries_match_the_module(self):
        manifest = json.loads((HERE / "manifest87.json").read_text())
        self.assertEqual(manifest["planned_geometries"], v.planned_geometries())


if __name__ == "__main__":
    unittest.main()