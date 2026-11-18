#!/usr/bin/env python3
"""Standard-library contract tests for the Round83 blocked FP32-output GEMM.

No torch, no torch_npu, no TIK, no CANN, no device. These tests run on a machine
without an accelerator and therefore cannot and do not claim any numerical or
performance result.

What they DO establish: every DMA stride/length in the tiling plan corresponds to
the documented byte layout, all H3 geometries are legal and stay inside L1/L0C/UB,
the K loop really accumulates into one FP32 L0C (no Half partial, no FP32-UB
partial sum), the kernel keeps the proven 5-GM-pointer ABI without a bias operand,
and the host pack/unpack mapping is a pure permutation plus zero padding.
"""

import ast
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE / "blocked_fp32_gemm83.py"
SOURCE = MODULE.read_text(encoding="utf-8")

import blocked_fp32_gemm83 as g  # noqa: E402  (stdlib-only import)


def tree():
    return ast.parse(SOURCE)


class NoHeavyImports(unittest.TestCase):
    def test_module_imports_without_torch_or_tik(self):
        for node in tree().body:
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
                self.assertNotIn("torch", names)
                self.assertNotIn("torch_npu", names)
                self.assertFalse([n for n in names if n.startswith("tbe")])
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, ("torch", "torch_npu", "tbe", "numpy"))

    def test_heavy_imports_are_lazy_inside_functions(self):
        for name in ("torch", "tbe.common.platform"):
            found = False
            for node in ast.walk(tree()):
                if isinstance(node, ast.Import):
                    found = found or any(a.name == name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    found = found or node.module == name
            self.assertTrue(found, f"{name} must be imported lazily somewhere")
        # and never at module top level
        top = {a.name for node in tree().body if isinstance(node, ast.Import)
               for a in node.names}
        self.assertNotIn("torch", top)


class ShapeAndPlanTests(unittest.TestCase):
    def test_all_h3_geometries_are_legal_and_fit_the_buffers(self):
        for name, shape in g.H3_SHAPES.items():
            plan = g.plan_tiling(*shape)
            g._check_budget(plan)  # raises if L1/L0C/UB overflow
            m, k, n = shape
            self.assertEqual(plan["padded"]["m_pad"] % plan["tiles"]["mb"], 0, name)
            self.assertGreaterEqual(plan["padded"]["m_pad"], m, name)
            self.assertLess(plan["padded"]["m_pad"] - m, plan["tiles"]["mb"], name)

    def test_documented_padding_for_2043_rows(self):
        plan = g.plan_tiling(2043, 5120, 128)
        self.assertEqual(plan["padded"]["m_pad"], 2048)
        self.assertEqual(plan["padded"]["m_pad_rows"], 5)

    def test_grid_is_round_robin_within_thirty_cores(self):
        for shape in g.H3_SHAPES.values():
            plan = g.plan_tiling(*shape)
            grid = plan["grid"]
            self.assertEqual(grid["tasks"], grid["m_tiles"] * grid["n_tiles"])
            self.assertEqual(grid["cores"], min(30, grid["tasks"]))
            self.assertLessEqual(grid["cores"], g.PINNED_MAX_CORES)
            self.assertGreaterEqual(grid["steps"] * grid["cores"], grid["tasks"])
            self.assertGreater((grid["steps"] - 1) * grid["cores"], -1)
            if grid["steps"] > 1:
                self.assertLess((grid["steps"] - 1) * grid["cores"], grid["tasks"])

    def test_illegal_tiles_and_shapes_are_rejected(self):
        for bad in ((2043, 5000, 128), (2043, 5120, 100), (0, 5120, 128)):
            with self.assertRaises(g.BlockedGemmError):
                g.plan_tiling(*bad)
        for bad_mb in (100, 0, -16):
            with self.assertRaises(g.BlockedGemmError):
                g.plan_tiling(2043, 5120, 128, mb=bad_mb)
        with self.assertRaises(g.BlockedGemmError):
            g.plan_tiling(2043.0, 5120, 128)


class DmaUnitTests(unittest.TestCase):
    """A Half K0=16 row is 32 B and a Float32 K0=16 row is 64 B; data_move units
    are 32 B, so the A/B strides need no factor of 2 and the C stride does."""

    def test_a_load_units(self):
        for shape in g.H3_SHAPES.values():
            plan = g.plan_tiling(*shape)
            t, d, pd = plan["tiles"], plan["dma"]["a_load"], plan["padded"]
            self.assertEqual(d["nburst"], t["kb"] // 16)
            self.assertEqual(d["burst"] * 32, t["mb"] * 16 * 2)          # mb rows x 32 B
            self.assertEqual(d["src_stride"] * 32, (pd["m_pad"] - t["mb"]) * 16 * 2)
            self.assertEqual(d["dst_stride"], 0)

    def test_b_load_units(self):
        for shape in g.H3_SHAPES.values():
            plan = g.plan_tiling(*shape)
            t, d = plan["tiles"], plan["dma"]["b_load"]
            self.assertEqual(d["nburst"], t["kb"] // 16)
            self.assertEqual(d["burst"] * 32, t["nb"] * 16 * 2)
            self.assertEqual(d["src_stride"] * 32, (plan["logical"]["n"] - t["nb"]) * 16 * 2)
            self.assertEqual(d["dst_stride"], 0)

    def test_c_store_uses_doubled_float32_units(self):
        for shape in g.H3_SHAPES.values():
            plan = g.plan_tiling(*shape)
            t, d, pd, n1 = plan["tiles"], plan["dma"]["c_store"], plan["padded"], plan["padded"]["n1"]
            self.assertEqual(d["nburst"], t["nb"] // 16)
            self.assertEqual(d["burst"] * 32, t["mb"] * 16 * 4)          # mb rows x 64 B
            self.assertEqual(d["burst"], 2 * t["mb"])
            self.assertEqual(d["src_stride"], 0)
            self.assertEqual(d["dst_stride"], 2 * (pd["m_pad"] - t["mb"]))
            self.assertEqual(d["dst_stride"] * 32, (pd["m_pad"] - t["mb"]) * 16 * 4)
            self.assertEqual(n1 * 16, plan["logical"]["n"])

    def test_l0c_to_ub_burst_is_kilobytes(self):
        for shape in g.H3_SHAPES.values():
            plan = g.plan_tiling(*shape)
            mv, t = plan["dma"]["l0c_to_ub"], plan["tiles"]
            self.assertEqual(mv["block_mode"], "m")
            self.assertEqual(mv["nburst"], 1)
            self.assertEqual(mv["burst_len_kb"] * 1024, t["mb"] * t["nb"] * 4)

    def test_every_gm_offset_is_thirty_two_byte_aligned(self):
        for shape in g.H3_SHAPES.values():
            plan = g.plan_tiling(*shape)
            t, pd = plan["tiles"], plan["padded"]
            elem = {"a": 2, "b": 2, "c": 4}
            for mb_idx in range(plan["grid"]["m_tiles"]):
                for nb_idx in range(plan["grid"]["n_tiles"]):
                    base_m, base_n = mb_idx * t["mb"], nb_idx * t["nb"]
                    for kt in range(plan["accumulation"]["k_tiles"]):
                        a_off = (kt * (t["kb"] // 16) * pd["m_pad"] + base_m) * 16
                        b_off = (kt * (t["kb"] // 16) * plan["logical"]["n"] + base_n) * 16
                        self.assertEqual(a_off * elem["a"] % 32, 0)
                        self.assertEqual(b_off * elem["b"] % 32, 0)
                    c_off = ((base_n // 16) * pd["m_pad"] + base_m) * 16
                    self.assertEqual(c_off * elem["c"] % 32, 0)

    def test_no_tile_leaves_its_operand(self):
        for shape in g.H3_SHAPES.values():
            plan = g.plan_tiling(*shape)
            t, pd, n = plan["tiles"], plan["padded"], plan["logical"]["n"]
            for mb_idx in range(plan["grid"]["m_tiles"]):
                base_m = mb_idx * t["mb"]
                self.assertLessEqual(base_m + t["mb"], pd["m_pad"])
            for nb_idx in range(plan["grid"]["n_tiles"]):
                self.assertLessEqual(nb_idx * t["nb"] + t["nb"], n)
            self.assertEqual(plan["accumulation"]["k_tiles"] * t["kb"], plan["logical"]["k"])

    def test_gm_sizes_match_padded_layout(self):
        plan = g.plan_tiling(2043, 5120, 128)
        self.assertEqual(plan["gm_shapes"]["a"], [320, 2048, 16])
        self.assertEqual(plan["gm_shapes"]["b"], [320, 128, 16])
        self.assertEqual(plan["gm_shapes"]["c"], [8, 2048, 16])
        self.assertEqual(plan["gm_bytes"]["a"], 320 * 2048 * 16 * 2)
        self.assertEqual(plan["gm_bytes"]["c"], 8 * 2048 * 16 * 4)


class AccumulationTests(unittest.TestCase):
    def test_single_l0c_across_all_k_blocks_and_first_block_only_init(self):
        plans = json.loads(json.dumps({k: g.plan_tiling(*v) for k, v in g.H3_SHAPES.items()}))
        for name, plan in plans.items():
            acc = plan["accumulation"]
            self.assertTrue(acc["single_l0c_across_all_k_blocks"], name)
            self.assertTrue(acc["init_first_k_block_only"], name)
            self.assertFalse(acc["half_partial"], name)
            self.assertFalse(acc["fp32_ub_partial_sum"], name)
            self.assertEqual(acc["final_output_dtype"], "float32")
            self.assertGreater(acc["k_tiles"], 1)

    def test_source_emits_init_true_only_for_the_first_k_block(self):
        self.assertIn("init_l1out=(kt == 0)", SOURCE)
        # exactly one matmul emission site, so no duplicated/patchy accumulation
        self.assertEqual(SOURCE.count("t.matmul("), 1)
        self.assertNotIn("init_l1out=True", SOURCE.replace("init_l1out=(kt == 0)", ""))
        self.assertNotIn("init_l1out=False", SOURCE)

    def test_one_l0c_tensor_allocated_outside_the_k_loop(self):
        # the L0C tensor is created once per core, before the tile loop
        self.assertEqual(SOURCE.count('scope=tik.scope_cc'), 1)
        body = SOURCE.split("for step in range(steps):")[1]
        self.assertNotIn("scope=tik.scope_cc", body)

    def test_no_half_or_ub_intermediate_accumulation_exists(self):
        ub_tensors = []
        for node in ast.walk(tree()):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Tensor":
                dtype = (node.args[0].value
                         if node.args and isinstance(node.args[0], ast.Constant) else None)
                scope = None
                for kw in node.keywords:
                    if kw.arg == "scope":
                        # scope is written as tik.scope_ubuf (an Attribute), not a str
                        scope = (kw.value.value if isinstance(kw.value, ast.Constant)
                                 else getattr(kw.value, "attr", None))
                if scope == "scope_ubuf":
                    ub_tensors.append(dtype)
        self.assertEqual(ub_tensors, ["float32"],
                         "the only UB tensor must be the FP32 output block")
        self.assertNotIn(".half()", SOURCE)
        self.assertEqual(SOURCE.count("t.tensor_mov("), 1)

    def test_single_final_l0c_to_ub_move_happens_after_the_k_loop(self):
        k_loop_body = SOURCE.split("for kt in range(k_tiles):")[1].split("c_off =")[0]
        self.assertNotIn("tensor_mov", k_loop_body)
        self.assertIn("t.tensor_mov(cu, c0", SOURCE.split("c_off =")[1])


class KernelAbiTests(unittest.TestCase):
    def test_five_gm_pointer_abi_with_unused_aux(self):
        self.assertEqual(g.plan_tiling(2043, 5120, 128)["kernel_arg_order"],
                         ["a_gm_float16", "b_gm_float16", "aux0_gm_float16",
                          "aux1_gm_float16", "c_gm_float32"])
        self.assertIn('name="aux0_gm"', SOURCE)
        self.assertIn('name="aux1_gm"', SOURCE)
        self.assertIn("inputs=[gm[\"a\"], gm[\"b\"], gm[\"aux0\"], gm[\"aux1\"]]", SOURCE)
        self.assertIn("outputs=[gm[\"c\"]]", SOURCE)
        self.assertEqual(SOURCE.count('scope=tik.scope_gm'), 5)

    def test_no_bias_operand_and_no_hardware_bias_path(self):
        self.assertNotIn("bias=", SOURCE.split("def _emit_kernel")[1])
        self.assertNotIn("tensor_bias", SOURCE)
        self.assertNotIn("biasl", SOURCE)
        self.assertIn('"no_bias_operand": True', SOURCE)

    def test_no_fp32_weight_shadow_and_input_dtype_is_half(self):
        self.assertIn('t.Tensor("float16", (k1, m_pad, 16)', SOURCE)
        self.assertIn('t.Tensor("float16", (k1, n, 16)', SOURCE)
        self.assertNotIn('t.Tensor("float32", (k1,', SOURCE)
        self.assertIn('"no_fp32_weight_shadow": True', SOURCE)

    def test_thirty_core_guard_is_pinned_with_reason(self):
        self.assertEqual(g.PINNED_CORE_NUM, 30)
        self.assertEqual(g.PINNED_MAX_CORES, 30)
        self.assertIn('report["core_num"] != PINNED_CORE_NUM', SOURCE)
        self.assertIn("32 GiB of memory is not 32 cores", SOURCE)

    def test_compile_soc_is_the_verified_profile_not_a2(self):
        self.assertEqual(g.PINNED_COMPILE_SOC, "Ascend910B")
        self.assertIn("Intrinsic_mmad", SOURCE)
        self.assertNotIn("Ascend910_93", SOURCE)

    def test_execution_is_never_claimed(self):
        self.assertIn('"executed": False', SOURCE)
        self.assertIn('"not_a_speed_claim": True', SOURCE)
        self.assertIn('"full_encoder_precision_accepted": False', SOURCE)
        self.assertNotIn("kernel_launch_executed\": True", SOURCE)

    def test_host_contract_keys_required_by_the_root_probe(self):
        """The root's probe/runtime consume exactly these; drift must fail here."""
        body = SOURCE.split("def _describe_cached")[1].split("def _emit_kernel")[0]
        for key in ('"obj_path"', '"obj_sha256"', '"kernel_symbol"', '"block_dim"',
                    '"dst_dtype": "float32"', '"GM_argument_count": 5',
                    '"gm_shapes"', '"logical_shape"', '"padded_shape"'):
            self.assertIn(key, body, key)
        self.assertIn('"uses_fp32_bias": False', body)
        # dst_dtype must be exactly float32: the whole point is no narrowing here
        self.assertIn('"dst_dtype": "float32"', body)
        self.assertNotIn('"dst_dtype": "float16"', body)

    def test_weight_cache_signature_includes_storage_offset(self):
        self.assertIn("int(weight.storage_offset())", SOURCE)
        # message is wrapped across source lines, so match the fragment
        self.assertIn("to silently re-cache a mutated parameter", SOURCE)
        self.assertIn("shared ``data_ptr`` alone must never be treated", SOURCE)


class HostPackingTests(unittest.TestCase):
    """Verify the documented element mapping in pure python (no torch needed)."""

    @staticmethod
    def reference_pack_a(a, m_pad):
        m, k = len(a), len(a[0])
        out = [[[0] * 16 for _ in range(m_pad)] for _ in range(k // 16)]
        for row in range(m):
            for col in range(k):
                out[col // 16][row][col % 16] = a[row][col]
        return out

    @staticmethod
    def reference_pack_w(w):
        n, k = len(w), len(w[0])
        return [[[w[row][col * 16 + k0] for k0 in range(16)] for row in range(n)]
                for col in range(k // 16)]

    def test_pack_a_is_a_permutation_plus_zero_padding(self):
        m, k = 5, 32
        a = [[row * k + col for col in range(k)] for row in range(m)]
        packed = self.reference_pack_a(a, 8)
        # values survive exactly
        for row in range(m):
            for col in range(k):
                self.assertEqual(packed[col // 16][row][col % 16], a[row][col])
        # padded rows are zero
        for col_block in range(k // 16):
            for row in range(m, 8):
                self.assertEqual(packed[col_block][row], [0] * 16)

    def test_pack_w_keeps_every_weight_value(self):
        n, k = 3, 32
        w = [[row * k + col for col in range(k)] for row in range(n)]
        packed = self.reference_pack_w(w)
        for row in range(n):
            for col in range(k):
                self.assertEqual(packed[col // 16][row][col % 16], w[row][col])

    def test_unpack_c_is_the_inverse_permutation_and_crops_padding(self):
        n1, m_pad, m = 2, 4, 3
        c = [[[1000 * n + 10 * row + e for e in range(16)] for row in range(m_pad)]
             for n in range(n1)]
        out = [[c[col // 16][row][col % 16] for col in range(n1 * 16)] for row in range(m)]
        for row in range(m):
            for col in range(n1 * 16):
                self.assertEqual(out[row][col], c[col // 16][row][col % 16])
        self.assertEqual(len(out), m)

    def test_helpers_are_lazy_and_shape_checked(self):
        for name in ("pack_a", "pack_w", "unpack_c"):
            body = SOURCE.split(f"def {name}(")[1].split("\ndef ")[0]
            self.assertIn("import torch", body, name)
            self.assertIn("raise BlockedGemmError", body, name)


class ProvenanceTests(unittest.TestCase):
    def test_plan_is_deterministic_and_serializable(self):
        a = g.plan_tiling(2043, 5120, 128)
        b = g.plan_tiling(2043, 5120, 128)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_manifest_pins_the_named_files(self):
        manifest = json.loads((HERE / "manifest83.json").read_text())
        self.assertFalse(manifest["executed_on_npu"])
        self.assertFalse(manifest["compiled"])
        self.assertFalse(manifest["proves_precision"])
        self.assertFalse(manifest["proves_speed"])
        for name, meta in manifest["files"].items():
            path = HERE / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(hashlib_sha(path.read_text(encoding="utf-8")), meta["sha256"], name)
            self.assertEqual(path.stat().st_size, meta["bytes"], name)
        self.assertEqual(manifest["module_sha256"],
                         hashlib_sha(MODULE.read_text(encoding="utf-8")))
        for key, shape in manifest["h3_shapes"].items():
            self.assertEqual(tuple(shape), g.H3_SHAPES[key])
        self.assertEqual(manifest["tiling_defaults"], {"mb": 128, "nb": 128, "kb": 128})

    def test_smoke_geometry_exercises_two_k_blocks(self):
        plan = g.plan_tiling(*g.SUGGESTED_SMOKE_GEOMETRY)
        self.assertEqual(plan["accumulation"]["k_tiles"], 2)
        self.assertEqual(plan["grid"]["tasks"], 1)
        self.assertEqual(plan["grid"]["cores"], 1)

    def test_module_has_no_side_effects_at_import(self):
        top_calls = [n for n in tree().body if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call)]
        self.assertEqual(top_calls, [])


def hashlib_sha(text):
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main(verbosity=2)