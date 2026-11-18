#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round96 fused two-Half-probability PV kernel.

No torch, torch_npu, TIK or CANN is imported: these are structural contracts about
tiling, budgets, GM offsets, the five-pointer ABI, the Half-vs-FP32 store geometry
and the arithmetic order. They are not numerical results and not speed claims.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
PV_PATH = HERE / "pv_high_low96.py"
MANIFEST = HERE / "manifest96.json"
ROUND85_BATCH = HERE.parent.parent / "round85" / "batch"


def _load_pv():
    spec = importlib.util.spec_from_file_location("pv_high_low96", PV_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pv_high_low96"] = module
    spec.loader.exec_module(module)
    return module


def _load_batch85():
    if str(ROUND85_BATCH) not in sys.path:
        sys.path.insert(0, str(ROUND85_BATCH))
    import batch_gemm85

    return batch_gemm85


pv = _load_pv()
FULL = (16, 512, 4096, 128)
TAIL = (16, 448, 4096, 128)


class TilingContracts(unittest.TestCase):
    def test_real_shapes_pad_to_512_and_fill_all_30_cores(self):
        for shape in (FULL, TAIL):
            plan = pv.plan_tiling(*shape)
            pv._check_budget(plan)
            self.assertEqual(plan["padded"]["m_pad"], 512)
            self.assertEqual(plan["padded"]["k1"], 256)
            self.assertEqual(plan["padded"]["n1"], 8)
            self.assertEqual(plan["grid"]["tasks"], 64)
            self.assertEqual(plan["grid"]["cores"], 30)
            self.assertEqual(plan["grid"]["steps"], 3)
            self.assertEqual(plan["accumulation"]["k_tiles"], 32)
            self.assertLessEqual(plan["grid"]["cores"], pv.PINNED_MAX_CORES)

    def test_padded_m_rows_are_the_only_difference_between_full_and_tail(self):
        full, tail = pv.plan_tiling(*FULL), pv.plan_tiling(*TAIL)
        self.assertEqual(full["padded"]["m_pad"], tail["padded"]["m_pad"])
        self.assertEqual(full["padded"]["m_pad_rows"], 0)
        self.assertEqual(tail["padded"]["m_pad_rows"], 64)
        self.assertEqual(full["grid"], tail["grid"])
        self.assertEqual(full["head_stride_elems"], tail["head_stride_elems"])

    def test_tiling_agrees_with_the_executed_round85_plan(self):
        """A/B layouts, DMA and grid must be byte-identical to the R85 conventions."""
        r85 = _load_batch85()
        for shape in (FULL, TAIL):
            mine, theirs = pv.plan_tiling(*shape), r85.plan_tiling(*shape)
            for key in ("logical", "tiles", "padded", "grid"):
                self.assertEqual(mine[key], theirs[key], f"{key} drifted for {shape}")
            self.assertEqual(mine["head_stride_elems"]["hi"],
                             theirs["head_stride_elems"]["a"])
            self.assertEqual(mine["head_stride_elems"]["lo"],
                             theirs["head_stride_elems"]["a"])
            self.assertEqual(mine["head_stride_elems"]["b"],
                             theirs["head_stride_elems"]["b"])
            self.assertEqual(mine["head_stride_elems"]["out"],
                             theirs["head_stride_elems"]["c"])
            self.assertEqual(mine["dma"]["a_load"], theirs["dma"]["a_load"])
            self.assertEqual(mine["dma"]["b_load"], theirs["dma"]["b_load"])
            self.assertEqual(mine["dma"]["l0c_to_ub"], theirs["dma"]["l0c_to_ub"])
            self.assertEqual(mine["buffers"]["l1_a_hi_bytes"],
                             theirs["buffers"]["l1_a_bytes"])
            self.assertEqual(mine["buffers"]["l1_b_bytes"],
                             theirs["buffers"]["l1_b_bytes"])
            self.assertEqual(mine["buffers"]["l0c_high_bytes"],
                             theirs["buffers"]["l0c_bytes"])

    def test_half_store_is_one_32b_row_not_the_float32_two_unit_row(self):
        """The P@V output is Half, so the NZ store stride must NOT carry the FP32 factor."""
        r85 = _load_batch85()
        for shape in (FULL, TAIL):
            mine, theirs = pv.plan_tiling(*shape), r85.plan_tiling(*shape)
            self.assertEqual(mine["dma"]["out_store"]["burst"], mine["tiles"]["mb"])
            self.assertEqual(theirs["dma"]["c_store"]["burst"],
                             2 * theirs["tiles"]["mb"])
            self.assertEqual(mine["dma"]["out_store"]["dst_stride"],
                             mine["padded"]["m_pad"] - mine["tiles"]["mb"])
            self.assertEqual(theirs["dma"]["c_store"]["dst_stride"],
                             2 * (theirs["padded"]["m_pad"] - theirs["tiles"]["mb"]))
            self.assertEqual(mine["dma"]["out_store"]["nburst"], 8)

    def test_budgets_fit_and_repeat_immediate_is_respected(self):
        for shape in (FULL, TAIL):
            plan = pv.plan_tiling(*shape)
            buf, vec = plan["buffers"], plan["vector"]
            self.assertLessEqual(buf["l1_total_bytes"], pv.PINNED_L1_SIZE)
            self.assertLessEqual(buf["l0c_total_bytes"], pv.PINNED_L0C_SIZE)
            self.assertLessEqual(buf["ub_total_bytes"], pv.PINNED_UB_SIZE)
            self.assertEqual(buf["l0c_total_bytes"], 2 * buf["l0c_high_bytes"])
            self.assertEqual(buf["ub_total_bytes"],
                             2 * buf["ub_high_bytes"] + buf["ub_half_bytes"])
            self.assertEqual(vec["repeats_per_chunk"] * vec["mask"],
                             vec["elements_per_chunk"])
            self.assertEqual(vec["chunks_per_tile"] * vec["elements_per_chunk"],
                             buf["ub_high_bytes"] // 4)
            # 16384 elements / 64 = 256 would overflow the 8-bit repeat immediate.
            self.assertEqual(buf["ub_high_bytes"] // 4 // vec["mask"], 256)
            self.assertLessEqual(vec["repeats_per_chunk"], vec["max_repeat_immediate"])

    def test_bad_geometry_is_refused_not_guessed(self):
        for kwargs in ({"mb": 100}, {"kb": 96}, {"nb": 96}):
            with self.assertRaises(pv.FusedPvError):
                pv.plan_tiling(*FULL, **kwargs)
        with self.assertRaises(pv.FusedPvError):
            pv.plan_tiling(16, 512, 4000, 128)
        with self.assertRaises(pv.FusedPvError):
            pv.plan_tiling(16, 512, 4096, 96)
        with self.assertRaises(pv.FusedPvError):
            pv.plan_tiling(0, 512, 4096, 128)
        with self.assertRaises(pv.FusedPvError):
            pv.plan_tiling(True, 512, 4096, 128)


class ArithmeticOrderContracts(unittest.TestCase):
    def test_two_independent_l0c_accumulators_over_the_full_k_axis(self):
        acc = pv.plan_tiling(*FULL)["accumulation"]
        self.assertEqual(acc["products"], ["high", "low"])
        self.assertTrue(acc["independent_l0c_per_product"])
        self.assertTrue(acc["single_l0c_across_all_k_blocks"])
        self.assertTrue(acc["init_first_k_block_only"])
        self.assertEqual(acc["k_tiles"], 32)
        self.assertFalse(acc["half_partial"])
        self.assertFalse(acc["fp32_ub_partial_sum"])
        self.assertFalse(acc["mid_product_gm_output"])
        self.assertFalse(acc["host_combine_required"])
        self.assertEqual(acc["final_output_dtype"], "float16")
        self.assertTrue(acc["one_half_cast"])

    def test_combine_is_high_plus_low_over_1024_then_one_half_cast(self):
        plan = pv.plan_tiling(*FULL)
        self.assertEqual(plan["low_scale"], 1024.0)
        self.assertEqual(plan["low_downscale"], 1.0 / 1024.0)
        self.assertTrue(plan["vector"]["scale_is_power_of_two"])
        order = plan["vector"]["order"]
        self.assertEqual(order[0], "tensor_mov_low_L0C_to_UB")
        self.assertEqual(order[1], "tensor_mov_high_L0C_to_UB")
        self.assertLess(order.index("vec_muls_low_downscale"), order.index("h_cast_once_none"))
        self.assertEqual(order[-1], "data_move_half_store")
        self.assertEqual(sum(1 for step in order if step.startswith("h_cast")), 1)
        self.assertEqual(plan["vector"]["h_cast_calls"], 1)
        self.assertIn("1/1024", plan["vector"]["scale_op"].replace("LOW_DOWNSCALE", "1/1024"))

    def test_v_l1_tile_is_loaded_once_for_both_products(self):
        batched = pv.plan_tiling(*FULL)["batched"]
        self.assertTrue(batched["b_l1_reused_by_both_products"])
        self.assertTrue(batched["per_head_distinct_hi_and_lo_a"])
        self.assertTrue(batched["per_head_distinct_b"])
        self.assertTrue(batched["head_offset_applied_to_all_gms"])
        self.assertEqual(batched["task_flatten"],
                         "head*(m_tiles*n_tiles) + m_tile*n_tiles + n_tile")


class AbiAndOffsetContracts(unittest.TestCase):
    def test_abi_is_five_pointers_hi_lo_v_aux_out(self):
        self.assertEqual(pv.KERNEL_ARG_ORDER,
                         ["hi_gm_float16", "lo_gm_float16", "v_gm_float16",
                          "aux_gm_float32", "out_gm_float16"])
        plan = pv.plan_tiling(*FULL)
        self.assertEqual(plan["kernel_arg_order"], pv.KERNEL_ARG_ORDER)
        self.assertEqual(plan["gm_shapes"]["aux"], [16])
        self.assertEqual(plan["gm_shapes"]["aux_dtype"], "float32")
        self.assertEqual(plan["gm_bytes"]["aux"], 64)
        self.assertEqual(plan["gm_shapes"]["hi"], [16, 256, 512, 16])
        self.assertEqual(plan["gm_shapes"]["lo"], [16, 256, 512, 16])
        self.assertEqual(plan["gm_shapes"]["b"], [16, 256, 128, 16])
        self.assertEqual(plan["gm_shapes"]["out"], [16, 8, 512, 16])

    def test_task_flatten_covers_every_tile_exactly_once(self):
        plan = pv.plan_tiling(*TAIL)
        gr = plan["grid"]
        seen = set()
        for task in range(gr["tasks"]):
            head, mt, nt = pv.flatten_task(task, 16, gr["n_tiles"], gr["per_head_tasks"])
            self.assertTrue(0 <= head < 16)
            self.assertTrue(0 <= mt < gr["m_tiles"])
            self.assertTrue(0 <= nt < gr["n_tiles"])
            seen.add((head, mt, nt))
        self.assertEqual(len(seen), gr["tasks"])
        with self.assertRaises(pv.FusedPvError):
            pv.flatten_task(gr["tasks"], 16, gr["n_tiles"], gr["per_head_tasks"])

    def test_gm_offsets_include_head_stride_and_k_block(self):
        plan = pv.plan_tiling(*FULL)
        hs, kb1, m_pad = plan["head_stride_elems"], 8, 512
        last = pv.plan_tiling(*FULL)["accumulation"]["k_tiles"] - 1
        off = pv.gm_offsets(plan, head=3, m_tile=2, n_tile=0, kt=last)
        self.assertEqual(off["hi"], 3 * hs["hi"] + (last * kb1 * m_pad + 256) * 16)
        self.assertEqual(off["lo"], 3 * hs["lo"] + (last * kb1 * m_pad + 256) * 16)
        self.assertEqual(off["b"], 3 * hs["b"] + (last * kb1 * 128 + 0) * 16)
        self.assertEqual(off["out"], 3 * hs["out"] + ((0) * m_pad + 256) * 16)
        self.assertEqual(off["out_elements"], 8 * 128 * 16)
        zero = pv.gm_offsets(plan, head=0, m_tile=0, n_tile=0, kt=0)
        self.assertEqual(zero["hi"], 0)
        self.assertEqual(zero["b"], 0)
        self.assertEqual(zero["out"], 0)
        for bad in ((16, 0, 0, 0), (0, 4, 0, 0), (0, 0, 1, 0), (0, 0, 0, 32)):
            with self.assertRaises(pv.FusedPvError):
                pv.gm_offsets(plan, *bad)

    def test_heads_never_alias_each_others_operands(self):
        plan = pv.plan_tiling(*FULL)
        hs = plan["head_stride_elems"]
        self.assertGreaterEqual(hs["hi"], plan["gm_elems"]["hi"] // 16)
        self.assertGreaterEqual(hs["b"], plan["gm_elems"]["b"] // 16)
        self.assertGreaterEqual(hs["out"], plan["gm_elems"]["out"] // 16)
        self.assertEqual(hs["hi"] * 16, plan["gm_elems"]["hi"])
        self.assertEqual(hs["out"] * 16, plan["gm_elems"]["out"])

    def test_kernel_names_encode_the_full_geometry_and_are_unique(self):
        full, tail = pv.plan_tiling(*FULL), pv.plan_tiling(*TAIL)
        full_name, tail_name = pv.kernel_name_for(full), pv.kernel_name_for(tail)
        self.assertNotEqual(full_name, tail_name)
        for token in ("h16", "m512", "k4096", "n128", "mb128", "nb128", "kb128", "c30"):
            self.assertIn(token, full_name)
        self.assertIn("m448", tail_name)
        self.assertEqual(len({full_name, tail_name}), 2)


class SourceContracts(unittest.TestCase):
    def test_pinned_round85_and_runtime_sources_still_match(self):
        r85_sha = hashlib.sha256((ROUND85_BATCH / "batch_gemm85.py").read_bytes()).hexdigest()
        self.assertEqual(r85_sha, pv.PINNED_R85_SHA256)
        runtime = HERE.parent.parent / "round83" / "runtime5gm83.py"
        self.assertEqual(hashlib.sha256(runtime.read_bytes()).hexdigest(),
                         pv.REFERENCE_RUNTIME_MODULE_SHA256)
        receipt = pv.check_reference_sources(r85_path=ROUND85_BATCH / "batch_gemm85.py",
                                             runtime_module_path=runtime, strict=True)
        self.assertEqual(receipt["proven_abi"], "fiveGMpointers")

    def test_reference_pins_fail_closed_on_drift(self):
        tampered = HERE / "_tmp_tampered85.py"
        try:
            tampered.write_text("# not the reviewed builder\n")
            with self.assertRaises(pv.FusedPvError):
                pv.check_reference_sources(r85_path=tampered, strict=True)
        finally:
            tampered.unlink()

    def test_module_is_stdlib_only_and_torch_is_lazy(self):
        source = PV_PATH.read_text()
        tree = __import__("ast").parse(source)
        top_imports = set()
        for node in tree.body:
            if isinstance(node, __import__("ast").Import):
                top_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, __import__("ast").ImportFrom) and node.module:
                top_imports.add(node.module.split(".")[0])
        self.assertEqual(top_imports, {"hashlib", "json", "os", "pathlib", "tempfile",
                                       "__future__"})
        # torch/TIK may only be imported lazily inside functions, never at module scope.
        self.assertEqual(sum(1 for line in source.splitlines()
                             if line.startswith("import torch")), 0)
        self.assertIn("SOURCE ONLY", source)
        self.assertIn("executed", source)

    def test_manifest_matches_delivered_files_and_makes_no_claim(self):
        manifest = json.loads(MANIFEST.read_text())
        for name, entry in manifest["files"].items():
            path = HERE / name
            self.assertTrue(path.is_file(), f"missing delivered file {name}")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             entry["sha256"], f"{name} drifted from its manifest SHA")
        self.assertFalse(manifest["executed"])
        self.assertFalse(manifest["compiled_by_author"])
        self.assertFalse(manifest["full_encoder_precision_accepted"])
        self.assertFalse(manifest["whole_goal_complete"])
        self.assertTrue(manifest["source_only"])
        self.assertIsNone(manifest["numerical_tolerance"])
        self.assertIsNone(manifest["measured_speedup"])
        self.assertEqual(manifest["kernel"]["GM_argument_count"], 5)
        self.assertEqual(manifest["kernel"]["dst_dtype"], "float16")


if __name__ == "__main__":
    unittest.main()