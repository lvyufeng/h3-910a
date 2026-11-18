#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round96 fused-PV vision attention engine.

No torch, torch_npu, TIK or CANN is imported. These tests pin the original helper
digest, the geometry and scale, the QK/fused-PV binary split, the launch-count
reduction, the five-GM argument order, the fail-closed activation check, the
disabled-path delegation and the scoped install/uninstall. They are structural
contracts, not numerical results and not speed claims.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "accurate_vision_attention_pv96.py"
PV_PATH = HERE / "pv_high_low96.py"
MANIFEST = HERE / "manifest96.json"
ROUND83 = HERE.parent.parent / "round83"
ROUND85_BATCH = HERE.parent.parent / "round85" / "batch"
CANDIDATE = HERE.parent.parent / "round79" / "candidate" / "candidate_encoder79.py"
RUNTIME = ROUND83 / "runtime5gm83.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(ROUND85_BATCH) not in sys.path:
    sys.path.insert(0, str(ROUND85_BATCH))
engine = _load("accurate_vision_attention_pv96", ENGINE_PATH)
pv = _load("pv_high_low96", HERE / "pv_high_low96.py")


class FakeCandidate:
    """Minimal stand-in for the private candidate module (no torch)."""

    __file__ = str(CANDIDATE)

    def __init__(self):
        self.calls = []
        self._vision_sdpa_bf16_compatible = self._original_helper

    def _original_helper(self, q, k, v):
        self.calls.append((q, k, v))
        return ("original", q, k, v)


class FakeRuntime:
    """Runtime stand-in whose module is registered as the pinned bridge file."""

    def __init__(self, device=0):
        self.device = device

    def snapshot(self):
        return {'library_sha256': engine.PINNED_RUNTIME_LIBRARY_SHA256}


def _bind_runtime_module(name="fake_runtime96", path=RUNTIME):
    """Register a module whose __file__ is the real pinned runtime bridge."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    FakeRuntime.__module__ = name
    return module


class GeometryContracts(unittest.TestCase):
    def test_QK_metadata_uses_actual_R85_logical_shape_field(self):
        source = ENGINE_PATH.read_text()
        self.assertIn('meta.get("logical_shape")', source)
        self.assertNotIn('logical_geometry', source)
        self.assertIn('"logical_shape": plan["logical"]', (ROUND85_BATCH / 'batch_gemm85.py').read_text())
        self.assertIs(engine.AccurateVisionAttentionPV96, engine.AccurateVisionAttentionPv96)

    def test_runtime_launcher_pin_is_verified_not_just_reported(self):
        self.assertIn("self._runtime.snapshot().get('library_sha256') != PINNED_RUNTIME_LIBRARY_SHA256", ENGINE_PATH.read_text())

    def test_eight_chunks_seven_full_plus_one_448_tail(self):
        chunks = engine.query_chunks()
        self.assertEqual(len(chunks), 8)
        self.assertEqual([m for _s, _e, m in chunks], [512] * 7 + [448])
        self.assertEqual(chunks[0], (0, 512, 512))
        self.assertEqual(chunks[-1], (3584, 4032, 448))
        self.assertEqual(sum(m for _s, _e, m in chunks), 4032)

    def test_scale_is_the_original_head_dim_not_the_padded_width(self):
        self.assertAlmostEqual(engine.ORIGINAL_SCALE, 72 ** -0.5, places=15)
        self.assertNotAlmostEqual(engine.ORIGINAL_SCALE, 128 ** -0.5, places=6)
        padding = engine.AccurateVisionAttentionPv96.padding_selfcheck(None)
        self.assertFalse(padding["padded_width_scale_used"])
        self.assertFalse(padding["softmax_masked"])
        self.assertTrue(padding["score_cropped_before_softmax"])

    def test_padding_and_residual_scale(self):
        self.assertEqual((engine.ORIGINAL_HEAD_DIM, engine.PADDED_HEAD_DIM), (72, 128))
        self.assertEqual((engine.ORIGINAL_KEY_LEN, engine.PADDED_KEY_LEN), (4032, 4096))
        self.assertEqual(engine.RESIDUAL_SCALE, 1024.0)
        self.assertEqual(engine.NUM_HEADS, 16)

    def test_four_binaries_two_qk_plus_two_fused_pv(self):
        qk, fused = engine.planned_qk_shapes(), engine.planned_pv_shapes()
        self.assertEqual(qk, ((16, 512, 128, 4096), (16, 448, 128, 4096)))
        self.assertEqual(fused, ((16, 512, 4096, 128), (16, 448, 4096, 128)))
        for shape in fused:
            plan = pv.plan_tiling(*shape)
            pv._check_budget(plan)
            self.assertEqual(plan["padded"]["m_pad"], 512)

    def test_launch_count_drops_from_three_to_two_per_chunk(self):
        counts = engine.planned_launch_counts()
        self.assertEqual(counts["query_chunks"], 8)
        self.assertEqual(counts["kernel_launches"], 16)
        self.assertEqual(counts["round88_launches_without_fusion"], 24)
        self.assertEqual(counts["fused_pv_launches"], 8)
        self.assertEqual(counts["qk_launches"], 8)
        self.assertEqual(counts["fused_pv_launches"] + counts["qk_launches"],
                         counts["kernel_launches"])

    def test_bytes_accounting_shows_one_half_product_buffer(self):
        b = engine.padded_buffer_bytes()
        m_pad = engine.QUERY_CHUNK
        self.assertEqual(b["fused_pv_half_gm"], 16 * 8 * m_pad * 16 * 2)
        self.assertEqual(b["score_fp32_gm"], 16 * 256 * m_pad * 16 * 4)
        self.assertLess(b["per_chunk_transient_total"],
                        b["round88_per_chunk_transient_total"])
        self.assertEqual(b["round88_per_chunk_transient_total"]
                         - b["per_chunk_transient_total"],
                         16 * 8 * m_pad * 16 * 4 - 16 * 8 * m_pad * 16 * 2)
        self.assertIn("not a measured speedup", b["note"])


class ActivationContract(unittest.TestCase):
    def test_real_geometry_half_npu_is_accepted(self):
        spec = engine.check_activation_spec((1, 16, 4032, 72), "torch.float16", "npu", 0, 0)
        self.assertEqual(spec["shape"], [1, 16, 4032, 72])
        self.assertEqual(spec["runtime_device"], 0)

    def test_wrong_dtype_shape_or_device_is_refused(self):
        cases = [
            ((1, 16, 4032, 72), "torch.float32", "npu", 0, 0),
            ((1, 16, 4032, 72), "torch.bfloat16", "npu", 0, 0),
            ((1, 16, 4032, 128), "torch.float16", "npu", 0, 0),
            ((1, 8, 4032, 72), "torch.float16", "npu", 0, 0),
            ((1, 16, 4032, 72), "torch.float16", "cpu", 0, 0),
            ((1, 16, 4032, 72), "torch.float16", "npu", 1, 0),
            ((1, 16, 4032, 72), "torch.float16", "npu", 0, -1),
        ]
        for shape, dtype, device_type, index, runtime_device in cases:
            with self.assertRaises(engine.VisionAttentionPvError):
                engine.check_activation_spec(shape, dtype, device_type, index,
                                             runtime_device)


class InstallAndGateContracts(unittest.TestCase):
    def setUp(self):
        _bind_runtime_module()
        self.candidate = FakeCandidate()
        self.runtime = FakeRuntime()
        self.original = self.candidate._vision_sdpa_bf16_compatible
        self.engine = engine.install(self.candidate, self.runtime, HERE / "_kernels96_test")

    def tearDown(self):
        try:
            self.engine.uninstall()
        except engine.VisionAttentionPvError:
            pass

    def test_install_patches_only_the_helper_and_stays_disabled(self):
        self.assertFalse(self.engine.snapshot()["enabled"])
        self.assertEqual(self.candidate._vision_sdpa_bf16_compatible,
                         self.engine.forward)
        self.assertIs(self.candidate._vision_sdpa_bf16_compatible.__self__,
                      self.engine)
        snap = self.engine.snapshot()
        self.assertTrue(snap["per_instance_patch_only"])
        self.assertFalse(snap["patches_torch_global"])
        self.assertFalse(snap["patches_torch_functional"])
        self.assertFalse(snap["patches_candidate_class"])
        self.assertEqual(snap["patched_helper"], "_vision_sdpa_bf16_compatible")

    def test_disabled_path_returns_the_captured_original_exactly(self):
        out = self.candidate._vision_sdpa_bf16_compatible("q", "k", "v")
        self.assertEqual(out, ("original", "q", "k", "v"))
        self.assertEqual(len(self.candidate.calls), 1)
        snap = self.engine.snapshot()
        self.assertEqual(snap["stats"]["calls_disabled"], 1)
        self.assertEqual(snap["stats"]["calls_enabled"], 0)
        self.assertTrue(snap["original_helper_preserved"])

    def test_enable_before_prepare_is_refused(self):
        with self.assertRaises(engine.VisionAttentionPvError):
            self.engine.set_enabled(True)
        with self.assertRaises(engine.VisionAttentionPvError):
            self.engine.set_enabled("yes")
        self.assertFalse(self.engine.snapshot()["enabled"])

    def test_uninstall_restores_the_original_and_is_idempotent(self):
        summary = self.engine.uninstall()
        self.assertTrue(summary["uninstalled"])
        restored = self.candidate._vision_sdpa_bf16_compatible
        self.assertEqual(restored, self.original)
        self.assertEqual(restored.__func__, self.original.__func__)
        self.engine.uninstall()

    def test_uninstall_refuses_to_clobber_a_third_party_patch(self):
        self.candidate._vision_sdpa_bf16_compatible = lambda q, k, v: "other"
        with self.assertRaises(engine.VisionAttentionPvError):
            self.engine.uninstall()
        self.assertEqual(self.candidate._vision_sdpa_bf16_compatible(None, None, None),
                         "other")


class PinnedSourceContracts(unittest.TestCase):
    def setUp(self):
        _bind_runtime_module()

    def test_runtime_and_kernel_and_candidate_pins_still_match(self):
        self.assertEqual(hashlib.sha256(RUNTIME.read_bytes()).hexdigest(),
                         engine.PINNED_RUNTIME_MODULE_SHA256)
        self.assertEqual(
            hashlib.sha256((ROUND85_BATCH / "batch_gemm85.py").read_bytes()).hexdigest(),
            engine.PINNED_KERNEL_SHA256)
        self.assertEqual(hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                         engine.PINNED_CANDIDATE_SHA256)
        self.assertEqual(engine.helper_source_digest(CANDIDATE),
                         engine.PINNED_HELPER_SOURCE_SHA256)

    def test_verify_sources_accepts_the_pinned_inputs_and_records_receipts(self):
        eng = engine.AccurateVisionAttentionPv96(FakeCandidate(), FakeRuntime(),
                                                 HERE / "_kernels96_test")
        receipt = eng._verify_sources()
        self.assertEqual(receipt["kernel_module"]["sha256"], engine.PINNED_KERNEL_SHA256)
        self.assertEqual(receipt["runtime_module"]["sha256"],
                         engine.PINNED_RUNTIME_MODULE_SHA256)
        self.assertEqual(receipt["runtime_module"]["ABI"], "fiveGMpointers")
        self.assertEqual(receipt["candidate"]["sha256"], engine.PINNED_CANDIDATE_SHA256)
        self.assertTrue(receipt["candidate"]["matches_pin"])
        eng.uninstall()

    def test_verify_sources_refuses_an_unpinned_runtime_module(self):
        class OtherRuntime:
            device = 0

        OtherRuntime.__module__ = __name__
        eng = engine.AccurateVisionAttentionPv96(FakeCandidate(), OtherRuntime(),
                                                 HERE / "_kernels96_test")
        with self.assertRaises(engine.VisionAttentionPvError):
            eng._verify_sources()
        eng.uninstall()

    def test_pv_module_receipt_pins_the_five_gm_order(self):
        eng = engine.AccurateVisionAttentionPv96(FakeCandidate(), FakeRuntime(),
                                                 HERE / "_kernels96_test")
        module, receipt = eng._verify_pv_module()
        self.assertEqual(Path(module.__file__).resolve(), PV_PATH.resolve())
        self.assertEqual(receipt["kernel_arg_order"], engine.EXPECTED_PV_ARG_ORDER)
        self.assertEqual(receipt["GM_argument_count"], 5)
        self.assertEqual(receipt["sha256"],
                         hashlib.sha256((HERE / "pv_high_low96.py").read_bytes()).hexdigest())
        eng.uninstall()

    def test_fused_pv_meta_gate_accepts_the_real_contract_and_refuses_tampering(self):
        plan = pv.plan_tiling(16, 512, 4096, 128)
        meta = {
            "logical_shape": {"batch": 16, "m": 512, "k": 4096, "n": 128},
            "dst_dtype": "float16", "input_dtype": "float16",
            "accumulator_dtype": "float32", "GM_argument_count": 5,
            "single_final_half_cast": True, "no_mid_product_fp32_gm": True,
            "no_host_combine": True, "fused_products": ["high", "low"],
            "kernel_arg_order": list(pv.KERNEL_ARG_ORDER),
            "one_kernel_launch_per_tile": True, "no_half_partial": True,
            "block_dim": plan["grid"]["cores"], "grid": plan["grid"],
        }
        engine.AccurateVisionAttentionPv96._check_pv_meta(meta, 16, 512, 4096, 128)
        for key, bad in (("dst_dtype", "float32"), ("GM_argument_count", 4),
                         ("single_final_half_cast", False),
                         ("no_mid_product_fp32_gm", False),
                         ("no_host_combine", False),
                         ("fused_products", ["high"]),
                         ("no_half_partial", False),
                         ("one_kernel_launch_per_tile", False),
                         ("block_dim", 31),
                         ("kernel_arg_order", list(reversed(pv.KERNEL_ARG_ORDER)))):
            tampered = dict(meta)
            tampered[key] = bad
            with self.assertRaises(engine.VisionAttentionPvError):
                engine.AccurateVisionAttentionPv96._check_pv_meta(
                    tampered, 16, 512, 4096, 128)

    def test_engine_imports_no_torch_and_claims_nothing(self):
        source = ENGINE_PATH.read_text()
        tree = __import__("ast").parse(source)
        top = set()
        for node in tree.body:
            if isinstance(node, __import__("ast").Import):
                top.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, __import__("ast").ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        self.assertEqual(top, {"ast", "hashlib", "pathlib", "textwrap", "__future__"})
        self.assertIn("SOURCE ONLY", source)
        snap = engine.AccurateVisionAttentionPv96(
            FakeCandidate(), FakeRuntime(), HERE / "_kernels96_test").snapshot()
        self.assertFalse(snap["full_encoder_precision_accepted"])
        self.assertFalse(snap["whole_goal_complete"])
        self.assertIsNone(snap["numerical_tolerance"])
        self.assertTrue(snap["not_a_speed_claim"])
        self.assertTrue(snap["float32_combine_in_vector_unit"])
        self.assertTrue(snap["host_combine_removed"])
        self.assertTrue(snap["no_fp32_weight_shadow"])
        self.assertTrue(snap["no_bf16_teacher"])

    def test_manifest_matches_delivered_files(self):
        manifest = json.loads(MANIFEST.read_text())
        for name, entry in manifest["files"].items():
            path = HERE / name
            self.assertTrue(path.is_file(), f"missing delivered file {name}")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             entry["sha256"], f"{name} drifted from its manifest SHA")
        self.assertFalse(manifest["executed"])
        self.assertFalse(manifest["full_encoder_precision_accepted"])
        self.assertFalse(manifest["whole_goal_complete"])
        self.assertTrue(manifest["source_only"])


if __name__ == "__main__":
    unittest.main()