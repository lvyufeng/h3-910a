#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round88 accurate vision SDPA engine.

These tests import no torch, torch_npu, TIK or CANN: they check the padding
geometry, the scale choice, the four planned binaries against the immutable
Round85 tiling plan, the source pins, the gating/uninstall behaviour and the
deterministic launch/byte accounting. They are structural contracts, not
numerical results and not performance claims.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "accurate_vision_attention88.py"
MANIFEST = HERE / "manifest88.json"
ROUND85_BATCH = HERE.parent.parent / "round85" / "batch"
CANDIDATE = HERE.parent.parent / "round79" / "candidate" / "candidate_encoder79.py"
ROUND83 = HERE.parent.parent / "round83"


def _load_engine():
    spec = importlib.util.spec_from_file_location("accurate_vision_attention88", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_batch85():
    if str(ROUND85_BATCH) not in sys.path:
        sys.path.insert(0, str(ROUND85_BATCH))
    import batch_gemm85

    return batch_gemm85


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


engine = _load_engine()


class FakeCandidate:
    """Minimal stand-in for the R79 private candidate module (no torch)."""

    __file__ = str(CANDIDATE)

    def __init__(self):
        self.calls = []
        self._vision_sdpa_bf16_compatible = self._original_helper

    def _original_helper(self, q, k, v):
        self.calls.append((q, k, v))
        return ("original", q, k, v)


class FakeRuntime:
    device = 0


class ModuleScopeTests(unittest.TestCase):
    def test_module_scope_imports_are_stdlib_only(self):
        tree = ast.parse(ENGINE_PATH.read_text())
        allowed = {"__future__", "ast", "hashlib", "pathlib", "textwrap"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertTrue(alias.name.split(".")[0] in allowed,
                                    f"non-stdlib module-scope import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                self.assertTrue(node.module.split(".")[0] in allowed,
                                f"non-stdlib module-scope import {node.module}")

    def test_torch_and_tik_are_lazy(self):
        tree = ast.parse(ENGINE_PATH.read_text())
        module_level = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_level.add(ast.dump(node))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""])
                lazy = any(n.split(".")[0] in {"torch", "torch_npu", "tbe", "batch_gemm85",
                                               "runtime5gm83"} for n in names)
                if lazy:
                    self.assertNotIn(ast.dump(node), module_level)


class GeometryTests(unittest.TestCase):
    def test_real_vision_geometry_constants(self):
        self.assertEqual(engine.NUM_HEADS, 16)
        self.assertEqual(engine.ORIGINAL_HEAD_DIM, 72)
        self.assertEqual(engine.PADDED_HEAD_DIM, 128)
        self.assertEqual(engine.ORIGINAL_KEY_LEN, 4032)
        self.assertEqual(engine.PADDED_KEY_LEN, 4096)
        self.assertEqual(engine.QUERY_CHUNK, 512)
        self.assertEqual(engine.EXPECTED_ACTIVATION_SHAPE, (1, 16, 4032, 72))

    def test_query_chunks_are_seven_512_plus_one_448(self):
        chunks = engine.query_chunks()
        self.assertEqual(len(chunks), 8)
        lengths = [m for _s, _e, m in chunks]
        self.assertEqual(lengths, [512] * 7 + [448])
        self.assertEqual(sum(lengths), engine.ORIGINAL_KEY_LEN)
        for index, (start, end, m) in enumerate(chunks):
            self.assertEqual(start, index * 512)
            self.assertEqual(end - start, m)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], engine.ORIGINAL_KEY_LEN)

    def test_original_scale_is_72_not_the_padded_128(self):
        self.assertAlmostEqual(engine.ORIGINAL_SCALE, 72 ** -0.5, places=15)
        self.assertAlmostEqual(engine.ORIGINAL_SCALE, 1.0 / (6.0 * 2.0 ** 0.5), places=15)
        self.assertAlmostEqual(engine.ORIGINAL_SCALE, 0.11785113019775792, places=15)
        self.assertGreater(engine.ORIGINAL_SCALE, 128 ** -0.5)
        # 72**-0.5 / 128**-0.5 == sqrt(128/72); using the padded width would
        # rescale every genuine score by that factor.
        self.assertAlmostEqual(engine.ORIGINAL_SCALE / (128 ** -0.5),
                               (128 / 72) ** 0.5, places=12)

    def test_four_planned_binaries_are_legal_under_the_round85_plan(self):
        kernel = _load_batch85()
        shapes = engine.planned_kernel_shapes()
        self.assertEqual(shapes, ((16, 512, 128, 4096), (16, 448, 128, 4096),
                                  (16, 512, 4096, 128), (16, 448, 4096, 128)))
        for batch, m, k, n in shapes:
            plan = kernel.plan_tiling(batch, m, k, n)
            self.assertEqual(plan["logical"]["batch"], 16)
            self.assertEqual(plan["padded"]["m_pad"], 512)
            self.assertEqual(plan["grid"]["cores"], 30)
            self.assertEqual(plan["padded"]["k1"], k // 16)
            self.assertEqual(plan["padded"]["n1"], n // 16)
            self.assertGreater(plan["grid"]["tasks"], 30)
            if k == engine.PADDED_HEAD_DIM:
                self.assertEqual(plan["accumulation"]["k_tiles"], 1)
                self.assertEqual(plan["logical"]["n"], 4096)
            else:
                self.assertEqual(plan["accumulation"]["k_tiles"], 32)
                self.assertEqual(plan["logical"]["n"], 128)
            buffers = plan["buffers"]
            self.assertLessEqual(buffers["l1_a_bytes"] + buffers["l1_b_bytes"],
                                 kernel.PINNED_L1_SIZE)
            self.assertLessEqual(buffers["l0c_bytes"], kernel.PINNED_L0C_SIZE)
            self.assertLessEqual(buffers["ub_bytes"], 196608)
            self.assertFalse(plan["accumulation"]["half_partial"])
            self.assertFalse(plan["accumulation"]["fp32_ub_partial_sum"])
            self.assertTrue(plan["accumulation"]["single_l0c_across_all_k_blocks"])
            self.assertTrue(plan["batched"]["per_head_distinct_b"])

    def test_planned_geometries_roles(self):
        roles = {g["role"] for g in engine.planned_geometries()}
        self.assertEqual(roles, {"score", "pv"})
        self.assertEqual(len(engine.planned_geometries()), 4)
        for geometry in engine.planned_geometries():
            self.assertEqual(geometry["m_pad"], 512)


class KernelMetaContractTests(unittest.TestCase):
    """Rehearse the validator against the real Round85 ``_describe_cached`` schema.

    No TIK and no CANN are involved: ``plan_tiling`` is the same pure-python plan
    ``build_kernel`` uses, and the dict below is assembled from exactly the keys
    ``_describe_cached`` emits. A fake ``.o`` stands in for the built object so the
    hash check can be exercised too.
    """

    def _meta(self, tmpdir, batch, m, k, n):
        kernel = _load_batch85()
        plan = kernel.plan_tiling(batch, m, k, n)
        obj = Path(tmpdir) / "fake.o"
        obj.write_bytes(b"round88 metadata rehearsal, not a device binary")
        import hashlib as _h

        return {
            "kernel_name": kernel.kernel_name_for(plan),
            "kernel_symbol": "fake_symbol",
            "block_dim": int(plan["grid"]["cores"]),
            "obj_path": str(obj),
            "obj_sha256": _h.sha256(obj.read_bytes()).hexdigest(),
            "dst_dtype": "float32",
            "input_dtype": "float16",
            "accumulator_dtype": "float32",
            "GM_argument_count": 5,
            "logical_shape": dict(plan["logical"]),
            "padded_shape": {"m_pad": plan["padded"]["m_pad"],
                             "k1": plan["padded"]["k1"], "n1": plan["padded"]["n1"]},
            "gm_shapes": plan["gm_shapes"],
            "grid": dict(plan["grid"]),
            "accumulation": dict(plan["accumulation"]),
        }

    def test_validator_accepts_every_planned_geometry(self):
        import tempfile

        instance = engine.AccurateVisionAttention88(FakeCandidate(), FakeRuntime(),
                                                   HERE / "kernels")
        with tempfile.TemporaryDirectory() as tmpdir:
            for batch, m, k, n in engine.planned_kernel_shapes():
                meta = self._meta(tmpdir, batch, m, k, n)
                self.assertIs(instance._validate_kernel_meta(meta, batch, m, k, n), meta)
        instance.uninstall()

    def test_validator_fails_closed_on_every_drift(self):
        import tempfile

        instance = engine.AccurateVisionAttention88(FakeCandidate(), FakeRuntime(),
                                                   HERE / "kernels")
        batch, m, k, n = engine.planned_kernel_shapes()[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self._meta(tmpdir, batch, m, k, n)
            drift = {
                "wrong output dtype": lambda meta: meta.update(dst_dtype="float16"),
                "wrong input dtype": lambda meta: meta.update(input_dtype="float32"),
                "half partial": lambda meta: meta["accumulation"].update(half_partial=True),
                "ub partial": lambda meta: meta["accumulation"].update(
                    fp32_ub_partial_sum=True),
                "wrong core count": lambda meta: meta.update(block_dim=64),
                "wrong gm argu32_count": lambda meta: meta.update(GM_argument_count=4),
                "wrong logical n": lambda meta: meta["logical_shape"].update(n=2048),
                "wrong m pad": lambda meta: meta["padded_shape"].update(m_pad=256),
                "wrong c shape": lambda meta: meta["gm_shapes"].update(
                    c=[batch, n // 16, 256, 16]),
                "tampered object": lambda meta: meta.update(obj_sha256="0" * 64),
            }
            for label, mutate in drift.items():
                meta = json.loads(json.dumps(baseline))
                mutate(meta)
                with self.assertRaises(engine.VisionAttentionError, msg=label):
                    instance._validate_kernel_meta(meta, batch, m, k, n)
        instance.uninstall()


class ActivationSpecTests(unittest.TestCase):
    def good(self, **overrides):
        values = dict(shape=(1, 16, 4032, 72), dtype_name="torch.float16",
                      device_type="npu", device_index=0, runtime_device=0)
        values.update(overrides)
        return engine.check_activation_spec(**values)

    def test_accepts_the_real_operands(self):
        spec = self.good()
        self.assertEqual(spec["shape"], [1, 16, 4032, 72])
        self.assertEqual(spec["runtime_device"], 0)

    def test_rejects_bf16_and_float32(self):
        for dtype in ("torch.bfloat16", "torch.float32"):
            with self.assertRaises(engine.VisionAttentionError):
                self.good(dtype_name=dtype)

    def test_rejects_other_sequence_or_head_dim(self):
        with self.assertRaises(engine.VisionAttentionError):
            self.good(shape=(1, 16, 4032, 128))
        with self.assertRaises(engine.VisionAttentionError):
            self.good(shape=(1, 16, 4096, 72))
        with self.assertRaises(engine.VisionAttentionError):
            self.good(shape=(2, 16, 4032, 72))

    def test_rejects_cpu_and_wrong_device(self):
        with self.assertRaises(engine.VisionAttentionError):
            self.good(device_type="cpu")
        with self.assertRaises(engine.VisionAttentionError):
            self.good(device_index=3)


class SourcePinTests(unittest.TestCase):
    def test_kernel_and_runtime_pins_match_the_committed_sources(self):
        self.assertEqual(_sha256(ROUND85_BATCH / "batch_gemm85.py"),
                         engine.PINNED_KERNEL_SHA256)
        self.assertEqual(_sha256(ROUND83 / "runtime5gm83.py"),
                         engine.PINNED_RUNTIME_MODULE_SHA256)
        self.assertEqual(engine.PINNED_RUNTIME_MODULE,
                         (ROUND83 / "runtime5gm83.py").stem)

    def test_helper_pin_matches_the_committed_candidate(self):
        self.assertEqual(_sha256(CANDIDATE), engine.PINNED_CANDIDATE_SHA256)
        self.assertEqual(engine.helper_source_digest(CANDIDATE),
                         engine.PINNED_HELPER_SOURCE_SHA256)

    def test_helper_digest_fails_closed_on_a_missing_helper(self):
        with self.assertRaises(engine.VisionAttentionError):
            engine.helper_source_digest(ENGINE_PATH)


class GatingTests(unittest.TestCase):
    def test_install_patches_only_the_vision_helper(self):
        candidate = FakeCandidate()
        original = candidate._vision_sdpa_bf16_compatible
        instance = engine.AccurateVisionAttention88(candidate, FakeRuntime(), HERE / "kernels")
        self.assertIsNot(candidate._vision_sdpa_bf16_compatible, original)
        self.assertFalse(instance.snapshot()["enabled"])
        self.assertTrue(instance.snapshot()["installed"])
        # Disabled means the captured original operator, byte for byte.
        sentinel = ("q", "k", "v")
        self.assertEqual(candidate._vision_sdpa_bf16_compatible(*sentinel),
                         ("original", *sentinel))
        instance.set_enabled(False)
        self.assertEqual(instance.snapshot()["stats"]["calls_disabled"], 1)
        summary = instance.uninstall()
        self.assertTrue(summary["uninstalled"])
        self.assertFalse(summary["installed"])
        self.assertIs(candidate._vision_sdpa_bf16_compatible, original)

    def test_enabling_before_prepare_fails_closed(self):
        instance = engine.AccurateVisionAttention88(FakeCandidate(), FakeRuntime(),
                                                    HERE / "kernels")
        with self.assertRaises(engine.VisionAttentionError):
            instance.set_enabled(True)
        instance.uninstall()

    def test_uninstall_is_idempotent(self):
        instance = engine.AccurateVisionAttention88(FakeCandidate(), FakeRuntime(),
                                                    HERE / "kernels")
        instance.uninstall()
        second = instance.uninstall()
        self.assertTrue(second["uninstalled"])

    def test_uninstall_refuses_a_third_party_replacement(self):
        candidate = FakeCandidate()
        instance = engine.AccurateVisionAttention88(candidate, FakeRuntime(), HERE / "kernels")
        candidate._vision_sdpa_bf16_compatible = lambda q, k, v: None
        with self.assertRaises(engine.VisionAttentionError):
            instance.uninstall()

    def test_install_requires_the_named_helper(self):
        class Bare:
            __file__ = str(CANDIDATE)

        with self.assertRaises(engine.VisionAttentionError):
            engine.AccurateVisionAttention88(Bare(), FakeRuntime(), HERE / "kernels")

    def test_install_factory_returns_a_disabled_engine(self):
        candidate = FakeCandidate()
        # ``candidate._original_helper`` is re-bound on every access, so capture
        # the stored object once instead of comparing two fresh bound methods.
        original = candidate._vision_sdpa_bf16_compatible
        instance = engine.install(candidate, FakeRuntime(), HERE / "kernels")
        self.assertIsInstance(instance, engine.AccurateVisionAttention88)
        self.assertFalse(instance.snapshot()["enabled"])
        self.assertEqual((candidate._vision_sdpa_bf16_compatible.__self__,
                          candidate._vision_sdpa_bf16_compatible.__func__),
                         (instance, type(instance).forward))
        instance.uninstall()
        self.assertIs(candidate._vision_sdpa_bf16_compatible, original)


class AccountingTests(unittest.TestCase):
    def test_launch_counts_are_the_structural_contract(self):
        counts = engine.planned_launch_counts()
        self.assertEqual(counts["query_chunks"], 8)
        self.assertEqual(counts["pack_w_calls"], 2)
        self.assertEqual(counts["kernel_launches"], 24)
        self.assertEqual(counts["float_before_half_chunks"], 8)

    def test_extra_buffer_accounting_is_deterministic(self):
        first, second = engine.padded_buffer_bytes(), engine.padded_buffer_bytes()
        self.assertEqual(first, second)
        persistent = (first["padded_q_half"] + first["padded_k_half"]
                      + first["padded_v_half"] + first["transposed_v_half"]
                      + first["packed_k_half"] + first["packed_v_half"])
        self.assertEqual(first["call_persistent_total"], persistent)
        self.assertEqual(first["padded_q_half"], 16 * 4032 * 128 * 2)
        self.assertEqual(first["padded_k_half"], 16 * 4096 * 128 * 2)
        self.assertEqual(first["packed_k_half"], 16 * 8 * 4096 * 16 * 2)
        self.assertEqual(first["packed_v_half"], 16 * 256 * 128 * 16 * 2)
        self.assertGreater(first["per_chunk_transient_total"], 0)


class NonClaimAndScopeTests(unittest.TestCase):
    def test_no_causal_mask_or_negative_infinity(self):
        text = ENGINE_PATH.read_text()
        for token in ("masked_fill", "is_causal=True", "float('-inf')", 'float("-inf")'):
            self.assertFalse(token in text, f"engine contains forbidden token {token!r}")
        self.assertTrue("torch.softmax(scores, dim=-1)" in text, "softmax call missing")

    def test_no_bias_text_sdpa_or_global_functional_patch(self):
        text = ENGINE_PATH.read_text()
        # Never referenced at all: the bias classes, the text SDPA symbol, BF16.
        for token in ("_NpuBF16CompatibleLinear", "_scaled_dot_product_attention",
                      "bfloat16", "bias_adapter"):
            self.assertFalse(token in text, f"engine must not reference {token!r}")
        # The docstrings *deny* touching torch.nn.functional; what must not exist
        # is any actual use or import of it.
        for token in ("torch.nn.functional.", "nn.functional.", "import torch.nn",
                      "from torch.nn", "F."):
            self.assertFalse(token in text, f"engine must not use {token!r}")
        # The only graph attribute this engine ever writes is the vision helper:
        # install and uninstall both target HELPER_NAME and nothing else.
        calls = []
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"):
                calls.append(node)
        self.assertEqual(len(calls), 2, "expected exactly one install and one uninstall write")
        for call in calls:
            self.assertEqual(len(call.args), 3)
            self.assertIsInstance(call.args[1], ast.Name)
            self.assertEqual(call.args[1].id, "HELPER_NAME")
        self.assertTrue("_vision_sdpa_bf16_compatible" == engine.HELPER_NAME)

    def test_engine_claims_no_result(self):
        text = ENGINE_PATH.read_text()
        self.assertTrue("full_encoder_precision_accepted" in text, "missing non-claim flag")
        self.assertTrue('"whole_goal_complete": False' in text, "missing goal non-claim")
        for token in ("speedup achieved", "lossless achieved", "accepted: true"):
            self.assertFalse(token in text.lower(), f"engine claims {token!r}")

    def test_padded_keys_are_dropped_before_softmax(self):
        text = ENGINE_PATH.read_text()
        score_line = text.index("scores = scores[:, :, :ORIGINAL_KEY_LEN] * ORIGINAL_SCALE")
        softmax_line = text.index("p = torch.softmax(scores, dim=-1)")
        self.assertLess(score_line, softmax_line)

    def test_one_half_cast_after_both_float_pv_products(self):
        text = ENGINE_PATH.read_text()
        index_pv = text.index("out = (self._gemm(p_high, packed_v, PADDED_HEAD_DIM)")
        index_cast = text.index("result[:, start:end, :] = out.to(q.dtype)")
        self.assertLess(index_pv, index_cast)
        self.assertFalse("p.to(torch.float32)" in text, "P must stay Float, not be widened later")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text())

    def test_manifest_hashes_match_the_sources(self):
        for name, entry in self.manifest["files"].items():
            path = HERE / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(_sha256(path), entry["sha256"], name)
            self.assertEqual(len(path.read_bytes()), entry["bytes"], name)

    def test_manifest_is_non_claiming(self):
        for flag in ("compiled", "executed_on_npu", "no_ssh_or_remote_command",
                     "proves_precision", "proves_speed"):
            self.assertIn(flag, self.manifest)
        self.assertFalse(self.manifest["compiled"])
        self.assertFalse(self.manifest["executed_on_npu"])
        self.assertTrue(self.manifest["no_ssh_or_remote_command"])
        self.assertFalse(self.manifest["proves_precision"])
        self.assertFalse(self.manifest["proves_speed"])

    def test_manifest_records_the_requested_api(self):
        self.assertEqual(self.manifest["module"], "accurate_vision_attention88.py")
        self.assertEqual(self.manifest["class"], "AccurateVisionAttention88")
        self.assertEqual(self.manifest["api"],
                         ["prepare", "set_enabled", "snapshot", "uninstall"])
        self.assertEqual(self.manifest["planned_geometries"],
                         engine.planned_geometries())


if __name__ == "__main__":
    unittest.main()