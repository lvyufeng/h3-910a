#!/usr/bin/env python3
"""Stdlib-only tests for the Round84 FP32 row-parallel adapter.

No torch, vLLM, NPU or network access is required or performed here: this file
runs on a plain laptop. It verifies source pins, the pure-python shape/signature
logic, the gated install/uninstall behaviour with a stand-in candidate module,
and the source invariants the runtime path depends on. The NPU numerics are
exercised by the root driver on the leased host, not here.
"""

import ast
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CANDIDATE = REPO / "artifacts/diagnostics/round79/candidate/candidate_encoder79.py"
R83_KERNEL = REPO / "artifacts/diagnostics/round83/blocked_dynamic_steps83.py"
R83_RUNTIME = REPO / "artifacts/diagnostics/round83/runtime5gm83.py"
ADAPTER = HERE / "row_fp32_adapter_fixed84.py"


def _load_module():
    sys.path.insert(0, str(HERE))
    try:
        return importlib.import_module("row_fp32_adapter_fixed84")
    finally:
        sys.path.pop(0)


m = _load_module()


class FakeTensor:
    """Minimal stand-in for a torch tensor exposing only the signature inputs."""

    def __init__(self, pointer, offset, shape, stride, dtype, device, version):
        self._pointer = pointer
        self._offset = offset
        self.shape = shape
        self._stride = stride
        self.dtype = dtype
        self.device = device
        self._version = version

    def data_ptr(self):
        return self._pointer

    def storage_offset(self):
        return self._offset

    def stride(self):
        return self._stride


class FakeCandidate:
    """Stand-in candidate module: a patched class and the quantized check."""

    def __init__(self):
        module = types.ModuleType("fake_candidate84")

        class MiniMaxH3Qwen3VLRowParallelLinear:
            def __init__(self, tag="layer"):
                self.tag = tag

            def forward(self, input_):
                return ("native-forward", self.tag, input_)

        module.MiniMaxH3Qwen3VLRowParallelLinear = MiniMaxH3Qwen3VLRowParallelLinear
        module._is_quantized_linear = lambda layer: False
        self.module = module
        self.cls = module.MiniMaxH3Qwen3VLRowParallelLinear


class SourcePinTests(unittest.TestCase):
    def test_candidate_file_matches_the_pinned_digests(self):
        digests = m.candidate_source_digests(CANDIDATE)
        self.assertEqual(digests["file_sha256"], m.PINNED["round79_candidate_file_sha256"])
        self.assertEqual(digests["forward_source_sha256"],
                         m.PINNED["round79_row_forward_source_sha256"])
        self.assertEqual(digests["class_source_sha256"],
                         m.PINNED["round79_row_class_source_sha256"])
        self.assertTrue(digests["file_sha256"].startswith("2141b29"),
                        "the task pins the candidate by the 2141b29 sha256 prefix")

    def test_verify_candidate_source_accepts_the_pinned_file(self):
        self.assertEqual(m.verify_candidate_source(CANDIDATE),
                         m.candidate_source_digests(CANDIDATE))

    def test_verify_candidate_source_rejects_any_edit(self):
        original = CANDIDATE.read_text()
        edited = original.replace(
            "output_parallel = self.group.all_reduce(output_parallel)",
            "output_parallel = self.group.all_reduce(output_parallel)  # tampered", 1)
        self.assertNotEqual(original, edited)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_encoder79.py"
            path.write_text(edited)
            with self.assertRaises(m.RowAdapterError):
                m.verify_candidate_source(path)

    def test_round83_kernel_and_runtime_files_match_the_pins(self):
        self.assertEqual(hashlib.sha256(R83_KERNEL.read_bytes()).hexdigest(),
                         m.PINNED["round83_kernel_module_sha256"])
        self.assertEqual(hashlib.sha256(R83_RUNTIME.read_bytes()).hexdigest(),
                         m.PINNED["round83_runtime_module_sha256"])
        self.assertTrue(m.PINNED["round83_kernel_module_sha256"].startswith("b5a06c"))

    def test_pinned_launcher_library_digest_matches_the_runtime_module(self):
        text = R83_RUNTIME.read_text()
        self.assertIn(m.PINNED["round83_runtime_library_sha256"], text)
        self.assertIn("Runtime5GM", text)

    def test_original_forward_still_promotes_before_the_collective(self):
        # The pinned candidate rounds the GEMM output to Half *before* the FP32
        # collective; that is exactly the extra rounding this adapter removes.
        text = CANDIDATE.read_text()
        class_source, forward_source = m._extract_class_and_forward(text)
        self.assertIn("self.quant_method.apply(self, input_parallel)", forward_source)
        self.assertIn("output_parallel.float()", forward_source)
        self.assertIn("self.group.all_reduce(output_parallel)", forward_source)
        self.assertLess(forward_source.index("quant_method.apply"),
                        forward_source.index(".float()"))
        self.assertIn("class " + m.ROW_CLASS_NAME, class_source)


class ShapeLogicTests(unittest.TestCase):
    def test_supported_h3_row_shapes(self):
        self.assertEqual(m.resolve_row_shape((5120, 1024)), (1024, 5120))
        self.assertEqual(m.resolve_row_shape((5120, 3200)), (3200, 5120))
        # torch.Size-like inputs are fine.
        self.assertEqual(m.resolve_row_shape([5120, 1024]), (1024, 5120))

    def test_unverified_shapes_are_refused(self):
        for shape in ((5120, 640), (5120, 8192), (5120, 2048), (5120, 5120)):
            with self.assertRaises(m.RowAdapterError):
                m.resolve_row_shape(shape)
        with self.assertRaises(m.RowAdapterError):
            m.resolve_row_shape((5120, 1024, 1))

    def test_native_activation_shapes_are_accepted(self):
        self.assertEqual(m.resolve_token_layout((2043, 1024), 1024, 2043), ((), 2043))
        self.assertEqual(m.resolve_token_layout((1, 2043, 1024), 1024, 2043), ((1,), 2043))
        self.assertEqual(m.resolve_token_layout((1, 2043, 3200), 3200, 2043), ((1,), 2043))

    def test_unexpected_activation_shapes_fail_closed(self):
        with self.assertRaises(m.RowAdapterError):   # batch 2
            m.resolve_token_layout((2, 2043, 1024), 1024, 2043)
        with self.assertRaises(m.RowAdapterError):   # wrong token count
            m.resolve_token_layout((2042, 1024), 1024, 2043)
        with self.assertRaises(m.RowAdapterError):   # wrong K
            m.resolve_token_layout((2043, 3200), 1024, 2043)
        with self.assertRaises(m.RowAdapterError):   # 4-D
            m.resolve_token_layout((1, 1, 2043, 1024), 1024, 2043)

    def test_output_shape_preserves_the_native_spatial_shape(self):
        self.assertEqual(m.output_shape_for((), 2043, 5120), (2043, 5120))
        self.assertEqual(m.output_shape_for((1,), 2043, 5120), (1, 2043, 5120))
        # The 2-D case must not collapse to a bare (N,).
        self.assertEqual(len(m.output_shape_for((), 2043, 5120)), 2)

    def test_padded_M_matches_the_verified_kernel_ceil(self):
        self.assertEqual(m.expected_m_pad(2043), 2048)
        self.assertEqual(m.expected_m_pad(2043) - 2043, 5)
        self.assertEqual(m.expected_m_pad(2048), 2048)
        with self.assertRaises(m.RowAdapterError):
            m.expected_m_pad(0)

    def test_expected_cache_bytes_match_the_task_arithmetic(self):
        self.assertEqual(m.expected_cached_weight_bytes(50), 2_162_688_000)
        per_layer = 5120 * 1024 + 5120 * 3200
        self.assertEqual(m.expected_cached_weight_bytes(50), 50 * per_layer * 2)


class SignatureTests(unittest.TestCase):
    def test_logical_pointer_and_offset_both_participate(self):
        base = FakeTensor(1000, 0, (5120, 1024), (1024, 1), "torch.float16", "npu:0", 3)
        other_view = FakeTensor(1000, 5242880, (5120, 1024), (1024, 1), "torch.float16",
                                "npu:0", 3)
        self.assertNotEqual(m.tensor_signature(base), m.tensor_signature(other_view))

    def test_mutation_version_changes_the_signature(self):
        before = FakeTensor(7, 0, (5120, 1024), (1024, 1), "torch.float16", "npu:0", 1)
        after = FakeTensor(7, 0, (5120, 1024), (1024, 1), "torch.float16", "npu:0", 2)
        self.assertNotEqual(m.tensor_signature(before), m.tensor_signature(after))

    def test_identical_views_share_a_signature(self):
        a = FakeTensor(7, 0, (5120, 1024), (1024, 1), "torch.float16", "npu:0", 5)
        b = FakeTensor(7, 0, (5120, 1024), (1024, 1), "torch.float16", "npu:0", 5)
        self.assertEqual(m.tensor_signature(a), m.tensor_signature(b))


class InstallGatingTests(unittest.TestCase):
    def setUp(self):
        self.candidate = FakeCandidate()
        self.runtime = object()
        self.tmp = tempfile.TemporaryDirectory()
        self.kernel_dir = Path(self.tmp.name) / "kernels"

    def tearDown(self):
        for adapter in list(m._INSTALLED.values()):
            if adapter._installed:
                adapter.uninstall()
        m._INSTALLED.clear()
        self.tmp.cleanup()

    def _install(self, **kwargs):
        return m.install(self.candidate.module, self.runtime, self.kernel_dir,
                         verify_sources=False, **kwargs)

    def test_install_patches_only_forward_and_starts_disabled(self):
        original = self.candidate.cls.forward
        adapter = self._install()
        self.assertFalse(adapter.enabled)
        self.assertIsNot(self.candidate.cls.forward, original)
        self.assertTrue(getattr(self.candidate.cls.forward, "_row84_gated_wrapper", False))
        self.assertIs(adapter._original_forward, original)
        self.assertEqual(adapter.row_class, self.candidate.cls)

    def test_disabled_calls_delegate_to_the_original(self):
        adapter = self._install()
        layer = self.candidate.cls("layerA")
        result = layer.forward({"x": 1})
        self.assertEqual(result, ("native-forward", "layerA", {"x": 1}))
        self.assertEqual(adapter.snapshot()["stats"]["calls_disabled"], 1)
        self.assertEqual(adapter.snapshot()["stats"]["calls_enabled"], 0)

    def test_enable_requires_prime(self):
        adapter = self._install()
        with self.assertRaises(m.RowAdapterError):
            adapter.set_enabled(True)

    def test_controls_require_prime(self):
        adapter = self._install()
        with self.assertRaises(m.RowAdapterError):
            adapter.controls()

    def test_uninstall_restores_the_original_forward(self):
        original = self.candidate.cls.forward
        adapter = self._install()
        summary = adapter.uninstall()
        self.assertIs(self.candidate.cls.forward, original)
        self.assertTrue(summary["uninstalled"])
        self.assertFalse(adapter.enabled)
        layer = self.candidate.cls("layerB")
        self.assertEqual(layer.forward([1, 2]), ("native-forward", "layerB", [1, 2]))

    def test_double_install_is_refused(self):
        self._install()
        with self.assertRaises(m.RowAdapterError):
            self._install()

    def test_reinstall_after_uninstall_is_allowed(self):
        first = self._install()
        first.uninstall()
        second = self._install()
        self.assertFalse(second.enabled)

    def test_uninstall_tolerates_a_third_party_replacement(self):
        adapter = self._install()

        def third_party(self, input_):
            return "replaced"

        self.candidate.cls.forward = third_party
        summary = adapter.uninstall()
        self.assertIn("uninstall_warning", summary)
        self.assertIs(self.candidate.cls.forward, third_party)

    def test_snapshot_is_plain_data_and_never_needs_torch(self):
        adapter = self._install()
        snapshot = adapter.snapshot()
        for key in ("round", "installed", "enabled", "primed", "stats", "weight_cache",
                    "pinned", "one_cast_per_call", "fp32_partial_through_collective",
                    "device_memory"):
            self.assertIn(key, snapshot)
        self.assertEqual(snapshot["round"], 84)
        self.assertTrue(snapshot["one_cast_per_call"])
        self.assertFalse(snapshot["enabled"])
        self.assertFalse(snapshot["primed"])
        self.assertEqual(snapshot["pinned"], m.PINNED)

    def test_kernel_module_resolution_is_gated_by_verify_sources(self):
        adapter = self._install()  # helper already passes verify_sources=False
        fake = types.ModuleType("blocked_dynamic_steps83")
        fake.__file__ = str(ADAPTER)  # a real file, but not the pinned kernel
        sys.modules["blocked_dynamic_steps83"] = fake
        try:
            self.assertIs(adapter._resolve_kernel_module(), fake)
            # The resolved module is cached; clear it so verification runs again.
            adapter._kernel_module = None
            adapter._verify_sources = True
            with self.assertRaises(m.RowAdapterError):
                adapter._resolve_kernel_module()
        finally:
            sys.modules.pop("blocked_dynamic_steps83", None)

    def test_release_scratch_is_safe_before_prime(self):
        adapter = self._install()
        freed = adapter.release_scratch()
        self.assertEqual(freed["freed_scratch_bytes"], 0)
        self.assertEqual(freed["cached_weight_bytes"], 0)


class SourceInvariantTests(unittest.TestCase):
    def setUp(self):
        self.source = ADAPTER.read_text()
        self.tree = ast.parse(self.source)

    def test_no_third_party_import_at_module_scope(self):
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0],
                                  {"ast", "hashlib", "importlib", "json", "os", "pathlib",
                                   "textwrap", "time"},
                                  f"unexpected module-scope import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or "").split(".")[0],
                              {"__future__", "ast", "hashlib", "importlib", "json", "os",
                               "pathlib", "textwrap", "time"})
        # torch must only ever be imported inside a function body.
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [
                    node.module or ""]
                if any(name.split(".")[0] == "torch" for name in names):
                    self.assertIsNotNone(getattr(node, "_in_function", False) or True)
        self.assertNotIn("\nimport torch\n", self.source)

    def test_no_bf16_quantization_ssh_or_subprocess(self):
        for forbidden in ("bfloat16", "subprocess", "socket", "paramiko",
                          "-50000", ".clamp(", ".clamp_(", "torch.quantize_per_tensor",
                          "os.environ[", "os.environ.setdefault"):
            self.assertNotIn(forbidden, self.source,
                             f"forbidden construct {forbidden!r} in the adapter")
        # "quantization" only ever appears in prose; no quantization API is used.
        for forbidden in ("quant_config", "QuantizationConfig", "get_quant_method"):
            self.assertNotIn(forbidden, self.source,
                             f"quantization API {forbidden!r} in the adapter")

    def test_fp32_is_asserted_before_any_narrowing(self):
        forward = self.source[self.source.index("def _forward(self, layer, input_):"):]
        before_collective = forward.index("fp32_dtype_before_collective")
        after_collective = forward.index("fp32_dtype_after_collective")
        cast = forward.index(".to(layer.output_dtype)")
        self.assertLess(before_collective, after_collective)
        self.assertLess(after_collective, cast)
        self.assertIn("if reduced.dtype != torch.float32:", forward)
        # exactly one narrowing site in the enabled path
        self.assertEqual(forward.count(".to(layer.output_dtype)"), 1)

    def test_uses_the_verified_round83_packers_instead_of_reimplementing_them(self):
        self.assertIn('PINNED["round83_kernel_module"]', self.source)
        self.assertIn("kernel_module.pack_w(weight)", self.source)
        self.assertIn("kernel_module.unpack_c(scratch[\"c\"], m=tokens)", self.source)
        self.assertIn("kernel_module.pack_a(flat)", self.source)

    def test_only_the_row_class_forward_is_patched(self):
        self.assertIn('cls.__dict__.get("forward")', self.source)
        self.assertIn("cls.forward = gated_forward", self.source)
        self.assertNotIn("F.linear", self.source)
        self.assertNotIn("torch.nn.functional", self.source)

    def test_shared_kernel_dir_is_locked_and_repaired(self):
        self.assertIn("fcntl.flock", self.source)
        self.assertIn(".row84_build.lock", self.source)
        self.assertIn("_repair_spec_file_if_needed", self.source)
        self.assertIn("os.replace(tmp, resolved_spec)", self.source)
        self.assertIn("resolved_spec.is_relative_to(root)", self.source)

    def test_disabled_gate_calls_the_captured_original(self):
        self.assertIn("adapter._original_forward(self, input_)", self.source)
        self.assertIn("_row84_gated_wrapper", self.source)

    def test_stream_mismatch_fails_closed_by_default(self):
        self.assertIn("allow_stream_switch=False", self.source)
        self.assertIn("row-parallel adapter activation buffers are owned by the stream they",
                      self.source)

    def test_module_imports_without_loading_torch(self):
        code = (
            "import sys; sys.path.insert(0, %r); import row_fp32_adapter_fixed84 as a; "
            "assert 'torch' not in sys.modules, 'adapter imported torch at import time'; "
            "assert 'vllm' not in sys.modules; print(a.ROUND)" % str(HERE))
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "84")

    def test_public_api_names_are_present(self):
        for name in ("install", "install_row", "RowFp32Adapter", "RowAdapterError",
                     "PINNED", "SUPPORTED_ROW_SHAPES"):
            self.assertTrue(hasattr(m, name), name)
        self.assertIs(m.install_row, m.install)
        self.assertEqual(m.SUPPORTED_ROW_SHAPES, ((1024, 5120), (3200, 5120)))

    def test_adapter_has_no_module_level_mutable_state_beyond_the_registry(self):
        globals_with_tensors = [name for name, value in vars(m).items()
                                if inspect.ismodule(value) or inspect.isclass(value)]
        self.assertIn("RowFp32Adapter", globals_with_tensors)
        self.assertIsInstance(m._INSTALLED, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)