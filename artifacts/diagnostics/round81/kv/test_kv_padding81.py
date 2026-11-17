#!/usr/bin/env python3
"""Static contract tests for the Round81 FP16 K/V zero-padding adapter.

Standard library only: no torch, no torch_npu, no NPU, no model execution.
The adapter is imported for its AST/transform logic and the real Round79
candidate source is parsed as text -- never imported -- so these tests run on a
machine without torch and cannot claim any numerical result.

What these tests CAN establish: the transform is exactly two call-site rewrites
(K/V only), it is fail-closed when the source does not match, and the adapter
never patches global ``torch.nn.functional``.
What they CANNOT establish: that the padded GEMM is repeat-stable or accurate on
910A. That requires the root's private NPU run.
"""

import ast
import hashlib
import json
import py_compile
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "kv_padding81.py"
MANIFEST = HERE / "manifest81.json"
README = HERE / "README.md"
PROJECT = HERE.parents[3]
CANDIDATE = PROJECT / "artifacts/diagnostics/round79/candidate/candidate_encoder79.py"

sys.path.insert(0, str(HERE))
import kv_padding81 as kv  # noqa: E402  (stdlib-only module)


def adapter_tree():
    return ast.parse(ADAPTER.read_text(encoding="utf-8"))


def adapter_source():
    return ADAPTER.read_text(encoding="utf-8")


def real_forward_source():
    """Extract the real candidate's TextAttention.forward source as text."""
    text = CANDIDATE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "MiniMaxH3Qwen3VLTextAttention"]
    if len(classes) != 1:
        raise AssertionError(f"expected one TextAttention class, found {len(classes)}")
    forwards = [n for n in classes[0].body
                if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    if len(forwards) != 1:
        raise AssertionError(f"expected one forward, found {len(forwards)}")
    segment = ast.get_source_segment(text, forwards[0])
    if not segment:
        raise AssertionError("could not extract forward source segment")
    return text, segment


class ProvenanceTests(unittest.TestCase):
    def test_adapter_compiles(self):
        py_compile.compile(str(ADAPTER), doraise=True)

    def test_candidate_and_manifest_are_pinned(self):
        self.assertTrue(MANIFEST.exists(), "manifest81.json is required")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["candidate_sha256"], kv.__dict__.get(
            "PINNED_CANDIDATE_SHA256", manifest["candidate_sha256"]))
        actual = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
        self.assertEqual(actual, manifest["candidate_sha256"],
                         "Round79 candidate changed; re-pin before running")
        self.assertEqual(hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
                         manifest["adapter_sha256"])
        self.assertEqual(manifest["original_cpu_gold"]["receipt_sha256"],
                         "44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9")
        self.assertEqual(manifest["original_cpu_gold"]["source_sha256"],
                         "50165e77adb393b4573fbe0fdb29c62a6ddb14b8f39082d0ea67e904848010ea")
        # The adapter must not require BF16 agreement: the user's gate is CPU FP16.
        self.assertFalse(manifest["original_cpu_gold"]["bf16_agreement_required"])
        self.assertEqual(manifest["expected_rewritten_call_count"], 2)
        self.assertEqual(tuple(manifest["expected_matched_calls"]), kv.EXPECTED_TARGETS)
        self.assertFalse(manifest["global_functional_patched"])


class LazyImportTests(unittest.TestCase):
    def test_no_top_level_torch_import(self):
        """The module must import on a machine without torch."""
        for node in adapter_tree().body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                self.assertNotIn("torch", names, "torch must not be a top-level import")
                self.assertNotIn("torch_npu", names)
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "torch")

    def test_torch_is_imported_lazily_inside_a_function(self):
        source = adapter_source()
        self.assertIn("def _torch():", source)
        tree = adapter_tree()
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_torch":
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Import):
                        found = any(a.name == "torch" for a in inner.names)
        self.assertTrue(found, "the lazy torch import must live inside _torch()")


class TransformTests(unittest.TestCase):
    def test_constants_are_the_round80_stable_geometry(self):
        self.assertEqual(kv.ORIGINAL_KV_ROWS, 128)
        self.assertEqual(kv.PAD_OUTPUT_ROWS, 256)
        self.assertEqual(kv.EXPECTED_CACHE_ENTRIES, 100)
        self.assertEqual(kv.EXPECTED_TARGETS, ("k_weight", "v_weight"))

    def test_real_candidate_transform_rewrites_exactly_k_and_v(self):
        _, segment = real_forward_source()
        new_source, matched = kv.transform_forward_source(segment)
        self.assertEqual(matched, ("k_weight", "v_weight"))
        self.assertEqual(new_source.count("F.linear(hidden_states, q_weight)"), 1,
                         "Q projection must be untouched")
        self.assertEqual(new_source.count("_r81_kv_linear(hidden_states, k_weight)"), 1)
        self.assertEqual(new_source.count("_r81_kv_linear(hidden_states, v_weight)"), 1)
        compile(new_source, "<transformed>", "exec")

    def test_real_candidate_transform_emits_no_other_linear_rewrites(self):
        _, segment = real_forward_source()
        new_source, _ = kv.transform_forward_source(segment)
        tree = ast.parse(new_source)
        helper_calls = [n for n in ast.walk(tree)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id == "_r81_kv_linear"]
        self.assertEqual(len(helper_calls), 2)
        f_linear_calls = [n for n in ast.walk(tree)
                          if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute)
                          and n.func.attr == "linear"]
        self.assertEqual(len(f_linear_calls), 1, "only the Q F.linear call may remain")

    def test_transform_rejects_source_without_targets(self):
        source = "def forward(self, hidden_states):\n    return hidden_states\n"
        with self.assertRaises(kv.KVPaddingError):
            kv.transform_forward_source(source)

    def test_transform_rejects_partial_rewrite(self):
        source = (
            "def forward(self, hidden_states):\n"
            "    key = F.linear(hidden_states, k_weight)\n"
            "    return key\n"
        )
        with self.assertRaises(kv.KVPaddingError):
            kv.transform_forward_source(source)

    def test_transform_rejects_duplicate_target_calls(self):
        source = (
            "def forward(self, hidden_states):\n"
            "    a = F.linear(hidden_states, k_weight)\n"
            "    b = F.linear(hidden_states, v_weight)\n"
            "    c = F.linear(hidden_states, k_weight)\n"
            "    return a, b, c\n"
        )
        with self.assertRaises(kv.KVPaddingError):
            kv.transform_forward_source(source)

    def test_transform_does_not_touch_keyword_form_or_other_weights(self):
        source = (
            "def forward(self, hidden_states):\n"
            "    a = F.linear(hidden_states, o_weight)\n"
            "    return a\n"
        )
        with self.assertRaises(kv.KVPaddingError):
            kv.transform_forward_source(source)


class AdapterContractTests(unittest.TestCase):
    def test_public_api_surface(self):
        for name in ("install", "prime", "snapshot", "clear", "uninstall"):
            self.assertTrue(hasattr(kv.KVPaddingAdapter, name), name)
        self.assertTrue(callable(kv.install))

    def test_no_global_functional_patch(self):
        source = adapter_source()
        for forbidden in ("F.linear =", "torch.nn.functional.linear =",
                          "torch.nn.functional =", "nn.functional.linear ="):
            self.assertNotIn(forbidden, source)

    def test_prime_runs_under_inference_or_no_grad(self):
        source = adapter_source()
        self.assertIn("_no_grad_context(torch)", source)
        self.assertIn("def _no_grad_context(torch):", source)
        self.assertIn("inference_mode", source)
        tree = adapter_tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "prime":
                withs = [n for n in ast.walk(node) if isinstance(n, ast.With)]
                self.assertTrue(withs, "prime() must run inside a grad-disabled context")

    def test_cache_copies_detached_parameters_and_verifies_zero_pad(self):
        source = adapter_source()
        self.assertIn("copy_(weight.detach())", source)
        self.assertIn("padded_requires_grad", source)
        self.assertIn("padded_is_leaf", source)
        self.assertIn("padded.storage_offset()) != 0", source)
        self.assertIn("tail_nonzero != 0", source)
        self.assertIn("prefix_exact", source)
        # Verification must stay device-side: no per-entry D2H value hashing.
        self.assertNotIn('reference = weight.detach().to("cpu")', source)
        self.assertIn("torch.equal(padded[:rows], weight.detach())", source)

    def test_cache_is_fail_closed_on_signature_change(self):
        source = adapter_source()
        self.assertIn("refusing to silently re-cache a mutated weight", source)
        self.assertIn("untyped_storage().data_ptr()", source)
        self.assertIn("int(weight.storage_offset())", source)
        self.assertIn("weight._version", source)

    def test_padding_only_applies_to_small_half_npu_kv(self):
        source = adapter_source()
        self.assertIn('x.device.type in ("npu", "privateuseone")', source)
        self.assertIn("torch.float16", source)
        self.assertIn("weight.shape[0] == ORIGINAL_KV_ROWS", source)
        self.assertIn("torch.nn.functional.linear(x, weight)", source,
                      "non-NPU/dtype passthrough must keep the original call")

    def test_snapshot_reports_measurements_and_honest_flags(self):
        snapshot_src = adapter_source()
        for key in ("cache_entries", "expected_cache_entries", "cache_bytes",
                    "padding_bytes", "all_prefix_exact", "tail_nonzero_count",
                    "all_cached_padded_are_zero_offset", "no_cached_padded_requires_grad",
                    "param_signature_contract", "stats", "transform"):
            self.assertIn(key, snapshot_src, key)
        self.assertIn('"not_executed_here": True', snapshot_src)
        self.assertIn('"proves_full_encoder_precision": False', snapshot_src)

    def test_install_uses_live_module_namespace(self):
        """Late module-level patches (e.g. the F32 MATH SDPA) must still resolve."""
        source = adapter_source()
        self.assertIn("module_globals = getattr(candidate, \"__dict__\", None)", source)
        self.assertNotIn("namespace = dict(module_globals)", source,
                         "a copied namespace would freeze module-level patches")
        self.assertIn('module_globals["_r81_kv_linear"] = self._kv_linear', source)
        self.assertIn("module_globals.pop(\"forward\", None)", source)

    def test_snapshot_is_json_serializable_without_torch(self):
        """The root puts snapshot() straight into an evidence receipt."""
        adapter = kv.KVPaddingAdapter(None)
        payload = adapter.snapshot()
        encoded = json.dumps(payload, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertFalse(decoded["installed"])
        self.assertFalse(decoded["primed"])
        self.assertEqual(decoded["cache_entries"], 0)
        self.assertEqual(decoded["expected_cache_entries"], 100)
        self.assertEqual(decoded["padded_output_rows"], 256)
        self.assertEqual(decoded["original_kv_rows"], 128)
        self.assertTrue(decoded["only_zero_output_rows_added"])
        self.assertTrue(decoded["trailing_output_discarded"])
        self.assertFalse(decoded["proves_full_encoder_precision"])

    def test_prime_geometry_matches_the_real_forward_slices(self):
        """prime() must warm exactly the tensors forward() will request."""
        _, segment = real_forward_source()
        tree = ast.parse(segment)
        sliced = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript)
                  and isinstance(n.value, ast.Attribute)
                  and n.value.attr == "weight"]
        self.assertEqual(len(sliced), 3, "Q/K/V weight slices expected in forward")
        source = adapter_source()
        # K start = q_local, V start = q_local + kv_local; same as forward.
        self.assertIn("for start in (q_local, q_local + local):", source)
        self.assertIn("q_local = int(attn.qkv_proj.local_num_heads) * int(attn.head_dim)", source)
        self.assertIn("local = int(attn.qkv_proj.local_num_kv_heads) * int(attn.head_dim)", source)
        self.assertIn('getattr(attn, "qkv_proj", None)', source)
        self.assertIn('getattr(layer, "self_attn", None)', source)

    def test_uninstall_restores_original_forward(self):
        source = adapter_source()
        self.assertIn("self._class.forward = self._original_forward", source)
        self.assertIn("__round81_original__", source)


class ReadmeTests(unittest.TestCase):
    def test_readme_states_not_executed_and_not_a_proof(self):
        self.assertTrue(README.exists(), "README.md must document the experiment")
        text = README.read_text(encoding="utf-8")
        self.assertIn("NOT EXECUTED", text)
        self.assertIn("256", text)
        self.assertIn("128", text)


if __name__ == "__main__":
    unittest.main()