#!/usr/bin/env python3
"""Static contract tests for the Round79 FP16 encoder candidate.

Standard library only: the macOS development host has no Torch, vLLM or vLLM-Omni,
so the candidate is never imported here. The tests read the generated candidate
as text, parse it with ``ast``, and compare it against the pinned production
source. That keeps the guarantees honest:

* the candidate is reproduced exactly from the pinned source and declared edits;
* the three BF16 I/O conversions are gone;
* the FP16 branches implement the reference HF rounding, not an FP32 gamma;
* the BF16 compatibility paths, the vision SDPA policy and the text SDPA call are
  untouched;
* nothing here claims a numeric pass: no tolerance exists and nothing executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

CANDIDATE_DIR = Path(__file__).resolve().parent
BUILD = CANDIDATE_DIR / "build_candidate79.py"

sys.path.insert(0, str(CANDIDATE_DIR))
import build_candidate79 as build  # noqa: E402  (path set above)

SOURCE = build.SOURCE
CANDIDATE = CANDIDATE_DIR / "candidate_encoder79.py"
MANIFEST = CANDIDATE_DIR / "build_manifest79.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_text() -> str:
    return CANDIDATE.read_text(encoding="utf-8")


def candidate_tree() -> ast.Module:
    return ast.parse(candidate_text())


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def find_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in find_class(tree, class_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"{class_name}.{method_name} not found")


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def attribute_names(node: ast.AST, base: str) -> set[str]:
    """Attributes of ``base`` referenced anywhere under ``node`` (e.g. torch.bfloat16)."""
    found = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == base
        ):
            found.add(child.attr)
    return found


def call_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            target = child.func
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
            elif isinstance(target, ast.Name):
                names.add(target.id)
    return names


class ProvenanceContracts(unittest.TestCase):
    def test_source_is_the_pinned_production_encoder(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["source_sha256"],
            "7c22d98c28f4e93550fd3848bfc174121250259fdf50bcd2f7920f826acf311d",
        )
        self.assertEqual(sha256_file(SOURCE), manifest["source_sha256"])
        self.assertEqual(manifest["source_line_count"], 1605)

    def test_manifest_names_the_candidate_file_and_hash_used_by_the_driver(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        # The root driver verifies the loaded module only through this key.
        self.assertEqual(manifest["candidate_path"], "artifacts/diagnostics/round79/candidate/candidate_encoder79.py")
        self.assertEqual(manifest["candidate_sha256"], sha256_file(CANDIDATE))
        self.assertEqual(CANDIDATE.name, "candidate_encoder79.py")

    def test_candidate_is_reproducible_from_pinned_source_and_declared_edits(self):
        original = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
        regenerated, resolved = build.apply_edits(original)
        self.assertEqual("".join(regenerated), candidate_text())
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["edit_count"], len(build.EDITS))
        self.assertEqual(len(resolved), len(build.EDITS))

    def test_every_edit_span_actually_differs_and_stays_inside_source(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source_lines = SOURCE.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(manifest["edits"]), len(build.EDITS))
        for edit in manifest["edits"]:
            start, end = edit["original_lines"]
            self.assertGreaterEqual(start, 1)
            self.assertLessEqual(end, len(source_lines))
            self.assertLessEqual(start, end)
        spans = sorted(tuple(edit["original_lines"]) for edit in manifest["edits"])
        for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLess(previous_end, next_start, "declared edit spans overlap")

    def test_candidate_compiles(self):
        compile(candidate_text(), str(CANDIDATE), "exec")

    def test_no_unexpected_module_side_effects_at_import(self):
        # Any import-time assertion on the operator flags would reject a driver
        # that legitimately sets TEXT_HF_MATH=1 / VISION_FP32=0.
        tree = candidate_tree()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                continue
            self.assertNotIsInstance(node, ast.Assert, "module-level assert would gate import")
            self.assertNotIsInstance(node, ast.Raise, "module-level raise would gate import")


class BF16BoundaryRemovalContracts(unittest.TestCase):
    def test_encode_has_no_hardcoded_bf16_conversion(self):
        encode = find_method(candidate_tree(), "MiniMaxH3Qwen3VLEncoder", "_encode")
        self.assertNotIn("bfloat16", attribute_names(encode, "torch"))

    def test_encode_takes_io_dtype_from_the_stored_vision_parameters(self):
        encode = find_method(candidate_tree(), "MiniMaxH3Qwen3VLEncoder", "_encode")
        io_dtype_calls = [
            child
            for child in ast.walk(encode)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "_r79_io_dtype"
        ]
        self.assertEqual(len(io_dtype_calls), 2)
        source = candidate_text()
        self.assertIn("pixel_values.to(device, self._r79_io_dtype())", source)
        self.assertIn("pixel_values_videos.to(device, self._r79_io_dtype())", source)
        self.assertNotIn("pixel_values.to(device, torch.bfloat16)", source)

    def test_encode_returns_the_graph_dtype_instead_of_a_bf16_cast(self):
        encode = find_method(candidate_tree(), "MiniMaxH3Qwen3VLEncoder", "_encode")
        returns = [node for node in ast.walk(encode) if isinstance(node, ast.Return)]
        final = returns[-1]
        self.assertIsInstance(final.value, ast.Name)
        self.assertEqual(final.value.id, "hidden")
        self.assertNotIn("hidden.to(torch.bfloat16)", candidate_text())

    def test_io_dtype_accessor_follows_vision_patch_embedding_parameters(self):
        method = find_method(candidate_tree(), "MiniMaxH3Qwen3VLEncoder", "_r79_io_dtype")
        returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        node = returns[0].value
        self.assertIsInstance(node, ast.Attribute)
        self.assertEqual(node.attr, "dtype")
        self.assertEqual(ast.unparse(node), "self.vision.patch_embed.proj.weight.dtype")


class ReferenceRoundingContracts(unittest.TestCase):
    def test_fp16_rmsnorm_rounds_normalised_value_before_gamma(self):
        method = find_method(candidate_tree(), "MiniMaxH3Qwen3VLRMSNorm", "forward_npu")
        text = ast.unparse(method)
        # Two reference rounding points: FP32 reduction -> cast -> FP16 gamma.
        self.assertIn(".to(input_dtype)", text)
        self.assertIn("self.weight * normalized", text)
        # The FP16 branch must not fold gamma into the FP32 point math.
        self.assertNotIn("normalized.float() * self.weight.float()", text.split("if not (")[0])

    def test_bf16_rmsnorm_block_is_preserved_verbatim(self):
        source = SOURCE.read_text(encoding="utf-8")
        original = "            output = (normalized.float() * self.weight.float()).to(input_dtype)"
        self.assertEqual(source.count(original), 1)
        self.assertEqual(candidate_text().count(original), 1)

    def test_fp16_text_rope_returns_activation_dtype_cos_sin(self):
        method = find_method(candidate_tree(), "MiniMaxH3Qwen3VLTextRotaryEmbedding", "forward")
        text = ast.unparse(method)
        self.assertIn("x.dtype == torch.float16", text)
        self.assertIn("emb.cos().to(x.dtype)", text)
        self.assertIn("emb.sin().to(x.dtype)", text)

    def test_position_interpolation_uses_fp32_weights_and_sum_for_fp16(self):
        method = find_method(candidate_tree(), "MiniMaxH3Qwen3VLVisionModel", "fast_pos_embed_interpolate")
        text = ast.unparse(method)
        self.assertIn("self.pos_embed.weight.dtype in (torch.bfloat16, torch.float16)", text)
        self.assertIn("weight_dtype = torch.float32 if npu_fp32_interpolation", text)
        self.assertIn("embedding_samples.float()", text)
        self.assertIn(".to(self.pos_embed.weight.dtype)", text)
        # The FP16 path must not depend on an opt-in flag.
        gate = text.split("npu_fp32_interpolation =")[1].split("#")[0]
        self.assertNotIn("_vision_fp32_enabled", gate)

    def test_fp16_embedding_reduce_promotes_collective_to_fp32(self):
        function = find_function(candidate_tree(), "_text_hf_embedding_all_reduce")
        text = ast.unparse(function)
        self.assertIn("value.dtype == torch.float16", text)
        self.assertEqual(text.count("group.all_reduce(value.float()).to(value.dtype)"), 2)
        self.assertEqual(text.count("group.all_reduce(value)"), 1)


class CompatibilityContracts(unittest.TestCase):
    def test_bf16_compatibility_classes_still_gate_on_bfloat16_only(self):
        tree = candidate_tree()
        linear = find_method(tree, "_NpuBF16CompatibleLinear", "forward")
        norm = find_method(tree, "_NpuBF16CompatibleLayerNorm", "forward")
        # These compatibility shims exist for the 910A BF16 defect and must stay
        # BF16-gated; FP16 deliberately takes the native operator instead of an
        # unvalidated whole-FP32 "shadow" implementation.
        self.assertIn("bfloat16", attribute_names(linear, "torch"))
        self.assertNotIn("float16", attribute_names(linear, "torch"))
        self.assertIn("bfloat16", attribute_names(norm, "torch"))
        self.assertNotIn("float16", attribute_names(norm, "torch"))

    def test_vision_fp32_sdpa_policy_is_unchanged(self):
        text = ast.unparse(find_function(candidate_tree(), "_vision_sdpa_bf16_compatible"))
        self.assertIn("torch.bfloat16", text)
        self.assertIn("torch.float16", text)
        self.assertIn("torch.softmax", text)
        self.assertEqual(
            ast.unparse(find_function(candidate_tree(), "_vision_sdpa_bf16_compatible")),
            ast.unparse(find_function(ast.parse(SOURCE.read_text(encoding="utf-8")), "_vision_sdpa_bf16_compatible")),
        )

    def test_text_sdpa_call_site_is_unchanged(self):
        original = SOURCE.read_text(encoding="utf-8")
        candidate = candidate_text()
        body = """    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )
"""
        self.assertIn(body, original)
        self.assertIn(body, candidate)

    def test_no_global_functional_monkeypatch_and_no_bf16_kernel_reuse(self):
        text = candidate_text()
        for forbidden in (
            "F.scaled_dot_product_attention =",
            "torch.nn.functional.scaled_dot_product_attention =",
            "setattr(F,",
            "fp16_unet",
            "vision_cast73",
            "fa910a",
            "FlashAttn910A",
        ):
            self.assertNotIn(forbidden, text)

    def test_round65_padded_shadow_symbols_do_not_exist_in_this_source(self):
        # The delegation note asked to disable TEXT_PAD_KV / TEXT_FP32_LINEAR /
        # _text_fp32_local_linear. They are absent from this R76 source (they are
        # round65 candidate constructs), so there was nothing to disable and the
        # candidate must not silently introduce them.
        for symbol in ("TEXT_PAD_KV", "TEXT_FP32_LINEAR", "_text_fp32_local_linear", "_vision_fp32_shadow"):
            self.assertNotIn(symbol, SOURCE.read_text(encoding="utf-8"))
            self.assertNotIn(symbol, candidate_text())


class VisionOnlyProbeContracts(unittest.TestCase):
    """What a single-rank vision-only probe has to know before it runs."""

    def test_vision_rotary_table_is_a_plain_attribute_not_a_buffer(self):
        # ``_inv_freq`` is intentionally not a registered buffer, so ``.to()``
        # never moves it and a meta-device construction leaves it on meta. A
        # vision-only probe must therefore install a real CPU FP32 rotary module
        # before the forward pass; that is a property of the source, not of the
        # candidate, and it is asserted here so the requirement is not forgotten.
        cls = find_class(candidate_tree(), "MiniMaxH3Qwen3VLVisionRotaryEmbedding")
        init = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"][0]
        registered = [
            child
            for child in ast.walk(init)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "register_buffer"
        ]
        self.assertEqual(registered, [])
        assigned = [
            target.attr
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        ]
        self.assertIn("_inv_freq", assigned)

    def test_vision_tower_can_be_constructed_without_text_model_or_encoder_group(self):
        cls = find_class(candidate_tree(), "MiniMaxH3Qwen3VLVisionModel")
        init = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"][0]
        arguments = [argument.arg for argument in init.args.args]
        self.assertEqual(arguments, ["self", "config"])

    def test_vision_path_does_not_require_text_only_flags(self):
        tree = candidate_tree()
        vision_classes = (
            "MiniMaxH3Qwen3VLVisionModel",
            "MiniMaxH3Qwen3VLVisionBlock",
            "MiniMaxH3Qwen3VLVisionAttention",
            "MiniMaxH3Qwen3VLVisionMLP",
            "MiniMaxH3Qwen3VLVisionPatchMerger",
            "MiniMaxH3Qwen3VLVisionPatchEmbed",
        )
        for name in vision_classes:
            text = ast.unparse(find_class(tree, name))
            self.assertNotIn("_text_hf_math_enabled", text, name)
            self.assertNotIn("_text_fp32", text, name)
        # The topology and merger still use the HF-faithful tanh GELU.
        mlp = ast.unparse(find_class(tree, "MiniMaxH3Qwen3VLVisionMLP"))
        self.assertIn("approximate='tanh'", mlp)


class HonestyContracts(unittest.TestCase):
    def test_manifest_does_not_claim_execution_or_accuracy(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertFalse(manifest["executed"])
        self.assertFalse(manifest["NPU_precision_established"])
        self.assertFalse(manifest["CPU_FP16_gold_consumed"])
        self.assertFalse(manifest["tolerance_set"])
        self.assertEqual(manifest["production_files_modified"], [])

    def test_self_check_fails_closed_on_mixed_dtypes(self):
        method = find_method(candidate_tree(), "MiniMaxH3Qwen3VLEncoder", "r79_self_check")
        raises = [node for node in ast.walk(method) if isinstance(node, ast.Raise)]
        self.assertGreaterEqual(len(raises), 2)
        text = ast.unparse(method)
        self.assertIn("does not match parameters", text)
        self.assertIn("not uniform", text)
        self.assertIn("self.text_model.embed_tokens.weight.dtype", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)