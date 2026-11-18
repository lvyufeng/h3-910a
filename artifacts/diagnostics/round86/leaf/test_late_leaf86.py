#!/usr/bin/env python3
"""Stdlib-only tests for the Round86 late-layer FP16 leaf controls.

No torch, vLLM, NPU or network access is required or performed: this runs on a
plain laptop. It proves the *shape/geometry/order* contracts and the pinned
Round79 two-round norm source, never a numeric result. The real leaf numbers are
produced by the root driver on the leased NPU.
"""

import ast
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MODULE_PATH = HERE / "late_leaf86.py"
R79_CANDIDATE = REPO / "artifacts/diagnostics/round79/candidate/candidate_encoder79.py"

SOURCE = MODULE_PATH.read_text()
TREE = ast.parse(SOURCE)


def _load():
    sys.path.insert(0, str(HERE))
    try:
        return importlib.import_module("late_leaf86")
    finally:
        sys.path.pop(0)


leaf = _load()


# --- stand-ins: exactly the attribute surface the validators inspect ----------
class FakeTensor:
    def __init__(self, shape, dtype="torch.float16", device="npu:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device


def _faked_class(name, **attrs):
    """Instance of a dynamically named class so ``type(x).__name__`` matches."""
    return type(name, (), dict(attrs))()


def make_layer(ip=3200, intermediate=25600, gate_up_shape=(6400, 5120),
               down_shape=(5120, 3200), weight_dtype="torch.float16",
               norm_weight_shape=(5120,), norm_weight_dtype="torch.float16",
               eps=1e-6, bias=False, method_name="UnquantizedLinearMethod",
               layer_name="MiniMaxH3Qwen3VLTextDecoderLayer"):
    Method = type(method_name, (), {})
    gate_up = _faked_class(
        "MiniMaxH3Qwen3VLMergedColumnParallelLinear",
        intermediate_size_per_partition=ip,
        weight=FakeTensor(gate_up_shape, weight_dtype),
        bias=FakeTensor((gate_up_shape[0],), weight_dtype) if bias else None,
        quant_method=Method())
    down = _faked_class(
        "MiniMaxH3Qwen3VLRowParallelLinear",
        weight=FakeTensor(down_shape, weight_dtype),
        bias=FakeTensor((down_shape[0],), weight_dtype) if bias else None,
        quant_method=Method())
    mlp = _faked_class("MiniMaxH3Qwen3VLTextMLP", hidden_size=5120,
                       intermediate_size=intermediate,
                       gate_up_proj=gate_up, down_proj=down)
    norm = _faked_class("MiniMaxH3Qwen3VLRMSNorm",
                        weight=FakeTensor(norm_weight_shape, norm_weight_dtype),
                        variance_epsilon=eps)
    return _faked_class(layer_name, input_layernorm=norm, mlp=mlp)


class ModuleScopeTests(unittest.TestCase):
    def test_only_stdlib_imports_at_module_scope(self):
        allowed = {"ast", "hashlib", "math", "pathlib", "textwrap", "__future__"}
        for node in TREE.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                self.assertIn(root, allowed, f"module-scope import from {node.module}")

    def test_torch_is_lazy_inside_the_oracles_and_controls(self):
        for name in ("cpu_half_rmsnorm_reference", "cpu_half_linear_reference", "run_controls"):
            fn = next(n for n in TREE.body
                      if isinstance(n, ast.FunctionDef) and n.name == name)
            imports = [n for n in ast.walk(fn)
                       if isinstance(n, ast.Import)
                       and any(a.name == "torch" or a.name.startswith("torch.") for a in n.names)]
            self.assertTrue(imports, f"{name} must import torch lazily")

    def test_run_controls_has_the_exact_root_signature(self):
        fn = next(n for n in TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == "run_controls")
        names = [a.arg for a in fn.args.args]
        self.assertEqual(names, ["layer", "block_input", "mlp_input", "compare", "tensor_info"])
        self.assertEqual(fn.args.vararg, None)
        self.assertEqual(fn.args.kwarg, None)


class GeometryTests(unittest.TestCase):
    def test_actual_checkpoint_geometry_is_25600_and_not_the_stale_figure(self):
        g = leaf.expected_geometry()
        self.assertEqual(g["intermediate_size_actual_checkpoint"], 25600)
        self.assertEqual(g["intermediate_per_partition"], 3200)
        self.assertEqual(g["gate_up_weight_shape"], [6400, 5120])
        self.assertEqual(g["down_weight_shape"], [5120, 3200])
        self.assertEqual(g["rejected_stale_intermediate_size"], 27648)
        self.assertEqual(leaf.INTERMEDIATE_SIZE // leaf.WORLD_SIZE,
                         leaf.INTERMEDIATE_PER_PARTITION)
        self.assertNotEqual(leaf.INTERMEDIATE_PER_PARTITION, 3456)

    def test_activation_validator_accepts_both_native_shapes(self):
        for shape in ((1, 2043, 5120), (2043, 5120)):
            spec = leaf.validate_activation(FakeTensor(shape))
            self.assertEqual(spec["shape"], list(shape))
            self.assertEqual(spec["dtype"], "torch.float16")

    def test_activation_validator_fails_closed(self):
        cases = [
            FakeTensor((1, 2043, 5120), dtype="torch.float32"),
            FakeTensor((1, 2043, 5120), dtype="torch.bfloat16"),
            FakeTensor((1, 2043, 5120), device="cpu"),
            FakeTensor((1, 2043, 5120), device="cuda:0"),
            FakeTensor((1, 1024, 5120)),
            FakeTensor((1, 2043, 4096)),
            FakeTensor((2, 2043, 5120)),
            FakeTensor((1, 2043, 5120, 1)),
        ]
        for value in cases:
            with self.assertRaises(leaf.LeafError):
                leaf.validate_activation(value)
        with self.assertRaises(leaf.LeafError):
            leaf.validate_activation(object())

    def test_linear_weight_validator_rejects_stale_or_non_half_weights(self):
        leaf.validate_linear_weight(FakeTensor((6400, 5120)), "gate", (6400, 5120))
        leaf.validate_linear_weight(FakeTensor((5120, 3200)), "down", (5120, 3200))
        for shape in ((6912, 5120), (5120, 3456), (6400, 5000), (5120, 3200)):
            with self.assertRaises(leaf.LeafError):
                leaf.validate_linear_weight(FakeTensor(shape), "w", (6400, 5120))
        with self.assertRaises(leaf.LeafError):
            leaf.validate_linear_weight(FakeTensor((6400, 5120), "torch.float32"),
                                        "gate", (6400, 5120))


class LayerValidationTests(unittest.TestCase):
    def test_accepts_a_real_shaped_r79_block(self):
        report = leaf.validate_layer_geometry(make_layer())
        self.assertEqual(report["intermediate_size_per_partition"], 3200)
        self.assertEqual(report["class_names"]["norm"], "MiniMaxH3Qwen3VLRMSNorm")
        self.assertEqual(report["class_names"]["down_proj"],
                         "MiniMaxH3Qwen3VLRowParallelLinear")
        self.assertTrue(report["bias_free_and_unquantised"])
        self.assertEqual(report["variance_epsilon"], 1e-6)

    def test_rejects_every_structural_drift(self):
        bad = {
            "stale_ip": make_layer(ip=3456),
            "stale_mlp": make_layer(intermediate=25600, gate_up_shape=(6912, 5120)),
            "stale_down": make_layer(down_shape=(5120, 3456)),
            "bias": make_layer(bias=True),
            "quantised": make_layer(method_name="Fp8LinearMethod"),
            "float_weight": make_layer(weight_dtype="torch.float32"),
            "float_gamma": make_layer(norm_weight_dtype="torch.float32"),
            "bad_gamma_shape": make_layer(norm_weight_shape=(4096,)),
            "bad_eps": make_layer(eps=0.0),
            "wrong_layer_class": make_layer(layer_name="SomeOtherBlock"),
        }
        for name, layer in bad.items():
            with self.subTest(name=name), self.assertRaises(leaf.LeafError):
                leaf.validate_layer_geometry(layer)


class PinnedNormOrderTests(unittest.TestCase):
    def test_the_pinned_branch_is_byte_identical_and_two_round(self):
        pins = leaf.verify_pinned_norm_order(R79_CANDIDATE)
        self.assertEqual(pins["candidate_file_sha256"], leaf.PINNED_R79_CANDIDATE_SHA256)
        self.assertEqual(pins["fp16_two_round_branch_sha256"],
                         leaf.PINNED_R79_NORM_FP16_BRANCH_SHA256)
        self.assertEqual(hashlib.sha256(R79_CANDIDATE.read_bytes()).hexdigest(),
                         leaf.PINNED_R79_CANDIDATE_SHA256)

    def test_branch_rounds_normalised_value_before_the_half_gamma_product(self):
        branch = leaf.r79_norm_branch_source(R79_CANDIDATE)
        self.assertIn("xf.square().mean(-1, keepdim=True)", branch)
        self.assertIn("torch.rsqrt(variance + self.variance_epsilon)", branch)
        cast = branch.index(".to(input_dtype)")
        gamma = branch.index("self.weight * normalized")
        self.assertLess(cast, gamma, "the Half cast must precede the gamma product")
        # The one-round alternative (Float gamma inside the point math) is absent.
        self.assertNotIn("self.weight.float()", branch)
        self.assertNotIn("self.weight * xf", branch)

    def test_a_semantically_edited_branch_changes_the_digest(self):
        edited = R79_CANDIDATE.read_text().replace(
            "output = self.weight * normalized",
            "output = (normalized.float() * self.weight.float()).to(input_dtype)")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_encoder79.py"
            path.write_text(edited)
            self.assertNotEqual(leaf.r79_norm_branch_digest(path),
                                leaf.PINNED_R79_NORM_FP16_BRANCH_SHA256)
            with self.assertRaises(leaf.LeafError):
                leaf.verify_pinned_norm_order(path)

    def test_missing_branch_or_class_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.py"
            path.write_text("x = 1\n")
            with self.assertRaises(leaf.LeafError):
                leaf.r79_norm_branch_source(path)


class CpuOracleContractTests(unittest.TestCase):
    def test_rmsnorm_oracle_is_two_round(self):
        fn = next(n for n in TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == "cpu_half_rmsnorm_reference")
        text = ast.get_source_segment(SOURCE, fn)
        self.assertIn("xf.square().mean(-1, keepdim=True)", text)
        self.assertIn("torch.rsqrt(variance + float(eps))", text)
        self.assertIn(".to(torch.float16)", text)
        self.assertIn("weight_half * normalized", text)
        self.assertNotIn(".float() * ", text)   # no FP32 gamma multiply
        self.assertNotIn("weight_half.float()", text)

    def test_linear_oracle_uses_half_operands_only(self):
        fn = next(n for n in TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == "cpu_half_linear_reference")
        text = ast.get_source_segment(SOURCE, fn)
        self.assertIn("functional.linear(x_half, weight_half)", text)
        self.assertNotIn(".float()", text)
        self.assertNotIn("torch.float32", text)

    def test_oracles_fail_closed_on_non_half(self):
        fake = type("T", (), {"dtype": "torch.float32"})()
        with self.assertRaises(leaf.LeafError):
            leaf.cpu_half_rmsnorm_reference(fake, fake, 1e-6)
        with self.assertRaises(leaf.LeafError):
            leaf.cpu_half_linear_reference(fake, fake)


class ControlSourceInvariantTests(unittest.TestCase):
    def test_only_the_norm_and_local_mlp_leaves_are_touched(self):
        for forbidden in ("all_reduce", "_scaled_dot_product_attention",
                          "scaled_dot_product_attention", "monkeypatch",
                          "torch.nn.functional.linear =", "setattr("):
            self.assertNotIn(forbidden, SOURCE, f"{forbidden} must not appear")

    def test_down_projection_collective_is_explicitly_not_executed(self):
        fn = next(n for n in TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == "run_controls")
        text = ast.get_source_segment(SOURCE, fn)
        self.assertIn('"collective_executed": False', text)
        self.assertIn('"no_collective_executed": True', text)
        # The row-parallel module's own forward (which all-reduces) is never called.
        self.assertNotIn("mlp.down_proj(", text)
        self.assertNotIn("mlp(", text)

    def test_repeat_and_frozen_gates_raise_instead_of_tolerating(self):
        fn = next(n for n in TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == "run_controls")
        text = ast.get_source_segment(SOURCE, fn)
        self.assertGreaterEqual(text.count("not repeat-stable"), 5)
        self.assertIn("raw_bits_equal", text)
        self.assertIn("changed during the leaf controls", text)
        self.assertIn('"frozen_unchanged": True', text)

    def test_no_accuracy_pass_or_tolerance_is_declared(self):
        fn = next(n for n in TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == "run_controls")
        text = ast.get_source_segment(SOURCE, fn)
        self.assertIn('"numerical_tolerance": None', text)
        self.assertIn('"numerical_pass_field_absent": True', text)
        self.assertIn('"full_encoder_precision_accepted": False', text)
        self.assertIn('"whole_goal_complete": False', text)
        self.assertNotIn("numerical_pass': True", SOURCE)
        self.assertNotIn("precision_accepted': True", SOURCE)

    def test_no_float_weight_teacher_or_bf16_teacher_in_source(self):
        self.assertNotIn("bfloat16", SOURCE.replace("no_bfloat16_teacher", ""))
        self.assertIn('"no_fp32_weight_shadow": True', SOURCE)
        self.assertIn('"float_weight_teacher": False', SOURCE)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_self_consistent_and_non_claiming(self):
        m = leaf.manifest()
        self.assertEqual(m["round"], 86)
        self.assertFalse(m["executes_npu"])
        self.assertFalse(m["proves_full_precision"])
        self.assertFalse(m["imports_torch_at_module_scope"])
        self.assertEqual(m["sha256"], hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest())
        self.assertEqual(m["pins"]["round79_candidate_file_sha256"],
                         leaf.PINNED_R79_CANDIDATE_SHA256)
        self.assertIn("run_controls", m["entry_points"])

    def test_manifest_matches_the_checked_in_json(self):
        checked = HERE / "manifest86.json"
        if not checked.exists():
            self.skipTest("manifest86.json not written yet")
        import json
        on_disk = json.loads(checked.read_text())
        self.assertEqual(on_disk["sha256"], leaf.manifest()["sha256"])
        self.assertEqual(on_disk["geometry"], leaf.expected_geometry())


if __name__ == "__main__":
    unittest.main()