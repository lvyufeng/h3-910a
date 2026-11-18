#!/usr/bin/env python3
"""Stdlib-only contracts for the Round89 vision Norm/GELU leaf module.

These tests are structural: they pin the real vision inventory, the original GELU
modes, the source it was read from, the control-matrix keys and the non-claims.
They deliberately contain no numerical result, no tolerance and no NPU import.
"""

import ast
import hashlib
import importlib.util
import inspect
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "vision_leaf89.py"
LEAF = None


def _load():
    global LEAF
    if LEAF is None:
        spec = importlib.util.spec_from_file_location("vision_leaf89_contract", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        LEAF = module
    return LEAF


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InventoryTests(unittest.TestCase):
    def test_seven_exact_leaf_names_and_kinds(self):
        m = _load()
        self.assertEqual(len(m.LEAF_MODULE_NAMES), 7)
        self.assertEqual(set(m.LEAF_MODULE_NAMES), set(m.LEAF_KINDS))
        self.assertEqual(set(m.LEAF_MODULE_NAMES), set(m.EXPECTED_LEAF_SHAPES))
        self.assertEqual(m.leaf_kind("blocks.0.norm1"), "layer_norm")
        self.assertEqual(m.leaf_kind("merger.act_fn"), "gelu")
        with self.assertRaises(m.VisionLeafError):
            m.leaf_kind("blocks.0.mlp.linear_fc1")

    def test_real_vision_counts_and_geometry(self):
        m = _load()
        self.assertEqual(m.EXPECTED_BLOCKS, 27)
        self.assertEqual(m.EXPECTED_MERGER_MODULES, 4)
        self.assertEqual(m.EXPECTED_NORM_MODULES, 58)   # 27*2 + 4
        self.assertEqual(m.EXPECTED_GELU_MODULES, 31)   # 27 + 4
        self.assertEqual(m.EXPECTED_NORMALIZED_SHAPE_COUNTS, {1152: 55, 4608: 3})
        self.assertEqual(m.EXPECTED_GELU_APPROXIMATE_COUNTS, {"tanh": 27, "none": 4})
        self.assertEqual(m.NORM_EPS, 1e-6)
        self.assertEqual(m.NORM_CLASS_NAME, "_NpuBF16CompatibleLayerNorm")

    def test_expected_shapes_are_the_reviewed_ones(self):
        m = _load()
        shapes = m.EXPECTED_LEAF_SHAPES
        self.assertEqual(shapes["blocks.0.norm1"], (8064, 1152))
        self.assertEqual(shapes["blocks.0.norm2"], (8064, 1152))
        self.assertEqual(shapes["blocks.0.mlp.act_fn"], (8064, 4304))
        self.assertEqual(shapes["merger.norm"], (8064, 1152))
        self.assertEqual(shapes["merger.act_fn"], (2016, 4608))
        self.assertEqual(shapes["deepstack_merger_list.0.norm"], (2016, 4608))
        self.assertEqual(shapes["deepstack_merger_list.0.act_fn"], (2016, 4608))
        plan = m.planned_leaf_plan()
        self.assertEqual([row["name"] for row in plan], list(m.LEAF_MODULE_NAMES))

    def test_original_gelu_modes_are_not_unified(self):
        m = _load()
        self.assertEqual(m.LEAF_GELU_APPROXIMATE, {
            "blocks.0.mlp.act_fn": "tanh",
            "deepstack_merger_list.0.act_fn": "none",
            "merger.act_fn": "none"})
        for name, approximate in m.LEAF_GELU_APPROXIMATE.items():
            self.assertEqual(m.leaf_kind(name), "gelu")
        self.assertEqual(m.planned_leaf_plan()[2]["gelu_approximate"], "tanh")


class SourcePinTests(unittest.TestCase):
    def test_pin_is_the_reviewed_r79_candidate(self):
        m = _load()
        self.assertEqual(m.PINNED_CANDIDATE_SHA256,
                         "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78")
        receipt = m.verify_candidate_source()
        self.assertTrue(receipt["matches_pin"])
        self.assertEqual(receipt["sha256"], m.PINNED_CANDIDATE_SHA256)

    def test_a_modified_candidate_fails_closed(self):
        m = _load()
        with self.assertRaises(m.VisionLeafError):
            m.verify_candidate_source(MODULE_PATH)
        with self.assertRaises(m.VisionLeafError):
            m.verify_candidate_source(HERE / "does_not_exist_89.py")


class ModuleScopeTests(unittest.TestCase):
    def test_only_stdlib_imports_and_no_torch_at_module_scope(self):
        m = _load()
        tree = ast.parse(MODULE_PATH.read_text())
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".")[0])
        self.assertLessEqual(top, {"__future__", "hashlib", "json", "pathlib"})
        self.assertNotIn("torch", top)
        self.assertNotIn("numpy", top)
        self.assertFalse(hasattr(m, "torch"))

    def test_torch_and_functional_are_lazy_inside_the_operators(self):
        tree = ast.parse(MODULE_PATH.read_text())
        module_level = [node for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(
            isinstance(node, ast.Import) and any(a.name.startswith("torch")
                                                 for a in node.names)
            for node in module_level))
        snip = MODULE_PATH.read_text()
        self.assertIn("import torch", snip)
        self.assertIn("import torch.nn.functional as F", snip)


class ControlContractTests(unittest.TestCase):
    def test_run_controls_signature_is_the_root_contract(self):
        m = _load()
        parameters = list(inspect.signature(m.run_controls).parameters)
        self.assertEqual(parameters[:4], ["vision", "inputs", "compare", "tensor_info"])
        self.assertEqual(parameters[4], "expected_shapes")

    def test_every_control_key_is_declared(self):
        m = _load()
        keys = m.expected_controls_report_keys()
        for required in ("native_vs_cpu_half", "npu_float_op_then_half_vs_cpu_half",
                         "cpu_float_op_then_half_vs_cpu_half",
                         "npu_float_op_vs_cpu_float_op",
                         "native_vs_npu_float_op_then_half",
                         "native_repeat_bit_identical", "cpu_half_repeat_bit_identical",
                         "input_sha_unchanged", "parameters_sha_unchanged",
                         "no_accuracy_tolerance"):
            self.assertIn(required, keys)
        self.assertEqual(len(keys), len(set(keys)))

    def test_native_reference_bypasses_instance_patches(self):
        source = MODULE_PATH.read_text()
        self.assertIn("return type(module).forward(module, value)", source)
        self.assertIn('"forward" in _module_for(norms, gelus, name).__dict__', source)

    def test_cpu_reference_uses_original_half_parameters(self):
        source = MODULE_PATH.read_text()
        self.assertIn('module.weight.detach().to("cpu").contiguous()', source)
        self.assertIn('module.bias.detach().to("cpu").contiguous()', source)
        self.assertIn("float(module.eps)", source)
        self.assertIn('approximate=str(getattr(module, "approximate", "none"))', source)

    def test_no_tolerance_and_no_teacher_is_declared(self):
        m = _load()
        source = MODULE_PATH.read_text()
        for banned in ("atol", "rtol", "allclose", "isclose"):
            self.assertNotIn(banned, source)
        self.assertIn('"no_accuracy_tolerance": True', source)
        self.assertIn('"numerical_tolerance": None', source)
        self.assertIn('"full_encoder_precision_accepted": False', source)
        self.assertIn('"whole_goal_complete": False', source)
        self.assertIn('"no_bf16_teacher": True', source)
        self.assertIn('"no_float_weight_teacher": True', source)
        self.assertNotIn("torch.bfloat16", source)
        self.assertNotIn(".bfloat16(", source)


class AdapterContractTests(unittest.TestCase):
    def test_adapter_api_and_guarded_restore(self):
        m = _load()
        for attribute in ("set_modes", "set_enabled", "snapshot", "uninstall"):
            self.assertTrue(callable(getattr(m.VisionNonlinearAdapter89, attribute)))
        source = MODULE_PATH.read_text()
        self.assertIn('getattr(current, "__self__", None) is module', source)
        self.assertIn('getattr(current, "__func__", None) is wrapper.__func__', source)
        self.assertIn("from types import MethodType", source)
        self.assertIn('setattr(module, "forward", wrapper)', source)
        self.assertIn('delattr(module, "forward")', source)
        self.assertIn('setattr(module, "forward", self._original[name])', source)
        self.assertIn("self._had_instance_forward", source)

    def test_adapter_is_per_instance_and_never_global(self):
        m = _load()
        source = MODULE_PATH.read_text()
        self.assertNotIn("setattr(torch", source)
        self.assertNotIn("torch.nn.functional.layer_norm =", source)
        self.assertNotIn("torch.nn.functional.gelu =", source)
        self.assertNotIn("self._float", source)
        self.assertIn('"patches_torch_global": False', source)
        self.assertIn('"patches_torch_functional": False', source)
        self.assertIn('"patches_biased_linears_or_conv": False', source)
        self.assertIn('"cached_float_weights": False', source)
        self.assertIn('"disabled_is_exact_native": True', source)
        snapshot_keys = m.adapter_snapshot_keys()
        for key in ("modes", "norm_modules_patched", "gelu_modules_patched",
                    "norm_eps_untouched", "strict_half_and_device_guard"):
            self.assertIn(key, snapshot_keys)

    def test_enabled_path_casts_once_and_keeps_eps(self):
        source = MODULE_PATH.read_text()
        self.assertEqual(source.count(".to(input_.dtype)"), 2)
        self.assertIn("F.layer_norm(input_.float(), tuple(module.normalized_shape)",
                      source)
        self.assertIn("float(module.eps)", source)
        self.assertIn("approximate=str(getattr(module, \"approximate\", \"none\"))", source)
        self.assertIn("input_.dtype is not torch.float16", source)


class ManifestTests(unittest.TestCase):
    def test_manifest_matches_sources_and_stays_non_claiming(self):
        import json
        m = _load()
        manifest_path = HERE / "manifest89.json"
        manifest = json.loads(manifest_path.read_text())
        for name, entry in manifest["files"].items():
            path = HERE / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(entry["bytes"], path.stat().st_size, name)
            self.assertEqual(entry["sha256"], _sha(path), name)
        self.assertFalse(manifest["compiled"])
        self.assertFalse(manifest["executed_on_npu"])
        self.assertFalse(manifest["proves_precision"])
        self.assertFalse(manifest["proves_speed"])
        self.assertFalse(manifest["claims_h3_improvement"])
        self.assertEqual(manifest["pin"]["candidate_sha256"], m.PINNED_CANDIDATE_SHA256)
        self.assertEqual(manifest["counts"], {
            "blocks": 27, "merger_modules": 4, "norm_modules": 58,
            "gelu_modules": 31, "gelu_tanh": 27, "gelu_erf": 4})


if __name__ == "__main__":
    unittest.main()