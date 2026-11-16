"""Portable scope and exact-operation contracts for vision-only factors."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent


def funcs(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


class VisionContracts(unittest.TestCase):
    def test_text_math_and_full_cpu_output_timer_match_verified_round70(self):
        old = funcs(ROOT.parent / 'round70' / 'complete_encode_tp8.py')
        new = funcs(ROOT / 'vision_encode_tp8.py')
        for name in ('row_forward', 'local', 'attention_forward', 'linear_fp32', 'mlp_forward',
                     'native_statistics_norm', 'prefix_statistics_norm', 'norm_dispatch70', 'measure_part'):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        before = next(node for node in ast.walk(ast.parse((ROOT.parent / 'round70' / 'complete_encode_tp8.py').read_text()))
                      if isinstance(node, ast.FunctionDef) and node.name == 'once_complete')
        after = next(node for node in ast.walk(ast.parse((ROOT / 'vision_encode_tp8.py').read_text()))
                     if isinstance(node, ast.FunctionDef) and node.name == 'once_complete')
        self.assertEqual(ast.dump(before), ast.dump(after))
    def test_only_vision_instance_is_wrapped_and_actual_module_budget_is_pinned(self):
        helper = (ROOT / 'vision_linear71.py').read_text()
        driver = (ROOT / 'vision_encode_tp8.py').read_text()
        self.assertIn('vision_modes.install(encoder.vision, candidate._NpuBF16CompatibleLinear)', driver)
        self.assertIn('module.forward = types.MethodType(wrapped, module)', helper)
        self.assertNotIn('compatible_linear_type.forward =', helper)
        self.assertNotIn('candidate._NpuBF16CompatibleLinear.forward =', driver)
        self.assertIn('len(modules) != 116', helper)
        self.assertIn('MAX_BYTES = 2362750400', helper)
        self.assertIn('MAX_ENTRIES = 232', helper)
    def test_same_three_parameter_linear_output_cast_and_no_packing_bias_split(self):
        tree = ast.parse((ROOT / 'vision_linear71.py').read_text())
        forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        linear = [node for node in ast.walk(forward) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'linear']
        self.assertEqual(len(linear), 1)
        self.assertEqual([ast.unparse(arg) for arg in linear[0].args], ['x32', 'w32', 'b32'])
        compute = next(node for node in forward.body if isinstance(node, ast.With))
        warm_nodes = [node for statement in compute.body if not isinstance(statement, ast.If) for node in ast.walk(statement)]
        # Verification legitimately views CPU bytes for raw readonly proof;
        # the GEMM dispatch must not reshape/pack any actual compute operand.
        calls = {node.func.attr for node in warm_nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse({'transpose', 'permute', 'view', 'reshape', 'copy_', 'add_', 'matmul'} & calls)
        self.assertIn('output = F.linear(x32, w32, b32).to(value.dtype)', ast.unparse(forward))
        self.assertIn('reference = original(value)', ast.unparse(forward))
    def test_actual_layout_and_full_weight_bias_output_identity_required(self):
        text = (ROOT / 'vision_linear71.py').read_text()
        for name in ('input_cast', 'weight_shadow', 'bias_shadow', 'output'):
            self.assertIn("'" + name + "': self.metric", text)
        for pair in ('input_cast_layout', 'weight_shadow_layout', 'bias_shadow_layout', 'output_layout'):
            self.assertIn(pair, text)
        self.assertIn("proof['raw_bits_equal'] and proof['finite']", text)
        self.assertIn("not row['input_readonly']", text)
        self.assertIn('expand_activation(value) if use_activation else value.float()', text)
    def test_fast_text_is_fixed_outside_vision_mode_loop_and_flags_unchanged(self):
        tree = ast.parse((ROOT / 'vision_encode_tp8.py').read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and ast.unparse(node.iter) == 'VisionLinearModes.MODES')
        assignments = {target.id for statement in ast.walk(loop) if isinstance(statement, ast.Assign) for target in statement.targets for target in ast.walk(target) if isinstance(target, ast.Name)}
        self.assertFalse({'NORM_TAIL', 'STATS_MODE', 'CAST_MODE', 'MODE', 'VERIFY_NORM70', 'VERIFY_CAST', 'VERIFY'} & assignments)
        text = (ROOT / 'vision_encode_tp8.py').read_text()
        self.assertIn("NORM_TAIL, STATS_MODE, CAST_MODE = True, 'add_square', 'both'", text)
        self.assertIn('assert r[\'math_flags_before\'] == r[\'math_flags_after\']', text)
        self.assertIn('PARTS_ACTIVE = vision_modes.verify = False', text)
    def test_mode_reset_releases_prior_shadow_and_restore_rebinds_original(self):
        tree = ast.parse((ROOT / 'vision_linear71.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        class Cache:
            def __init__(self, max_bytes, max_entries): self.max_bytes, self.max_entries = max_bytes, max_entries; self.entries = {}; self.cleared = False
            def clear(self): self.entries.clear(); self.cleared = True
        ns = {'ShadowWeights': Cache}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<actual vision mode class>', 'exec'), ns)
        modes = ns['VisionLinearModes'](metric=None)
        old = modes.shadow; old.entries['large_weight'] = object()
        modes.begin_mode('shadow_only')
        self.assertTrue(old.cleared); self.assertEqual(old.entries, {}); self.assertIsNot(modes.shadow, old)
        self.assertEqual(modes.shadow.max_entries, 232); self.assertEqual(modes.shadow.max_bytes, 2362750400)
        module = SimpleNamespace(forward='wrapped'); original = object(); modes.bound.append((module, original))
        modes.shadow.entries['bias'] = object()
        modes.restore()
        self.assertIs(module.forward, original); self.assertEqual(modes.bound, []); self.assertEqual(modes.shadow.entries, {})
        with self.assertRaises(ValueError): modes.begin_mode('unmeasured')


if __name__ == '__main__':
    unittest.main()
