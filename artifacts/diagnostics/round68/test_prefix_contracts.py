"""Portable prefix access, native reduction ordering and timer/dispatch contracts."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


class PrefixContracts(unittest.TestCase):
    def test_no_reduction_rsqrt_or_bf16_square_rounding(self):
        tree = ast.parse((ROOT / 'norm_prefix68.py').read_text())
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertFalse({'vec_reduce_add', 'vec_rsqrt', 'h_rec', 'h_exp'} & attrs)
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertEqual(calls.count('_round_bf16'), 1)
        self.assertNotIn('_fix_product_zero', calls)
        build = functions(ROOT / 'norm_prefix68.py')['build']
        mul = [node for node in ast.walk(build) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'h_mul']
        self.assertEqual(len(mul), 1)
        self.assertEqual([node.id for node in mul[0].args], ['square', 'x', 'x'])
    def test_real_gm_widths_tail_padding_and_current_device(self):
        text = (ROOT / 'norm_prefix68.py').read_text()
        self.assertIn('t.data_move(sgm[row, 0], square, 0, 1, width // 8, 0, 0)', text)
        self.assertIn('t.data_move(ugm[row, 0], words, 0, 1, width // 16, 0, 0)', text)
        self.assertIn('t.vector_dup(128, words, 0, padded // 128, 1, 8)', text)
        self.assertIn('value.data_ptr() % 32', text)
        self.assertIn('torch.npu.current_stream().npu_stream', text)
        for rows in (17, 2043, 16344):
            cores = min(rows, 32)
            visited = [core + step * cores for core in range(cores) for step in range((rows + cores - 1) // cores) if core + step * cores < rows]
            self.assertEqual(sorted(visited), list(range(rows)))
            self.assertEqual(len(visited), len(set(visited)))
    def test_current_tail_and_native_mean_rsqrt_are_preserved(self):
        actual = functions(ROOT / 'text_prefix_tp8.py')
        prior = functions(ROOT.parent / 'round67' / 'text_norm_tp8.py')
        self.assertEqual(ast.dump(actual['native_statistics_norm']), ast.dump(prior['native_statistics_norm']))
        node = actual['prefix_statistics_norm']
        with_node = next(statement for statement in node.body if isinstance(statement, ast.With))
        variance = next(statement for statement in with_node.body if isinstance(statement, ast.Assign) and statement.targets[0].id == 'variance')
        inverse = next(statement for statement in with_node.body if isinstance(statement, ast.Assign) and statement.targets[0].id == 'inverse')
        self.assertEqual(ast.unparse(variance.value), 'squares.mean(-1, keepdim=True)')
        old_with = next(statement for statement in prior['native_statistics_norm'].body if isinstance(statement, ast.With))
        self.assertEqual(ast.dump(inverse), ast.dump(old_with.body[3]))
        self.assertEqual(ast.unparse(with_node.body[-1].value), 'norm_tail(updated, self.weight, inverse)')
    def test_non_norm_math_and_clean_timer_same_as_exact_prior_stack(self):
        before = functions(ROOT.parent / 'round67' / 'text_norm_tp8.py')
        after = functions(ROOT / 'text_prefix_tp8.py')
        for name in ('once', 'row_forward', 'local', 'linear_fp32', 'attention_forward', 'mlp_forward', 'parameters_sha', 'metric'):
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
    def test_profile_binding_does_not_overwrite_native_norm_or_include_checks(self):
        tree = ast.parse((ROOT / 'text_prefix_tp8.py').read_text())
        names = [statement.targets[0].id for statement in tree.body if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)]
        self.assertEqual(names.count('old_norm'), 1)
        self.assertEqual(names.count('profile_norm_impl'), 1)
        profile = functions(ROOT / 'text_prefix_tp8.py')['profile_norm']
        self.assertEqual(profile.body[-1].value.args[1].id, 'profile_norm_impl')
        text = (ROOT / 'text_prefix_tp8.py').read_text()
        self.assertIn('PREFIX_VERIFY = False\n        if baseline is None:', text)
        self.assertIn("MODE, VERIFY, NORM_TAIL, VERIFY_NORM = 'shared_qkv', False, True, False", text)


if __name__ == '__main__':
    unittest.main()
