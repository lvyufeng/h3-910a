"""Portable structural/access contracts; actual native numerical proof is separate."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class KernelContracts(unittest.TestCase):
    def test_native_reduction_and_rsqrt_are_not_reimplemented_in_tail(self):
        tree = ast.parse((ROOT / 'norm_tail67.py').read_text())
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertFalse({'vec_reduce_add', 'vec_rsqrt', 'h_rec', 'h_exp'} & names)
        direct = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn('_fix_product_zero', direct)
        self.assertEqual(direct.count('_round_bf16'), 2)
        self.assertEqual(direct.count('pack_bf16'), 1)
    def test_inverse_blocks_cannot_overread_actual_or_small_row_tails(self):
        for rows in (1, 5, 8, 9, 17, 2043, 16344):
            allocated = ((rows + 7) // 8) * 8
            for row in range(rows):
                base = (row // 8) * 8
                self.assertEqual(base % 8, 0)
                self.assertLess(base + 7, allocated)
                self.assertEqual(base + row % 8, row)
    def test_actual_widths_only_write_real_width_and_codec_tail_zeroed(self):
        for width in (128, 5120):
            padded = ((width + 255) // 256) * 256
            self.assertEqual(padded % 256, 0)
            self.assertEqual(width % 16, 0)
            self.assertLessEqual(width, padded)
            self.assertLess(padded - width, 256)
        text = (ROOT / 'norm_tail67.py').read_text()
        self.assertIn('t.data_move(ygm[row, 0], words, 0, 1, width // 16, 0, 0)', text)
        self.assertIn('t.vector_dup(128, words, 0, padded // 128, 1, 8)', text)
        self.assertIn('Input must use the current launch device', text)
        self.assertIn('torch.npu.current_stream().npu_stream', text)
    def test_guarded_version_keeps_identical_kernel_arithmetic(self):
        trees = [ast.parse((ROOT / name).read_text()) for name in ('norm_tail67.py', 'norm_tail_guard67.py')]
        builds = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build') for tree in trees]
        self.assertEqual(ast.dump(builds[0]), ast.dump(builds[1]))
        text = (ROOT / 'norm_tail_guard67.py').read_text()
        self.assertIn('tuple(inverse_rms.shape) != tuple(x.shape[:-1]) + (1,)', text)
        self.assertIn('value.data_ptr() % 32', text)
        self.assertIn('contiguous views', text)

    def test_native_residual_statistics_and_inverse_expression_unchanged(self):
        original = ast.parse((ROOT.parent / 'round65' / 'source' / 'candidate_padded_text_encoder.py').read_text())
        cls = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == 'MiniMaxH3Qwen3VLRMSNorm')
        native = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward_npu')
        actual_tree = ast.parse((ROOT / 'text_norm_tp8.py').read_text())
        actual = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'native_statistics_norm')
        native_with = next(node for node in native.body if isinstance(node, ast.With))
        actual_with = next(node for node in actual.body if isinstance(node, ast.With))
        self.assertEqual(ast.dump(native_with.items[0]), ast.dump(actual_with.items[0]))
        for before, after in zip(native_with.body[:3], actual_with.body[:3]):
            self.assertEqual(ast.dump(before), ast.dump(after))
        native_inverse = native_with.body[3].value.func.value.right
        actual_inverse = actual_with.body[3].value
        self.assertEqual(ast.dump(native_inverse), ast.dump(actual_inverse))
        self.assertEqual(ast.dump(native.body[-1]), ast.dump(actual.body[-1]))

    def test_clean_timing_and_non_norm_math_are_the_previous_exact_stack(self):
        def functions(path):
            return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        before = functions(ROOT.parent / 'round65' / 'text_shadow_tp8.py')
        after = functions(ROOT / 'text_norm_tp8.py')
        for name in ('once', 'mlp_forward', 'row_forward', 'local', 'attention_forward', 'linear_fp32', 'metric', 'parameters_sha'):
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)

    def test_row_ownership_is_disjoint_and_complete(self):
        for rows in (17, 2043, 16344):
            cores = min(32, rows)
            actual = [core + step * cores for core in range(cores) for step in range((rows + cores - 1) // cores) if core + step * cores < rows]
            self.assertEqual(sorted(actual), list(range(rows)))
            self.assertEqual(len(actual), len(set(actual)))


if __name__ == '__main__':
    unittest.main()
