import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


class CastContracts(unittest.TestCase):
    def test_integer_expansion_has_no_arithmetic_rounding_or_zero_fix(self):
        tree = ast.parse((ROOT / 'activation_cast69.py').read_text())
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertFalse({'h_mul', 'vmuls', 'vconv', 'h_add', 'vec_reduce_add', 'vec_rsqrt', 'h_exp'} & attrs)
        names = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertEqual(names.count('unpack_bf16'), 1)
        self.assertNotIn('_round_bf16', names)
        self.assertNotIn('_fix_product_zero', names)
    def test_chunk_tail_gm_ownership_and_codec_padding(self):
        for elements in (128, 384, 2043 * 1024, 2043 * 3200, 2043 * 5120):
            tasks = (elements + 4095) // 4096
            ranges = [(task * 4096, min(4096, elements - task * 4096)) for task in range(tasks)]
            self.assertEqual(sum(count for _, count in ranges), elements)
            for offset, count in ranges:
                self.assertEqual(offset % 128, 0)
                self.assertEqual(count % 128, 0)
                padded = ((count + 255) // 256) * 256
                self.assertLessEqual(padded, 4096)
                self.assertLessEqual(offset + count, elements)
        text = (ROOT / 'activation_cast69.py').read_text()
        self.assertIn('t.data_move(ygm[offset], values, 0, 1, count // 8, 0, 0)', text)
        self.assertIn('x.data_ptr() % 32', text)
        self.assertIn('not x.is_contiguous()', text)
    def test_norm_causal_and_projection_math_remain_prior_exact_stack(self):
        before = functions(ROOT.parent / 'round68' / 'text_prefix_tp8.py')
        after = functions(ROOT / 'text_cast_tp8.py')
        for name in ('native_statistics_norm', 'prefix_statistics_norm', 'row_forward', 'local', 'linear_fp32',
                     'attention_forward', 'mlp_forward', 'once', 'metric', 'parameters_sha'):
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        text = (ROOT / 'text_cast_tp8.py').read_text()
        self.assertIn("MODE, NORM_TAIL, VERIFY_NORM, STATS_MODE, PREFIX_VERIFY = 'shared_qkv', True, False, 'add_square', False", text)
    def test_only_shared_and_fallback_conversion_preserving_shared_hits(self):
        tree = ast.parse((ROOT / 'text_cast_tp8.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MeasuredActivation')
        get = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'get')
        returns = [node for node in ast.walk(get) if isinstance(node, ast.Return)]
        self.assertTrue(any(isinstance(node.value, ast.Name) and node.value.id == 'converted' for node in returns))
        self.assertTrue(any(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == '_convert' for node in returns))
        scope = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'scope')
        tries = [node for node in ast.walk(scope) if isinstance(node, ast.Try)]
        self.assertEqual(len(tries), 1)
        self.assertEqual(ast.unparse(tries[0].finalbody[0]), 'self.active = None')


if __name__ == '__main__':
    unittest.main()
