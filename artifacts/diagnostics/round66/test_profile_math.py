import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def functions(path):
    tree = ast.parse(path.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class MathUnchanged(unittest.TestCase):
    def test_projection_attention_mlp_norm_input_and_timer_body_unchanged(self):
        original = functions(ROOT.parent / 'round65' / 'text_shadow_tp8.py')
        actual = functions(ROOT / 'text_profile_tp8.py')
        for name in ('lookup', 'linear_fp32', 'local', 'row_forward', 'attention_forward',
                     'mlp_forward', 'parameters_sha', 'once', 'metric', 'raw_hash'):
            self.assertEqual(ast.dump(actual[name]), ast.dump(original[name]), name)

    def test_profile_wrappers_only_delegate_to_original_math(self):
        actual = functions(ROOT / 'text_profile_tp8.py')
        for name, original in (('profile_local', 'local'), ('profile_row', 'row_forward'),
                               ('profile_norm', 'old_norm'), ('profile_deepstack', 'old_deepstack'),
                               ('profile_linear', 'old_linear')):
            node = actual[name]
            calls = [call for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'measured']
            self.assertEqual(len(calls), 1)
            self.assertIsInstance(calls[0].args[1], ast.Name)
            self.assertEqual(calls[0].args[1].id, original)


if __name__ == '__main__':
    unittest.main()
