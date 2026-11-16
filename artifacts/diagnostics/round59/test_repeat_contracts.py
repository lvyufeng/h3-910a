#!/usr/bin/env python3
"""Structural regression tests for repeat audit and measured-width padding."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'round56' / 'candidate_text_linear_encoder.py'


class RepeatContracts(unittest.TestCase):
    def test_candidate_changes_only_local_linear(self):
        a = ast.parse(BASE.read_text())
        b = ast.parse((ROOT / 'candidate_padded_text_encoder.py').read_text())
        def without_changed_fn(tree):
            tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != '_text_fp32_local_linear']
            return ast.dump(tree, include_attributes=False)
        self.assertEqual(without_changed_fn(a), without_changed_fn(b))

    def test_padding_is_measured_shape_default_off(self):
        candidate = (ROOT / 'candidate_padded_text_encoder.py').read_text()
        helper = (ROOT / 'padded_fp32_projection59.py').read_text()
        self.assertIn('get("VLLM_OMNI_910A_TEXT_PAD_KV", "0")', candidate)
        self.assertIn('x.shape == (1, 2043, 5120)', helper)
        self.assertIn('w.shape == (128, 5120)', helper)
        self.assertIn('not torch.is_grad_enabled()', helper)
        self.assertIn('out[..., :128].contiguous().to(x.dtype)', helper)
        self.assertNotIn('.half()', helper)

    def test_cached_projection_keeps_source_and_version(self):
        helper = (ROOT / 'padded_fp32_projection59.py').read_text()
        self.assertIn('version = w._version', helper)
        self.assertIn('value = (w, F.pad(', helper)
        self.assertIn('_MAX_ENTRIES = 128', helper)
        self.assertIn('w.device', helper)
        self.assertIn('w.data_ptr()', helper)

    def test_padding_does_not_change_valid_matrix_columns(self):
        x = [[.5, -2., 3.], [2., 0., -1.]]
        weight = [[1., 2., 3.], [-2., .125, 4.]]
        extended = weight + [[0.] * 3] * 6
        def linear(w):
            return [[sum(a*b for a, b in zip(row, out)) for out in w] for row in x]
        original = linear(weight)
        larger = [row[:2] for row in linear(extended)]
        self.assertEqual(original, larger)

    def test_observation_hooks_never_replace_outputs(self):
        tree = ast.parse((ROOT / 'repeat_audit_tp8.py').read_text())
        hooks = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.endswith('_hook')]
        self.assertEqual(len(hooks), 3)
        for hook in hooks:
            returns = [n for n in ast.walk(hook) if isinstance(n, ast.Return)]
            self.assertTrue(returns)
            self.assertTrue(all(isinstance(n.value, ast.Constant) and n.value.value is None for n in returns))

    def test_parameter_and_input_hashes_cover_actual_data(self):
        text = (ROOT / 'repeat_audit_tp8.py').read_text()
        self.assertIn('for name, p in model.named_parameters():', text)
        self.assertIn("r['parameters_before'] == r['parameters_after']", text)
        self.assertIn('after == before', text)
        self.assertIn('before == source_hash', text)
        self.assertIn('fresh_hook0', text)
        self.assertIn('memoryview(x.view(torch.uint8).numpy())', text)

    def test_projection_probe_tests_rejected_alternatives(self):
        text = (ROOT / 'projection_repeat.py').read_text()
        for word in ('deterministic_resident_linear', 'explicit_resident_mm', 'padded_rows_2048', 'resident_fp32_linear'):
            self.assertIn(word, text)
        self.assertIn('for _ in range(20):', text)
        self.assertIn('torch.use_deterministic_algorithms(original)', text)

    def test_full_padding_test_repeats_and_excludes_checks(self):
        text = (ROOT / 'text_padded_vectors_tp8.py').read_text()
        self.assertIn('VERIFY_ACT = False', text)
        self.assertIn("'third_vs_repeat': metric(third_cpu, repeat_cpu)", text)
        self.assertIn('assert len(ACT_CHECKS) == 50', text)
        self.assertIn("'vs_padded_eager_first'", text)
        self.assertIn('cache_stats()', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
