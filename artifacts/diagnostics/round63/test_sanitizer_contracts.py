"""Portable finite-select reporting, AST integration and task-coverage checks."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent


def extract_function(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<actual extracted function>', 'exec'), namespace)
    return namespace[name]


class SanitizerContracts(unittest.TestCase):
    def test_exhaustive_proof_keeps_native_nonfinite_discrepancy_visible(self):
        row = json.loads((ROOT / 'sanitize_validation02.json').read_text())
        self.assertTrue(row['completed'])
        self.assertEqual(row['exhaustive']['all_half_bit_patterns'], 65536)
        self.assertEqual(row['exhaustive']['nonfinite_words'], 2048)
        self.assertTrue(row['exhaustive']['vs_integer_bit_oracle']['raw_bits_equal'])
        self.assertTrue(row['exhaustive']['finite_words_vs_native']['raw_bits_equal'])
        self.assertEqual(row['exhaustive']['finite_words_vs_native']['elements'], 63488)
        self.assertFalse(row['exhaustive']['vs_native_npu']['raw_bits_equal'])
        self.assertTrue(row['exhaustive']['native_output_vs_unfiltered_words']['raw_bits_equal'])
        self.assertEqual(row['actual_raw_output_nonfinite_count_cpu'], 0)
        self.assertFalse(row['production_installed'])
        for name, digest in row['sources'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        for case in row['cases'].values():
            self.assertTrue(case['vs_actual']['raw_bits_equal'])
            self.assertTrue(case['timing']['first_vs_repeat']['raw_bits_equal'])

    def test_failed_reference_assertion_is_not_hidden(self):
        row = json.loads((ROOT / 'sanitize_validation01.json').read_text())
        self.assertEqual(row['cases'], {})
        self.assertEqual(row['failure']['type'], 'AssertionError')
        self.assertTrue(row['exhaustive']['vs_integer_bit_oracle']['raw_bits_equal'])
        self.assertFalse(row['exhaustive']['vs_native_npu']['raw_bits_equal'])
        self.assertNotIn('completed', row)

    def test_actual_remote_attention_body_changes_only_checked_return(self):
        replace = extract_function('sanitize_integration63.py', 'replace_final_return', {'ast': ast, 'copy': copy})
        text = (ROOT / 'source' / 'flash.py').read_text()
        original = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == '_sdpa_attention')
        patched = replace(ast.get_source_segment(text, original))
        before = ast.parse(ast.get_source_segment(text, original))
        self.assertEqual(ast.dump(before.body[0].body[-1].value.func), ast.dump(ast.parse('torch.where', mode='eval').body))
        self.assertEqual(patched.body[0].body[-1].value.func.id, '_h3_sanitize63')
        before.body[0].body.pop()
        patched.body[0].body.pop()
        self.assertEqual(ast.dump(before), ast.dump(patched))

    def test_unexpected_remote_function_is_rejected(self):
        replace = extract_function('sanitize_integration63.py', 'replace_final_return', {'ast': ast, 'copy': copy})
        for text in ('def f(out):\n return out',
                     'def f(out):\n if out: return out\n return out',
                     'def f(out):\n return torch.nan_to_num(out)'):
            with self.assertRaises(ValueError):
                replace(text)

    def test_runtime_dispatch_only_uses_supported_half_shapes(self):
        marker = object()
        torch = SimpleNamespace(float16=marker, isfinite=lambda x: 'mask', zeros=lambda *a, **k: 0,
                                where=lambda *a: 'native')
        namespace = {'torch': torch, 'sanitize': lambda x: 'tbe'}
        fn = extract_function('sanitize_integration63.py', 'finite_select', namespace)
        for device, dtype, elements, expected in [('npu', marker, 128, 'tbe'), ('npu', marker, 0, 'native'),
                                                   ('npu', marker, 129, 'native'), ('cpu', marker, 128, 'native'),
                                                   ('npu', 'float32', 128, 'native')]:
            x = SimpleNamespace(device=SimpleNamespace(type=device), dtype=dtype, numel=lambda: elements)
            self.assertEqual(fn(x), expected)

    def test_complete_video_all_ranks_cases_and_restoration(self):
        from summarize import aggregate_decode, source_matches
        paths = sorted((ROOT / 'decode_sanitize').glob('decode_sanitize01_rank*.json'))
        self.assertEqual(len(paths), 8)
        rows = [json.loads(path.read_text()) for path in paths]
        actual_latent = json.loads((ROOT / 'recovered' / 'video_latent.json').read_text())
        for row in rows:
            self.assertTrue(source_matches(row, ROOT / 'decode_sanitize_tp8.py'))
            self.assertTrue(row['isolated_patch']['only_final_return_changed'])
            self.assertFalse(row['isolated_patch']['attention_algorithm_changed'])
            self.assertTrue(row['original_function_restored'])
            self.assertFalse(row.get('cleanup_errors'))
            self.assertEqual(row['latent']['sha256'], actual_latent['sha256'])
        result = aggregate_decode(rows, modes=('baseline_c128', 'sanitize_c128', 'sanitize_c256', 'baseline_restored'))
        for case in result['cases'].values():
            self.assertTrue(case['accepted_for_this_captured_latent'])
            self.assertTrue(case['all_repeats_exact'])
        self.assertTrue(result['cases']['sanitize_c128']['warm_calls_all_under_20s_resident_only'])
        self.assertTrue(result['cases']['sanitize_c256']['warm_calls_all_under_20s_resident_only'])

    def test_production_and_executed_sources_verified_unchanged(self):
        row = json.loads((ROOT / 'final_source_verification.json').read_text())
        expected = json.loads((ROOT / 'expected_sources.json').read_text())
        self.assertTrue(row['all_production_unchanged'])
        self.assertTrue(row['all_diagnostic_sources_match'])
        self.assertFalse(row['production_fast_path_installed'])
        self.assertEqual(len(row['production']), 18)
        for category in ('production', 'diagnostic_sources'):
            self.assertEqual(set(row[category]), set(expected[category]))
            for name, proof in row[category].items():
                self.assertTrue(proof['unchanged'])
                self.assertEqual(proof['sha256'], expected[category][name])

    def test_task_intervals_cover_real_and_tail_lengths_exactly(self):
        # Match the builder's explicit guarded full/tail chunk scheme, not a
        # hardware arithmetic emulator. Every interval is 32-byte DMA aligned.
        for elements in (128, 65536, 3680256, 2048 * 1541, 4096):
            chunk, cores = 4096, 30
            tasks = (elements + chunk - 1) // chunk
            cores = min(cores, tasks)
            intervals = []
            for core in range(cores):
                for step in range((tasks + cores - 1) // cores):
                    task = core + step * cores
                    if task < tasks:
                        start = task * chunk
                        end = min(start + chunk, elements)
                        self.assertEqual((end - start) % 128, 0)
                        intervals.append((start, end))
            intervals.sort()
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], elements)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])))


if __name__ == '__main__':
    unittest.main()
