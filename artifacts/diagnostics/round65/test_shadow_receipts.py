"""Regress real eight-rank output identity, reporting, dependency and guard gates."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from summarize import aggregate, summarize

ROOT = Path(__file__).resolve().parent


def rows():
    return [json.loads((ROOT / 'ranks' / ('shadow_rank%d.json' % rank)).read_text()) for rank in range(8)]


class TextShadowEvidence(unittest.TestCase):
    def test_all_modes_full_output_and_identity_scope(self):
        result = summarize()
        self.assertFalse(result['strict_goal_complete'])
        self.assertFalse(result['whole_service_text_under2s_proven'])
        self.assertFalse(result['production_deployed'])
        for mode, case in result['cases'].items():
            self.assertTrue(case['full_output_raw_bits_equal'])
            self.assertEqual(case['full_output_elements_per_rank'], 10460160)
            self.assertFalse(case['all_clean_resident_calls_under2s'])
            self.assertEqual(len(case['warm_per_call_rank_max_s']), 3)
        self.assertEqual(result['cases']['activation_only']['same_input_projection_checks'], 1600)
        for mode in ('weight_shadow', 'shared_qkv'):
            self.assertEqual(result['cases'][mode]['same_input_projection_checks'], 2400)
            self.assertEqual(result['cases'][mode]['nonkv_shadow_bytes_per_rank'], [11927552000] * 8)
        self.assertEqual(result['cases']['restored']['nonkv_shadow_entries_per_rank'], [200] * 8)

    def test_rank_max_before_median_and_first_call_excluded(self):
        data = rows()
        for row in data:
            case = row['cases']['baseline']
            case['first_s'] = 999.
            for call in case['warm_calls']:
                call['seconds'] = 1.
        for i in range(3):
            data[i]['cases']['baseline']['warm_calls'][i]['seconds'] = 10.
        case = aggregate(data)['cases']['baseline']
        self.assertEqual(case['median_rank_max_s'], 10.)
        self.assertEqual(case['first_forward_rank_max_s_checks_and_cache_excluded'], 999.)

    def test_failure_wrong_rank_changed_state_output_or_projection_rejected(self):
        mutations = [lambda r: r[0].update(failure={'type': 'RuntimeError'}),
                     lambda r: r[0].update(rank=1),
                     lambda r: r[0].update(parameters_after='mutated'),
                     lambda r: r[0]['cases']['shared_qkv']['warm_calls'][0]['vs_baseline'].update(raw_bits_equal=False),
                     lambda r: r[0]['cases']['shared_qkv']['projection_checks'][0].update(raw_bits_equal=False),
                     lambda r: r[0]['cases']['shared_qkv']['projection_checks'].pop(),
                     lambda r: r[0]['cases']['shared_qkv']['shadow_cache'].update(entries=199),
                     lambda r: r[0]['cases']['shared_qkv']['shared_activation'].update(scope_closed=False)]
        for mutation in mutations:
            data = rows()
            mutation(data)
            with self.assertRaises(ValueError):
                aggregate(data)

    def test_actual_npu_lifecycle_controls_and_data_write_limitation(self):
        row = json.loads((ROOT / 'cache_guard_npu.json').read_text())
        self.assertTrue(row['completed'])
        self.assertNotIn('failure', row)
        self.assertEqual(len(row['checks']), 13)
        self.assertTrue(all(check['passed'] for check in row['checks']))
        self.assertFalse(row['data_write_version_changed'])
        self.assertEqual(row['source_sha256'], hashlib.sha256((ROOT / 'cache_guard_npu.py').read_bytes()).hexdigest())

    def test_all_old_dependencies_and_production_hashes_stay_pinned(self):
        row = json.loads((ROOT / 'final_sources.json').read_text())
        expected = json.loads((ROOT / 'expected_sources.json').read_text())
        self.assertFalse(row['production_modified'])
        self.assertTrue(row['all_sources_match'])
        self.assertEqual(len(row['production']), 23)
        self.assertEqual(len(row['diagnostics']), 12)
        for category in ('production', 'diagnostics'):
            self.assertEqual(set(row[category]), set(expected[category]))
            self.assertTrue(all(value['matches'] and value['sha256'] == expected[category][name]
                                for name, value in row[category].items()))


if __name__ == '__main__':
    unittest.main()
