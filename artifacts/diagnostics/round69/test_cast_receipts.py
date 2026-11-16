import copy
import hashlib
import json
from pathlib import Path
import unittest
from summarize import aggregate, summarize

ROOT = Path(__file__).resolve().parent


def rows():
    return [json.loads((ROOT / 'ranks' / ('cast_rank%d.json' % rank)).read_text()) for rank in range(8)]


class ActualCastEvidence(unittest.TestCase):
    def test_resident_threshold_and_complete_identity_scope(self):
        r = summarize()
        self.assertEqual(r['same_input_cast_checks'], 4800)
        self.assertEqual(r['same_input_projection_checks'], 7200)
        self.assertEqual(sum(case['complete_outputs_checked'] for case in r['cases'].values()), 160)
        self.assertTrue(r['cases']['both']['all_clean_resident_calls_under2s'])
        self.assertTrue(r['cases']['fallback_only']['all_clean_resident_calls_under2s'])
        self.assertFalse(r['cases']['shared_only']['all_clean_resident_calls_under2s'])
        self.assertFalse(r['whole_service_text_under2s_proven'])
        self.assertFalse(r['production_deployed'])
        self.assertFalse(r['official_goal_complete'])
    def test_all_word_native_and_integer_oracle_not_nan_numeric_equality(self):
        r = summarize()
        self.assertTrue(r['all65536_bf16_words_native_and_integer_oracle_equal'])
        v = json.loads((ROOT / 'validation01.json').read_text())
        proof = v['all_bf16_words']
        self.assertEqual(proof['elements'], 65536)
        self.assertTrue(proof['matches_bit_expansion_oracle'])
        self.assertTrue(proof['matches_native_cast'])
        self.assertEqual(proof['native_layout'], proof['candidate_layout'])
        self.assertTrue(v['contiguous_unaligned_view_rejected'])
    def test_wrong_cast_bits_shape_layout_input_or_projected_partial_rejected(self):
        changes = [lambda r: r[0].update(completed=False), lambda r: r[0].update(rank=1),
                   lambda r: r[0].update(parameters_after='mutated'),
                   lambda r: r[0]['cases']['both']['cast_checks'][0].update(input_readonly=False),
                   lambda r: r[0]['cases']['both']['cast_checks'][0]['proof'].update(raw_bits_equal=False),
                   lambda r: r[0]['cases']['both']['cast_checks'][0].update(input_shape=[1, 2043, 3200]),
                   lambda r: r[0]['cases']['both']['cast_checks'][0]['candidate_layout'].__setitem__(3, 29),
                   lambda r: r[0]['cases']['both']['projection_checks'][0].update(elements=1),
                   lambda r: r[0]['cases']['both']['projection_checks'][0].update(slot=['q', 49]),
                   lambda r: r[0]['cases']['both']['projection_checks'].pop(),
                   lambda r: r[0]['cases']['both']['warm_calls'][0]['vs_baseline'].update(max_abs_error=1.),
                   lambda r: r[0]['cases']['both'].update(activation_scope_closed=False)]
        for change in changes:
            data = rows()
            change(data)
            with self.assertRaises(ValueError):
                aggregate(data)
    def test_same_call_rank_max_precedes_median_and_excludes_first(self):
        data = rows()
        for rank in data:
            case = rank['cases']['both']
            case['first_s_excluded_checks_compilation'] = 1000.
            for call in case['warm_calls']:
                call['seconds'] = 1.
        for i in range(3):
            data[i]['cases']['both']['warm_calls'][i]['seconds'] = 10.
        case = aggregate(data)['cases']['both']
        self.assertEqual(case['median_rank_max_s'], 10.)
        self.assertEqual(case['first_rank_max_s_excluded_checks_compilation'], 1000.)
    def test_source_fingerprints_exact_and_npus_released(self):
        proof = json.loads((ROOT / 'final_sources.json').read_text())
        expected = json.loads((ROOT / 'expected_sources.json').read_text())
        self.assertTrue(proof['all_sources_match'])
        self.assertFalse(proof['production_modified'])
        self.assertEqual(len(proof['production']), 23)
        self.assertEqual(len(proof['diagnostics']), 14)
        for category in ('production', 'diagnostics'):
            self.assertEqual(set(proof[category]), set(expected[category]))
            self.assertTrue(all(value['matches'] and value['sha256'] == expected[category][name] for name, value in proof[category].items()))
        cleanup = json.loads((ROOT / 'cleanup_receipt.json').read_text())
        self.assertTrue(cleanup['lease_released'])
        self.assertEqual(cleanup['accelerator_processes'], 0)
        self.assertEqual(cleanup['accelerator_utilization_percent'], [0] * 8)


if __name__ == '__main__':
    unittest.main()
