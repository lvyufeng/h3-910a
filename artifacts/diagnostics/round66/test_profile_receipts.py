import hashlib
import json
from pathlib import Path
import unittest
from summarize import summarize

ROOT = Path(__file__).resolve().parent


class RetainedProfileEvidence(unittest.TestCase):
    def test_complete_text_identity_and_same_rank_representative_vectors(self):
        result = summarize()
        self.assertTrue(result['all_complete_outputs_raw_bits_equal'])
        self.assertTrue(result['all_inputs_and_weights_immutable'])
        self.assertFalse(result['clean_resident_under2s'])
        self.assertFalse(result['official_goal_complete'])
        forwards = 0
        for rank in range(8):
            row = json.loads((ROOT / 'ranks' / ('profile_rank%d.json' % rank)).read_text())
            for case in row['cases'].values():
                forwards += 1 + sum(len(case[phase]) for phase in ('clean_before', 'profiles', 'clean_after'))
        self.assertEqual(forwards, 192)
        for mode in ('baseline', 'shared_qkv'):
            for profile in result['cases'][mode]['profiles']:
                slow = profile['slowest_wall_rank']
                self.assertEqual(profile['representative'], profile['all_rank_vectors'][slow])
                self.assertEqual(profile['representative']['profile_wall_s'], max(vector['profile_wall_s'] for vector in profile['all_rank_vectors']))
                expected = 1103 if mode == 'shared_qkv' else 1053
                for vector in profile['all_rank_vectors']:
                    self.assertEqual(sum(value['count'] for value in vector['categories'].values()), expected)
                    self.assertIn('child.all_reduce', vector['categories'])
                    self.assertNotIn('top.all_reduce', vector['categories'])

    def test_compact_breakdown_is_from_actual_median_wall_same_rank_call(self):
        result = summarize()
        compact = json.loads((ROOT / 'breakdown.json').read_text())
        for mode, observed in compact['modes'].items():
            profiles = result['cases'][mode]['profiles']
            chosen = sorted(profiles, key=lambda profile: profile['representative']['profile_wall_s'])[1]
            self.assertEqual(observed['profile_call'], chosen['profile_call'])
            self.assertEqual(observed['slowest_wall_rank'], chosen['slowest_wall_rank'])
            self.assertEqual(observed['profile_wall_s'], chosen['representative']['profile_wall_s'])
            self.assertEqual(observed['top_nonoverlap_event_sum_s'], chosen['representative']['top_nonoverlap_event_sum_s'])
            self.assertAlmostEqual(sum(observed['top_groups_event_elapsed_s'].values()), observed['top_nonoverlap_event_sum_s'])

    def test_pinned_sources_and_final_cleanup(self):
        result = json.loads((ROOT / 'final_sources.json').read_text())
        expected = json.loads((ROOT / 'expected_sources.json').read_text())
        self.assertTrue(result['all_sources_match'])
        self.assertFalse(result['production_modified'])
        self.assertEqual(len(result['production']), 23)
        self.assertEqual(len(result['diagnostics']), 13)
        for category in ('production', 'diagnostics'):
            self.assertEqual(set(result[category]), set(expected[category]))
            self.assertTrue(all(value['matches'] and value['sha256'] == expected[category][name] for name, value in result[category].items()))
        cleanup = json.loads((ROOT / 'cleanup_receipt.json').read_text())
        self.assertTrue(cleanup['lease_released'])
        self.assertEqual(cleanup['accelerator_processes'], 0)
        self.assertEqual(cleanup['accelerator_utilization_percent'], [0] * 8)


if __name__ == '__main__':
    unittest.main()
