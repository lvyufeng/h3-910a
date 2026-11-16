"""Vision event-profile scope and actual receipt regression contracts."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest
from summarize_profile71 import aggregate_profile, event_vector

ROOT = Path(__file__).resolve().parent


class ProfileSource(unittest.TestCase):
    def test_preallocated_profile_only_delegates_and_does_not_change_math(self):
        text = (ROOT / 'vision_events71.py').read_text()
        self.assertIn('EventProfile(torch, capacity=512)', text)
        self.assertIn('return self.profile.call(category, original, *args, **kwargs)', text)
        self.assertIn("bind(candidate, '_vision_sdpa_bf16_compatible', 'attention')", text)
        for forbidden in ('synchronize(', 'scaled_dot_product_attention(', 'F.linear(', 'matmul(', 'float(', 'contiguous('):
            self.assertNotIn(forbidden, text)
        self.assertIn('reversed(self.bindings)', text)
    def test_event_begin_before_timer_finish_and_hash_after_complete_timer(self):
        tree = ast.parse((ROOT / 'vision_profile_tp8.py').read_text())
        timer = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'once_complete')
        body = ast.unparse(timer)
        self.assertLess(body.index('vision_events.begin()'), body.index('t = time.perf_counter()'))
        self.assertLess(body.index('output = encoder.encode_ids('), body.index('seconds = time.perf_counter() - t'))
        self.assertLess(body.index('seconds = time.perf_counter() - t'), body.index('vision_events.finish()'))
        self.assertLess(body.index('vision_events.finish()'), body.index('tensor_identity(CALL_OBJECTS)'))
        self.assertIn('finally:\n        CALL_OBJECTS.clear()', body)


class ActualVisionEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = [ROOT / 'profile' / ('vision_rank%d.json' % rank) for rank in range(8)]
        if not all(path.is_file() for path in paths):
            raise unittest.SkipTest('Actual eight-rank independent vision event-profile receipts are not yet collected')
        cls.rows = [json.loads(path.read_text()) for path in paths]
        cls.main = [json.loads((ROOT / 'ranks' / ('vision_rank%d.json' % rank)).read_text()) for rank in range(8)]
    def test_all_full_outputs_match_and_no_full_service_or_quality_acceptance(self):
        report = aggregate_profile(self.rows, self.main)
        self.assertEqual(report['complete_outputs_compared'], 240)
        self.assertEqual(report['profile_events_verified'], 20880)
        self.assertTrue(report['complete_output_identity_accepted'])
        self.assertFalse(report['complete_encode_ids_under2s_accepted'])
        self.assertFalse(report['whole_service_text_under2s_proven'])
        self.assertFalse(report['official_goal_complete'])
    def test_parent_is_not_added_to_children_or_device_and_host_combined(self):
        records = self.rows[0]['cases']['both']['event_profile'][0]['event_records']
        vector = event_vector(records)
        expected = sum(row['current_stream_elapsed_ms'] / 1000 for row in records if row['name'] != 'parent.vision')
        self.assertAlmostEqual(vector['flat_child_sum_s'], expected)
        self.assertAlmostEqual(vector['vision_parent_minus_flat_children_s'], vector['vision_parent_s'] - expected)
        self.assertNotEqual(vector['flat_child_sum_s'], expected + vector['vision_parent_s'])
    def test_missing_duplicate_layer_or_negative_event_fails_closed(self):
        for mutation in ('missing', 'duplicate_layer', 'negative'):
            records = deepcopy(self.rows[0]['cases']['both']['event_profile'][0]['event_records'])
            if mutation == 'missing': records.pop()
            elif mutation == 'negative': records[0]['current_stream_elapsed_ms'] = -1.
            else:
                indices = [i for i, row in enumerate(records) if row['name'].startswith('linear.')]
                records[indices[0]]['name'] = records[indices[1]]['name']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): event_vector(records)
    def test_representative_is_same_call_complete_wall_slowest_rank(self):
        report = aggregate_profile(self.rows, self.main)
        self.assertEqual(report, aggregate_profile(list(reversed(self.rows)), self.main))
        for mode, case in report['cases'].items():
            for vector in case['profile_vectors']:
                expected = max(vector['all_rank_vectors'], key=lambda row: row['complete_encode_ids_wall_s'])
                self.assertEqual(vector['representative'], expected)
                self.assertEqual(vector['slowest_complete_wall_rank'], expected['rank'])
            chosen = sorted(case['profile_vectors'], key=lambda row: row['representative']['complete_encode_ids_wall_s'])[1]
            published = json.loads((ROOT / 'breakdown.json').read_text())['modes'][mode]
            self.assertEqual(published['profile_call'], chosen['call'])
            self.assertEqual(published['current_stream_category_s'], chosen['representative']['current_stream_category_s'])
            self.assertEqual(published['host_dispatch_category_s'], chosen['representative']['host_dispatch_category_s'])
    def test_clean_call_recording_or_changed_shadow_budget_cannot_pass(self):
        for mutation in ('clean_events', 'cache'):
            rows = deepcopy(self.rows)
            if mutation == 'clean_events': rows[0]['cases']['both']['clean_before'][0]['event_records'] = []
            else: rows[0]['cases']['both']['vision_shadow_cache']['hits'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): aggregate_profile(rows, self.main)


if __name__ == '__main__':
    unittest.main()
