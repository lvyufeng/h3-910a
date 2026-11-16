import copy
import unittest
from summarize import CHILD_COUNTS, TOP_COUNTS, group_events


def profile(combined=False):
    expected = {**TOP_COUNTS, **CHILD_COUNTS}
    if combined:
        expected['top.shared_activation_cast'] = 50
    events = []
    for name, count in expected.items():
        events += [{'name': name, 'current_stream_elapsed_ms': 1., 'host_dispatch_s': .002} for _ in range(count)]
    return {'seconds': 2., 'events': events, 'stream_begin': 'npu_stream0', 'stream_end': 'npu_stream0'}


class AttributionContract(unittest.TestCase):
    def test_child_categories_are_not_added_to_top_or_host(self):
        row = group_events(profile(), False)
        self.assertAlmostEqual(row['top_nonoverlap_event_sum_s'], sum(TOP_COUNTS.values()) / 1000)
        self.assertAlmostEqual(row['wall_minus_top_event_sum_s'], 2 - sum(TOP_COUNTS.values()) / 1000)
        self.assertEqual(row['categories']['child.all_reduce']['count'], 100)
        self.assertAlmostEqual(row['categories']['child.all_reduce']['current_stream_elapsed_s'], .100)
        self.assertAlmostEqual(row['categories']['child.all_reduce']['host_dispatch_s'], .200)

    def test_combined_activation_prepare_exact_count(self):
        row = group_events(profile(True), True)
        self.assertEqual(row['categories']['top.shared_activation_cast']['count'], 50)
        self.assertAlmostEqual(row['top_nonoverlap_event_sum_s'], (sum(TOP_COUNTS.values()) + 50) / 1000)
        with self.assertRaises(ValueError):
            group_events(profile(False), True)

    def test_missing_duplicate_event_invalid_duration_and_stream_fail_closed(self):
        data = profile()
        for mutation in (lambda d: d['events'].pop(),
                         lambda d: d['events'].append(copy.deepcopy(d['events'][0])),
                         lambda d: d['events'][0].update(current_stream_elapsed_ms=-1.),
                         lambda d: d['events'][0].update(host_dispatch_s=float('nan')),
                         lambda d: d.update(stream_end='other')):
            changed = copy.deepcopy(data)
            mutation(changed)
            with self.assertRaises(ValueError):
                group_events(changed, False)

    def test_uncovered_wall_not_forced_into_cpu_network_or_clamped(self):
        data = profile()
        data['seconds'] = .1
        row = group_events(data, False)
        self.assertLess(row['wall_minus_top_event_sum_s'], 0)
        self.assertNotIn('CPU', set(row['categories']))


if __name__ == '__main__':
    unittest.main()
