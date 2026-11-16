"""Portable timing-orchestration tests; not emulated NPU performance results."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('event_profile66_contracts', ROOT / 'event_profile66.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Clock:
    def __init__(self):
        self.device_ms = 0.
        self.drains = 0
        self.constructed = 0
        self.stream = 'stream0'
    def event(self, enable_timing):
        assert enable_timing
        self.constructed += 1
        clock = self
        class Event:
            def record(self):
                self.timestamp = clock.device_ms
            def elapsed_time(self, end):
                return end.timestamp - self.timestamp
        return Event()
    def synchronize(self):
        self.drains += 1
    def torch(self):
        return SimpleNamespace(npu=SimpleNamespace(Event=self.event, synchronize=self.synchronize, current_stream=lambda: self.stream))


class EventContracts(unittest.TestCase):
    def test_preallocate_prime_before_measurement(self):
        clock = Clock()
        profile = module.EventProfile(clock.torch(), capacity=3)
        self.assertEqual(clock.constructed, 6)
        self.assertEqual(clock.drains, 1)
        profile.begin()
        profile.call('top.norm', lambda: setattr(clock, 'device_ms', 5.))
        self.assertEqual(clock.drains, 1)
        row = profile.finish()
        self.assertEqual(clock.drains, 2)
        self.assertEqual(clock.constructed, 6)
        self.assertEqual(row[0]['current_stream_elapsed_ms'], 5.)

    def test_nested_pairs_are_distinct_and_not_claimed_as_additive(self):
        clock = Clock()
        profile = module.EventProfile(clock.torch(), capacity=4)
        def parent():
            clock.device_ms += 2.
            profile.call('child.linear', lambda: setattr(clock, 'device_ms', clock.device_ms + 3.))
            clock.device_ms += 1.
        profile.begin()
        profile.call('top.row', parent)
        rows = profile.finish()
        self.assertEqual([row['name'] for row in rows], ['top.row', 'child.linear'])
        self.assertEqual([row['current_stream_elapsed_ms'] for row in rows], [6., 3.])
        self.assertEqual(clock.drains, 2)

    def test_inactive_has_no_events_or_component_sync(self):
        clock = Clock()
        profile = module.EventProfile(clock.torch(), capacity=2)
        self.assertEqual(profile.call('top.op', lambda x: x + 1, 4), 5)
        self.assertEqual(profile.records, [])
        self.assertEqual(clock.drains, 1)

    def test_capacity_exhaustion_fails_instead_of_reusing_pair(self):
        clock = Clock()
        profile = module.EventProfile(clock.torch(), capacity=1)
        profile.begin()
        profile.call('top.op', lambda: None)
        with self.assertRaises(RuntimeError):
            profile.call('top.other', lambda: None)
        self.assertEqual(len(profile.finish()), 1)

    def test_exception_records_tail_and_still_propagates(self):
        clock = Clock()
        profile = module.EventProfile(clock.torch(), capacity=2)
        profile.begin()
        def fail():
            clock.device_ms += 3.
            raise ValueError('original failure')
        with self.assertRaisesRegex(ValueError, 'original failure'):
            profile.call('top.op', fail)
        self.assertEqual(profile.finish()[0]['current_stream_elapsed_ms'], 3.)

    def test_stream_change_rejected_not_silently_mixed(self):
        clock = Clock()
        profile = module.EventProfile(clock.torch(), capacity=2)
        profile.begin()
        profile.call('top.op', lambda: None)
        clock.stream = 'stream1'
        with self.assertRaisesRegex(RuntimeError, 'stream changed'):
            profile.finish()

    def test_invalid_begin_or_finish_rejected(self):
        clock = Clock()
        profile = module.EventProfile(clock.torch(), capacity=2)
        with self.assertRaises(RuntimeError):
            profile.finish()
        profile.begin()
        with self.assertRaises(RuntimeError):
            profile.begin()


if __name__ == '__main__':
    unittest.main()
