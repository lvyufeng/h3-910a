"""Dependency-free receipt and extracted timing-contract regressions.

Hardware correctness is in retained NPU receipts; these tests prevent reporting
and timing errors without pretending to emulate Ascend arithmetic on the host.
"""
import ast
from contextlib import nullcontext
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

from summarize import aggregate_decode, MODES, EXPECTED_VIDEO_BYTES, EXPECTED_VIDEO_SHAPE, summarize

ROOT = Path(__file__).resolve().parent


def synthetic_rows():
    rows = []
    for rank in range(8):
        output = {'sha256': 'actual', 'shape': EXPECTED_VIDEO_SHAPE.copy(),
                  'dtype': 'torch.uint8', 'bytes': EXPECTED_VIDEO_BYTES}
        case = {'first_call': {'wall_s': 999., 'stream_s': 999., 'output': output},
                'warm_calls': [{'wall_s': 10. + i, 'stream_s': 9. + i, 'output': copy.deepcopy(output)}
                               for i in range(3)]}
        rows.append({'rank': rank, 'completed': True, 'source_sha256': 'source',
                     'reference_output_sha256': 'actual', 'cases': {mode: copy.deepcopy(case) for mode in MODES}})
    return rows


class ReceiptContracts(unittest.TestCase):
    def test_requires_eight_unique_ranks(self):
        rows = synthetic_rows()
        with self.assertRaises(ValueError):
            aggregate_decode(rows[:-1])
        rows[-1]['rank'] = 0
        with self.assertRaises(ValueError):
            aggregate_decode(rows)

    def test_rank_max_precedes_median_and_excludes_first_call(self):
        rows = synthetic_rows()
        for row in rows:
            for case in row['cases'].values():
                for call in case['warm_calls']:
                    call['wall_s'] = 1.
        for i in range(3):
            rows[i]['cases']['baseline_c128']['warm_calls'][i]['wall_s'] = 9.
        result = aggregate_decode(rows)['cases']['baseline_c128']
        self.assertEqual(result['per_call_rank_max_wall_s'], [9., 9., 9.])
        self.assertEqual(result['median_rank_max_wall_s'], 9.)
        self.assertEqual(result['first_call_rank_max_wall_s'], 999.)

    def test_does_not_trust_declared_exact_flag(self):
        rows = synthetic_rows()
        case = rows[-1]['cases']['candidate_c256']
        case['all_calls_capture_exact'] = True
        case['warm_calls'][1]['output']['sha256'] = 'wrong'
        result = aggregate_decode(rows)['cases']['candidate_c256']
        self.assertFalse(result['all_calls_capture_exact'])
        self.assertFalse(result['all_repeats_exact'])
        self.assertFalse(result['accepted_for_this_captured_latent'])

    def test_correct_hash_does_not_waive_shape_dtype_or_bytes(self):
        for key, bad in [('shape', [1, 3, 120, 768, 1344]), ('dtype', 'torch.float16'), ('bytes', 42)]:
            rows = synthetic_rows()
            rows[0]['cases']['baseline_c128']['warm_calls'][0]['output'][key] = bad
            self.assertFalse(aggregate_decode(rows)['cases']['baseline_c128']['accepted_for_this_captured_latent'])

    def test_failed_or_mixed_source_proof_is_rejected(self):
        rows = synthetic_rows()
        rows[3]['failure'] = {'type': 'RuntimeError'}
        with self.assertRaises(ValueError):
            aggregate_decode(rows)
        rows = synthetic_rows()
        rows[4]['source_sha256'] = 'other'
        with self.assertRaises(ValueError):
            aggregate_decode(rows)

    def test_requires_three_warm_calls_everywhere(self):
        rows = synthetic_rows()
        rows[4]['cases']['baseline_c128']['warm_calls'].pop()
        with self.assertRaises(ValueError):
            aggregate_decode(rows)

    def test_executed_tile_and_attention_sources_and_scope(self):
        result = summarize()
        self.assertFalse(result['loader_contract']['default_fp32_construction_matches_capture'])
        self.assertTrue(result['loader_contract']['default_half_then_fp32_storage_matches_capture'])
        for case in result['actual_tile']['cases'].values():
            self.assertTrue(case['all_repeats_equal'])
            self.assertTrue(case['raw_bits_equal_to_actual_capture'])
        self.assertFalse(result['actual_attention']['custom_fa_lossless_accepted'])
        self.assertFalse(result['production_fast_path_installed'])
        self.assertFalse(result['strict_goal_complete'])


class ExtractedTimingContract(unittest.TestCase):
    def test_once_drains_device_before_timing_end_and_excludes_hashing(self):
        tree = ast.parse((ROOT / 'decode_replay_tp8.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'once')
        clock = {'now': 0., 'queued': 0., 'syncs': 0, 'calls': 0}
        class Event:
            def __init__(self, **kwargs):
                self.at = None
            def record(self):
                self.at = clock['now'] + clock['queued']
            def elapsed_time(self, other):
                return (other.at - self.at) * 1000
        def sync():
            clock['now'] += clock['queued']
            clock['queued'] = 0.
            clock['syncs'] += 1
        def barrier():
            clock['now'] += 7.  # excluded start alignment
        def decode(latent):
            clock['calls'] += 1
            clock['now'] += 2.  # Python enqueue work
            clock['queued'] += 8.  # asynchronous outstanding device work
            return object()
        def proof(out, expected):
            clock['now'] += 100.  # deliberately expensive CPU download/hash
            return {'sha256': expected}
        torch = SimpleNamespace(distributed=SimpleNamespace(barrier=barrier),
                                npu=SimpleNamespace(synchronize=sync, Event=Event),
                                autocast=lambda *a, **kw: nullcontext(), float16='half')
        namespace = {'torch': torch, 'time': SimpleNamespace(perf_counter=lambda: clock['now']),
                     'output_receipt': proof}
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, '<extracted actual once>', 'exec'), namespace)
        row = namespace['once'](SimpleNamespace(decode_latent=decode), object(), 'actual')
        self.assertEqual(row['wall_s'], 10.)
        self.assertEqual(row['stream_s'], 10.)
        self.assertEqual(clock['now'], 117.)
        self.assertEqual(clock['syncs'], 2)
        self.assertEqual(clock['calls'], 1)

    def test_default_dtype_restoration_is_in_finally(self):
        for filename in ('tile_replay03.py', 'decode_replay_tp8.py'):
            tree = ast.parse((ROOT / filename).read_text())
            finalizers = [node.finalbody for node in ast.walk(tree) if isinstance(node, ast.Try)]
            self.assertTrue(any(any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                                    and call.func.attr == 'set_default_dtype'
                                    for statement in statements for call in ast.walk(statement))
                                for statements in finalizers), filename)


if __name__ == '__main__':
    unittest.main()
