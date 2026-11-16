"""Portable opt-in/isolation/AST/timer safety regressions; no NPU emulation."""
import ast
from contextlib import nullcontext
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PATCHES = ROOT.parent.parent / 'design' / 'patches'
spec = importlib.util.spec_from_file_location('h3_vae64_test', PATCHES / 'h3_910a_vae.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def extract(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<actual function>', 'exec'), namespace)
    return namespace[name]


class InstanceIntegration(unittest.TestCase):
    def test_default_off_before_tensor_or_backend_import(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(module.enabled())
            self.assertFalse(module.install(object(), object()))
        for flag in ('0', 'true', 'yes', ''):
            with patch.dict(os.environ, {'VLLM_OMNI_910A_VAE_FINITE_SELECT': flag}, clear=True):
                self.assertFalse(module.enabled())

    def test_unsupported_device_does_not_touch_vae(self):
        with patch.dict(os.environ, {'VLLM_OMNI_910A_VAE_FINITE_SELECT': '1'}, clear=True):
            self.assertFalse(module.install(object(), SimpleNamespace(type='cpu')))

    def test_clone_changes_only_private_globals(self):
        namespace = {'VALUE': 3}
        exec('def f(x=2, *, y=5): return VALUE + x + y', namespace)
        original = namespace['f']
        clone = module.clone(original, VALUE=100)
        self.assertEqual(original(), 10)
        self.assertEqual(clone(), 107)
        self.assertEqual(namespace['VALUE'], 3)
        self.assertIsNot(clone.__globals__, original.__globals__)
        self.assertEqual(clone.__kwdefaults__, original.__kwdefaults__)

    def test_sdpa_only_final_return_changes(self):
        text = (ROOT.parent / 'round63' / 'source' / 'flash.py').read_text()
        function = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == '_sdpa_attention')
        source = ast.get_source_segment(text, function)
        patched = module.replace_return(source)
        native = ast.parse(source)
        self.assertEqual(patched.body[0].body[-1].value.func.id, '_finite_select64')
        patched.body[0].body.pop()
        native.body[0].body.pop()
        self.assertEqual(ast.dump(patched), ast.dump(native))
        with self.assertRaises(ValueError):
            module.replace_return('def f(out):\n return out')

    def test_mode_restores_state_on_exception(self):
        controller = SimpleNamespace(active=True)
        vae = SimpleNamespace(_h3_910a_finite_select=controller)
        with self.assertRaises(RuntimeError):
            with module.mode(vae, False):
                self.assertFalse(controller.active)
                raise RuntimeError('simulate decode failure')
        self.assertTrue(controller.active)
        with self.assertRaises(RuntimeError):
            with module.mode(object(), True):
                pass

    def test_pipeline_original_decode_body_is_unchanged(self):
        old = ast.parse((ROOT / 'source' / 'pipeline_minimax_h3.py').read_text())
        new = ast.parse((ROOT / 'candidate_pipeline.py').read_text())
        method = lambda tree, name: next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name)
        a = method(old, 'decode')
        b = copy.deepcopy(method(new, '_decode_native64'))
        b.name = 'decode'
        self.assertEqual(ast.dump(a), ast.dump(b))

    def test_adapter_hook_only_occurs_in_video_and_is_default_off(self):
        old = (ROOT / 'source' / 'vae.py').read_text()
        candidate = (ROOT / 'candidate_vae.py').read_text()
        self.assertEqual(candidate.count('install_910a_vae(self, device)'), 1)
        self.assertIn('os.environ.get("VLLM_OMNI_910A_VAE_FINITE_SELECT", "0") == "1"', candidate)
        start = old.index('class MiniMaxH3AudioVAE')
        self.assertEqual(old[start:], candidate[candidate.index('class MiniMaxH3AudioVAE'):])


class ServiceAggregation(unittest.TestCase):
    def test_normal_rejects_mixed_modes_and_unused_candidate(self):
        from summarize import normal
        rows = [{'mode': 'candidate', 'controller_calls_delta': 10,
                 'calls': [{'candidate_enabled': True, 'wall_s': rank + 1., 'stream_s': rank + .5}]}
                for rank in range(8)]
        self.assertEqual(normal(rows)['rank_max_wall_s'], 8.)
        rows[4]['controller_calls_delta'] = 0
        with self.assertRaises(ValueError):
            normal(rows)
        rows[4]['controller_calls_delta'] = 10
        rows[0]['mode'] = 'baseline'
        with self.assertRaises(ValueError):
            normal(rows)

    def test_paired_requires_full_video_both_audio_and_video_identity(self):
        from summarize import paired
        output = {'video': {'shape': [1, 3, 124, 768, 1344], 'dtype': 'torch.uint8', 'bytes': 383975424, 'sha256': 'video'},
                  'audio': {'shape': [1, 2, 166624], 'dtype': 'torch.float32', 'bytes': 1332992, 'sha256': 'audio'}}
        rows = [{'mode': 'paired', 'input_before': {'sha': 'input'}, 'input_after': {'sha': 'input'},
                 'return_call': {'candidate_enabled': True}, 'controller_calls_delta': 40,
                 'calls': [{'candidate_enabled': active, 'wall_s': 15. if active else 30., 'output': copy.deepcopy(output)}
                           for active in (False, True, False, True, False, True, False, True, False)],
                 'return_output': copy.deepcopy(output)} for _ in range(8)]
        result = paired(rows)
        self.assertEqual(result['complete_output_checks'], 80)
        self.assertEqual(result['speedup'], 2.)
        self.assertTrue(result['cases']['candidate']['all_warm_calls_under20s'])
        rows[-1]['calls'][-1]['output']['audio']['sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            paired(rows)


class RetainedServiceEvidence(unittest.TestCase):
    def test_failed_whole_av_gate_is_not_promoted_by_fast_video(self):
        from summarize import summarize
        row = summarize()
        self.assertTrue(row['timing_only_target_met'])
        self.assertFalse(row['stage_accepted_under20s'])
        self.assertFalse(row['official_lossless_goal_complete'])
        self.assertFalse(row['http_run_completed'])
        pair = row['same_input_service_decode']
        self.assertTrue(pair['failed_whole_audio_video_identity_gate'])
        self.assertTrue(pair['video_only_identity_accepted'])
        self.assertEqual(pair['complete_video_checks'], 72)
        for rank in pair['audio_repeat_evidence']:
            self.assertEqual(rank['native_baseline_unique_audio_hashes'], 5)
            self.assertEqual(rank['candidate_unique_audio_hashes'], 4)
        self.assertFalse(row['normal_downloaded_mp4s_all_identical'])
        control = row['native_audio_control']
        self.assertTrue(control['input_and_cpu_weight_masters_immutable'])
        self.assertFalse(control['production_audio_modified'])
        self.assertEqual(len(control['cases']), 4)
        for case in control['cases'].values():
            self.assertEqual(case['distinct_hashes'], 5)
            self.assertTrue(case['all_finite'])
            self.assertTrue(case['rng_state_unchanged'])

    def test_corrupt_video_or_input_cannot_pass_video_identity(self):
        import json
        from summarize import diagnose_failed_pair
        rows = [json.loads((ROOT / 'timing' / ('paired_rank%d.json' % rank)).read_text()) for rank in range(8)]
        changed = copy.deepcopy(rows)
        changed[4]['calls'][2]['output']['video']['sha256'] = 'not_the_reference'
        self.assertFalse(diagnose_failed_pair(changed)['video_only_identity_accepted'])
        changed = copy.deepcopy(rows)
        changed[0]['input_after']['video']['sha256'] = 'mutated'
        with self.assertRaises(ValueError):
            diagnose_failed_pair(changed)
        changed = copy.deepcopy(rows)
        changed[0]['mode'] = 'baseline'
        with self.assertRaises(ValueError):
            diagnose_failed_pair(changed)

    def test_restoration_and_executed_source_fingerprints(self):
        import hashlib
        import json
        final = json.loads((ROOT / 'final_verification.json').read_text())
        expected = json.loads((ROOT / 'expected_final.json').read_text())
        restored = json.loads((ROOT / 'restoration.json').read_text())
        self.assertTrue(restored['restored'])
        self.assertTrue(final['all_sources_match'])
        self.assertTrue(final['original_adapter_and_pipeline_restored'])
        self.assertEqual(len(final['production']), 23)
        self.assertEqual(len(final['diagnostics']), 12)
        for category in ('production', 'diagnostics'):
            self.assertEqual(set(final[category]), set(expected[category]))
            self.assertTrue(all(value['matches'] and value['sha256'] == expected[category][name]
                                for name, value in final[category].items()))
        for name in ('h3_910a_vae.py', 'h3_910a_vae_kernel.py'):
            remote = '/tmp/h3_round64_service01/' + name
            self.assertEqual(hashlib.sha256((PATCHES / name).read_bytes()).hexdigest(), expected['diagnostics'][remote])


class CompleteStageTiming(unittest.TestCase):
    def test_timer_includes_async_tail_and_excludes_start_barrier(self):
        clock = {'now': 0., 'pending': 0., 'mode': None}
        class Event:
            def __init__(self, **kw):
                self.at = None
            def record(self):
                self.at = clock['now'] + clock['pending']
            def elapsed_time(self, other):
                return (other.at - self.at) * 1000
        def synchronize():
            clock['now'] += clock['pending']
            clock['pending'] = 0.
        def barrier():
            clock['now'] += 100.
        def native(video, audio, **kw):
            clock['now'] += 3.
            clock['pending'] += 11.
            return ('video', 'audio')
        torch = SimpleNamespace(distributed=SimpleNamespace(barrier=barrier),
                                npu=SimpleNamespace(synchronize=synchronize, Event=Event))
        fn = extract('h3_service_audit64.py', 'synchronized_call', {'torch': torch,
                     'mode': lambda *a: nullcontext(), 'time': SimpleNamespace(perf_counter=lambda: clock['now'])})
        out, row = fn(SimpleNamespace(video_vae=object()), native, object(), object(), True, width=1344, height=768)
        self.assertEqual(out, ('video', 'audio'))
        self.assertEqual(row['wall_s'], 14.)
        self.assertEqual(row['stream_s'], 14.)
        self.assertTrue(row['candidate_enabled'])


if __name__ == '__main__':
    unittest.main()
