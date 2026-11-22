#!/usr/bin/env python3
"""Portable contracts for unchanged original fixed-input component diagnostics."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_authorized_native_only(self):
        a=json.loads((ROOT/'approval137.json').read_text());self.assertTrue(a['original_math_and_parameters_unchanged'])
        for key in ('new_candidate_or_numeric_tolerance','full49_generation','copy_candidate','model_weights_or_input_activation_archives_to_client'):self.assertFalse(a[key])
    def test_minimal_bootstrap_not_R136_request_wrapper(self):
        s=(ROOT/'integration_hooks127.py').read_text()
        self.assertNotIn('request_directory()',s);self.assertNotIn('pipeline.MiniMaxH3Pipeline.decode=',s)
        self.assertNotIn('pipeline.MiniMaxH3Pipeline.forward=',s);self.assertNotIn('_initial_noise=',s)
        self.assertIn('install_patch_projection_guard123(self)+install_time_projection_guard123(self)',s)
        self.assertIn('must never execute DiT or a sampler',s)
    def test_encoder_native_entry_and_capture(self):
        s=(ROOT/'component_rpc137.py').read_text()
        self.assertIn('pipeline.encode_prompt(prepared)',s);self.assertIn("pipeline._encode_text_hidden(fixed['args'][0],fixed['kwargs'])",s)
        self.assertIn('output=original_encode(*args,**kwargs)',s)
        self.assertIn('types.MethodType(capture,encoder)',s);self.assertIn("delattr(encoder,'_encode')",s)
        self.assertIn('before==after and fmt_before==fmt_after',s)
    def test_original_full_audio_decode_not_remote_shortcut(self):
        s=(ROOT/'component_rpc137.py').read_text()
        self.assertIn('with pipeline._component_on_device(audio),half_nan_scope(enabled=True)',s)
        self.assertIn('output=audio.decode_latent(latent)',s);self.assertNotIn('audio.remote.decode(',s)
        self.assertIn('shape==(2,32,207)',s);self.assertIn('cpu.shape==(1,2,165600)',s)
    def test_parametrization_native_output_not_dynamic_weight_property(self):
        s=(ROOT/'component_rpc137.py').read_text()
        self.assertIn("vars(module).get('_modules',{}).get('parametrizations')",s)
        self.assertIn("sequence=parametrizations['weight']",s);self.assertIn("parametrizations.weight.0',sequence[0]",s)
        parsed=ast.parse(s);self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='weight' and isinstance(n.value,ast.Name) and n.value.id=='module' for n in ast.walk(parsed)))
        self.assertNotIn('parametrize.cached(',s)
        trace=(ROOT/'boundary_trace137.py').read_text();self.assertNotIn("getattr(module, 'weight'",trace)
    def test_native_trace_restored_counts_separate(self):
        s=(ROOT/'component_rpc137.py').read_text()
        self.assertIn("(('native',8),('trace',2),('observer_restored',4))",s)
        self.assertIn('original_parameters_buffers_and_CPU_masters_unchanged',s)
        self.assertIn('first_boundary_change(traced_rows[0],traced_rows[1])',s)
        driver=(ROOT/'generate_components137.py').read_text();self.assertNotIn('omni.generate(',driver)
        self.assertIn("len(c['passes'])==14",driver)
    def test_false_flags_no_candidate(self):
        s=(ROOT/'component_rpc137.py').read_text()
        self.assertNotIn('torch.use_deterministic_algorithms(True',s);self.assertNotIn('FinalAdaLNCandidate',s)
        self.assertNotIn('ExplicitFP32Bias',s);self.assertIn('parametrize._cache_enabled',s)
        self.assertIn("report['final_runtime']==report['initial_runtime']",s)
    def test_shared_private_control_and_failure(self):
        s=(ROOT/'component_rpc137.py').read_text()
        self.assertIn("name='r137_original_components_fixed_repeat'",s)
        self.assertIn("'phase':'component_RPC_entry'",s);self.assertIn('cleanup_state_unknown',s)
        self.assertIn('if primary is None:raise RuntimeError',s)
        abort=(ROOT/'private_abort137.py').read_text();self.assertIn('pidfd_send_signal',abort);self.assertNotIn('os.kill(',abort)
    def test_exact_runtime_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();session=root/'engine_components137a';session.mkdir()
            text=(ROOT/'runtime_context127.py').read_text().replace("ROOT=Path('/tmp/h3_components13701')",f'ROOT=Path({str(root)!r})')
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(session)}):
                ns={'__name__':'runtime_test137'};exec(compile(text,'runtime_test137','exec'),ns);self.assertEqual(ns['SESSION'],session)
        self.assertIn("SESSION.name=='engine_components137a'",(ROOT/'runtime_context127.py').read_text())
    def test_refuse_uncertain_launch(self):
        s=(ROOT/'launch_components137.py').read_text()
        self.assertIn('not SESSION.exists() and not receipt.exists()',s)
        self.assertIn('original_audio_latent_SHA_verified',s);self.assertIn('native_source_SHA_verified',s)
        self.assertIn('deadline_monotonic=time.monotonic()+5400',s)


if __name__=='__main__':unittest.main()
