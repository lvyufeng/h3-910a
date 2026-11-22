#!/usr/bin/env python3
"""Portable contracts for the measurement-only video-VAE timing/repeatability run."""
import ast
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_measurement_only_contract(self):
        c=json.loads((ROOT/'issue7_contract138.json').read_text())
        self.assertTrue(c['measurement_only']);self.assertTrue(c['no_math_change'])
        self.assertTrue(c['no_candidate']);self.assertTrue(c['no_tolerance'])
        self.assertEqual(c['fixed_latent']['shape'],[1,24,37,48,84])
        self.assertEqual(c['fixed_latent']['file_sha256'],'bf9daec88b451f9cd7273a1544260ce6e8bd9c594225e5cbef3529ff0f5d02e0')
    def test_rpc_measures_original_component_only(self):
        s=(ROOT/'vae_rpc138.py').read_text()
        self.assertIn('video_vae.decode_latent(latent)',s)
        self.assertIn('pipeline._component_on_device(video_vae)',s)
        self.assertIn("REPEATS=4",s)
        self.assertIn('decode_base_s',s)
        self.assertIn('_decoder_tile_count',s)
        self.assertIn('local_tile_indices',s)
        # no sampling / no DiT / no candidate installer
        for banned in ('omni.generate(','pipeline.forward(','FinalAdaLN','ExplicitFP32Bias','use_deterministic_algorithms(True'):
            self.assertNotIn(banned,s)
    def test_no_dit_or_sampler_in_driver(self):
        s=(ROOT/'generate_vae138.py').read_text()
        self.assertNotIn('omni.generate(',s)
        self.assertIn("'r138_native_video_vae_timing_repeat'",s)
        self.assertIn('video_vae_decodes_allranks',s)
    def test_timer_is_removed_and_state_verified(self):
        s=(ROOT/'vae_rpc138.py').read_text()
        self.assertIn("report['decode_base_timer_removed']=True",s)
        self.assertIn('same_original_state(report[\'state_before\'],report[\'state_after\'])',s)
        self.assertIn('original_parameter_state_unchanged',s)
        self.assertIn("assert report['final_runtime']==report['initial_runtime']",s)
    def test_launcher_refuses_replay_and_requires_preflight(self):
        s=(ROOT/'launch_vae138.py').read_text()
        self.assertIn('Refuse replay of uncertain launch',s)
        self.assertIn("preflight138.json",s)
        self.assertIn('fixed_video_latent_SHA_verified',s)
        self.assertIn('pidfd_open',s)
        self.assertIn('deadline_monotonic=time.monotonic()+5400',s)
    def test_no_r137_roots_left(self):
        for name in ('vae_rpc138.py','launch_vae138.py','generate_vae138.py','collect_vae138.py','runtime_context127.py','cohort128.py','private_abort137.py','sitecustomize.py','full_bootstrap127.py'):
            text=(ROOT/name).read_text()
            self.assertNotIn('h3_components13703',text)
            self.assertNotIn('engine_components137c',text)


if __name__=='__main__':unittest.main()
