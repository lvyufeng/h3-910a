#!/usr/bin/env python3
"""Contracts for the stack_tiling candidate run under the user-approved band rule."""
import ast
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):
            if p.name=='test_boundary_trace137.py':continue
            ast.parse(p.read_text(),filename=str(p))
    def test_rule_recorded_and_scoped(self):
        r=json.loads((ROOT/'acceptance_rule139.json').read_text())
        self.assertEqual(r['user_choice'],'option 1 (native noise band as upper bound)')
        self.assertIn('no worse than the native decode',r['rule'])
        self.assertIn('reported as unsatisfiable',r['control_requirement'])
        self.assertIn('Issue #7',r['scope'])
        self.assertEqual(r['measured_native_band_from_R138']['maxabs_uint8_LSB'],1)
    def test_three_phases_and_control(self):
        s=(ROOT/'stack_rpc139.py').read_text()
        for call in ("run_phase('native',False,REPEATS)","run_phase('candidate',True,REPEATS)","run_phase('restored',original_stack_tiling,REPEATS)"):
            self.assertIn(call,s)
        self.assertIn("restored_control_within_band",s)
        self.assertIn("rule_satisfiable_by_pure_native_control",s)
        self.assertIn("candidate_passes",s)
    def test_candidate_is_native_knob_only(self):
        s=(ROOT/'stack_rpc139.py').read_text()
        self.assertIn("'name':'stack_tiling'",s)
        self.assertIn("'native_config_option':True",s)
        self.assertIn("'tile_windows_or_blend_changed':False",s)
        self.assertIn("'parameters_changed':False",s)
        self.assertIn("assert bool(model.stack_tiling)==original_stack_tiling",s)
        for banned in ('omni.generate(','FinalAdaLN','ExplicitFP32Bias','use_deterministic_algorithms(True'):
            self.assertNotIn(banned,s)
    def test_band_uses_complete_uint8_domain(self):
        s=(ROOT/'stack_rpc139.py').read_text()
        self.assertIn('np.count_nonzero(av!=bv)',s)
        self.assertIn('total_words',s)
        self.assertIn("max_abs_LSB",s)
    def test_no_r138_roots(self):
        for name in ('stack_rpc139.py','launch_stack139.py','generate_stack139.py','collect_stack139.py'):
            text=(ROOT/name).read_text()
            self.assertNotIn('h3_vae13803',text)
            self.assertNotIn('engine_vae138c',text)


if __name__=='__main__':unittest.main()
