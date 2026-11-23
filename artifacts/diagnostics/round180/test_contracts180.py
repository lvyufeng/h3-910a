#!/usr/bin/env python3
"""Contracts for the R180 acceptance protocol and golden reference."""
import json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_reference_self_check_passes(self):
        r=subprocess.run([sys.executable,str(ROOT/'reference180.py')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('failures=0',r.stdout)
    def test_reference_is_dependency_free(self):
        src=(ROOT/'reference180.py').read_text()
        self.assertNotIn('import numpy',src)
        self.assertIn('import struct',src)
    def test_key_adversarial_cases_present(self):
        sys.path.insert(0,str(ROOT))
        import importlib
        mod=importlib.import_module('reference180')
        names={c[0] for c in mod.CASES}
        for must in ('nan_and_saturation','inf_and_saturation','many_bad',
                     'just_below_max','rounds_up_to_max','max_finite_many'):
            self.assertIn(must,names)
    def test_rounding_trap_is_pinned(self):
        sys.path.insert(0,str(ROOT))
        import importlib
        mod=importlib.import_module('reference180')
        self.assertEqual(mod.max_magnitude_bits([65472.0]),0x7BFE)
        self.assertEqual(mod.max_magnitude_bits([65503.0]),0x7BFF)
    def test_verdict_semantics_documented(self):
        text=(ROOT/'accept180.md').read_text()
        self.assertIn('判定',text)
        self.assertIn('不是**精确计数**',text)
    def test_pass_criteria_pre_committed(self):
        text=(ROOT/'accept180.md').read_text()
        self.assertIn('37/37',text)
        self.assertIn('465/465',text)
        self.assertIn('不可用',text)

if __name__=='__main__':unittest.main()
