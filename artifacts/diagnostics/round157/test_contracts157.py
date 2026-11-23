#!/usr/bin/env python3
"""Contracts for the R157 correction and the resident control attempt."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_correction_is_stated_explicitly(self):
        b=json.loads((ROOT/'res157.json').read_text())
        c=b['correction']
        self.assertIn('three forward passes',c['what_is_true'])
        self.assertIn('No hook was ever installed more than once',c['what_is_true'])
        self.assertIn('the explanation was wrong, not the data',c['why_the_numbers_survived'])
    def test_resident_control_is_recorded_as_unavailable(self):
        b=json.loads((ROOT/'res157.json').read_text())
        r=b['resident_control']
        self.assertIn('OOM',r['result'])
        self.assertIn('31.06',r['detail'])
        self.assertIn('cannot be tested by residency',r['consequence'])
    def test_values_that_survive_are_listed(self):
        b=json.loads((ROOT/'res157.json').read_text())
        joined=' '.join(b['correction']['still_correct'])
        self.assertIn('7736',joined)
        self.assertIn('closure',joined)
    def test_next_control_is_diagnostic_only(self):
        b=json.loads((ROOT/'res157.json').read_text())
        self.assertIn('diagnostic only',b['next_control'])
        self.assertIn('never be proposed as an optimisation',b['next_control'])

if __name__=='__main__':unittest.main()
