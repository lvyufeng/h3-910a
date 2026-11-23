#!/usr/bin/env python3
"""Contracts for the R164 adversarial test that disproved the R163 equivalence claim."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_the_cheap_implementation_fails(self):
        b=json.loads((ROOT/'equiv164.json').read_text())
        self.assertFalse(b['verdicts_always_agree'])
        self.assertGreater(b['verdict_mismatches'],0)
        self.assertIn('NOT EQUIVALENT',b['conclusion'])
    def test_the_two_failing_cases_are_named(self):
        b=json.loads((ROOT/'equiv164.json').read_text())
        names={c['case'] for c in b['failing_cases']}
        self.assertIn('nan_and_saturation',names)
        self.assertIn('many_bad',names)
    def test_r163_claim_is_narrowed(self):
        b=json.loads((ROOT/'equiv164.json').read_text())
        c=b['correction_to_r163']
        self.assertIn('zero bad elements',c)
        self.assertIn('must not be adopted',c)
    def test_multiplicity_difference_is_recorded_not_hidden(self):
        b=json.loads((ROOT/'equiv164.json').read_text())
        names={c['case'] for c in b['count_only_differences']}
        self.assertIn('two_pos_saturation_multiplicity',names)
    def test_failsafe_design_is_specified(self):
        b=json.loads((ROOT/'equiv164.json').read_text())
        d=b['salvageable_design']
        self.assertIn('no non-finite',d)
        self.assertIn('fall back',d)
    def test_no_adoption_claimed(self):
        b=json.loads((ROOT/'equiv164.json').read_text())
        self.assertTrue(b['nothing_adopted'])
        self.assertTrue(b['no_check_removed'])

if __name__=='__main__':unittest.main()
