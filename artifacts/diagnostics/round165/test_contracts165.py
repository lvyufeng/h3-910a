#!/usr/bin/env python3
"""Contracts for the R165 device-predicate finding."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_predicates_are_measured_unreliable(self):
        b=json.loads((ROOT/'equiv_chain165.json').read_text())
        p=b['float_predicates_are_unreliable']
        self.assertGreater(p['isnan_plus_isinf_disagrees_with_native'],0)
        self.assertGreater(p['self_inequality_disagrees_with_native'],0)
        self.assertEqual(b['cases'],45)
    def test_worst_observations_are_recorded(self):
        b=json.loads((ROOT/'equiv_chain165.json').read_text())
        joined=' '.join(b['worst_observations'])
        self.assertIn('isfinite().all() = True',joined)
        self.assertIn('65504.0',joined)
    def test_root_cause_points_at_existing_code(self):
        b=json.loads((ROOT/'equiv_chain165.json').read_text())
        self.assertIn('without unreliable float predicates',b['root_cause_already_documented_here'])
    def test_cheap_check_is_declared_unsound_by_construction(self):
        b=json.loads((ROOT/'equiv_chain165.json').read_text())
        self.assertIn('unsound by construction',b['consequence'])
    def test_salvage_keeps_bit_patterns(self):
        b=json.loads((ROOT/'equiv_chain165.json').read_text())
        d=b['salvageable_direction']
        self.assertIn('bit-pattern',d)
        self.assertIn('int16',d)
        self.assertIn('0x7c00',d)
    def test_double_verification_required(self):
        b=json.loads((ROOT/'equiv_chain165.json').read_text())
        self.assertEqual(len(b['must_still_pass']),2)

if __name__=='__main__':unittest.main()
