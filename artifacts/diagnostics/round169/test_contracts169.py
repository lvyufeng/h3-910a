#!/usr/bin/env python3
"""Contracts for the R169 read-only memory feasibility note."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_verdict_is_infeasible(self):
        b=json.loads((ROOT/'defer169.json').read_text())
        self.assertIn('MARGINAL',b['verdict'])
        self.assertIn('NOT proven impossible',b['verdict'])
    def test_arithmetic_is_shown_and_is_tight(self):
        b=json.loads((ROOT/'defer169.json').read_text())
        self.assertGreater(b['total_deferred_gib'],15.0)
        self.assertLess(b['margin_gib'],3.0)

    def test_assumptions_are_flagged(self):
        b=json.loads((ROOT/'defer169.json').read_text())
        self.assertIn('not measured',json.dumps(b['assumed_terms']))
        self.assertIn('recorded this shape directly',b['measured_terms']['source'])
    def test_self_correction_is_recorded(self):
        b=json.loads((ROOT/'defer169.json').read_text())
        self.assertIn('plainly infeasible',b['self_correction'])
        self.assertIn('corrected',b['self_correction'])
    def test_no_machine_time_claimed(self):
        b=json.loads((ROOT/'defer169.json').read_text())
        self.assertTrue(b['no_machine_time'])
        self.assertTrue(b['read_only_analysis'])
    def test_surviving_and_dead_parts_named(self):
        b=json.loads((ROOT/'defer169.json').read_text())
        self.assertIn('equivalence proof',b['what_survives'])
    def test_only_side_stream_variant_remains(self):
        b=json.loads((ROOT/'defer169.json').read_text())
        self.assertIn('side-stream',b['only_remaining_variant'])
        self.assertIn('open measurement',b['only_remaining_variant'])

if __name__=='__main__':unittest.main()
