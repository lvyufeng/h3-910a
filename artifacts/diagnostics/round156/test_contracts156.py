#!/usr/bin/env python3
"""Contracts for the R156 allocator check."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_probe_is_read_only_and_native_path(self):
        s=(ROOT/'mem_probe156.py').read_text()
        self.assertNotIn('mlp.forward=',s)
        self.assertIn('memory_stats',s)
        self.assertIn("if _STATE['installed']:",s)
    def test_allocator_hypothesis_rejected_on_evidence(self):
        b=json.loads((ROOT/'mem156.json').read_text())
        self.assertEqual(b['allocator_hypothesis'],'REJECTED')
        self.assertEqual(b['allocator_counters']['num_alloc_retries'],0)
        self.assertEqual(b['allocator_counters']['num_ooms'],0)
    def test_mlp_medians_match_r155(self):
        b=json.loads((ROOT/'mem156.json').read_text())
        self.assertAlmostEqual(b['mlp_total_median_ms'],88.217,delta=1.5)
        self.assertAlmostEqual(b['silu_median_ms'],1.14,delta=0.5)
    def test_r125_cross_check_is_recorded(self):
        b=json.loads((ROOT/'mem156.json').read_text())
        self.assertIn('18.83',b['cross_check_from_r125'])
        self.assertIn('waiting rather than',b['cross_check_from_r125'])
    def test_remaining_hypotheses_admit_the_boundary_problem(self):
        b=json.loads((ROOT/'mem156.json').read_text())
        joined=' '.join(b['remaining_hypotheses'])
        self.assertIn('pre-hook',joined)
        self.assertIn('staging wait',joined)

if __name__=='__main__':unittest.main()
