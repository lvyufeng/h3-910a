#!/usr/bin/env python3
"""Contracts for the R160 withdrawal of the FA padding candidate."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_read_only(self):
        b=json.loads((ROOT/'fa_padding160.json').read_text())
        self.assertTrue(b['read_only']); self.assertTrue(b['no_machine_used'])
        self.assertTrue(b['no_code_changed'])
    def test_candidate_is_withdrawn(self):
        b=json.loads((ROOT/'fa_padding160.json').read_text())
        self.assertIn('WITHDRAWN',b['verdict'])
        self.assertIn('before any run',b['verdict'])
    def test_tile_counts_prove_it(self):
        b=json.loads((ROOT/'fa_padding160.json').read_text())
        t=b['why']['tile_counts']
        self.assertEqual(t['unpadded_41792']['br'],64)
        self.assertEqual(t['unpadded_41792']['tiles'],426409)
        self.assertEqual(t['padded_43776']['tiles'],49248)
        self.assertGreater(t['unpadded_41792']['tiles']/t['padded_43776']['tiles'],8.0)
    def test_search_shows_current_choice_is_best(self):
        b=json.loads((ROOT/'fa_padding160.json').read_text())
        best=b['padding_optimality_search']['best_five_by_tile_count']
        self.assertEqual(best[0]['kernel_n'],43776)
        self.assertEqual(b['padding_optimality_search']['admissible_candidates'],90)
        self.assertLess(best[0]['tiles'],best[1]['tiles'])
    def test_line_is_closed_not_solved(self):
        b=json.loads((ROOT/'fa_padding160.json').read_text())
        self.assertIn('CLOSED',b['line_status'])
        self.assertIn('real kernel work',b['remaining_avenue'])

if __name__=='__main__':unittest.main()
