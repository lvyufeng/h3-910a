#!/usr/bin/env python3
"""Contracts for the R170 measured ceiling judgement."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_deferral_is_closed_by_measurement(self):
        b=json.loads((ROOT/'ceiling170.json').read_text())
        self.assertIn('NOT FEASIBLE',b['deferral_verdict'])
        self.assertGreater(b['deferred_total_gib'],b['headroom_gib'])
    def test_measured_shapes_are_recorded(self):
        b=json.loads((ROOT/'ceiling170.json').read_text())
        self.assertEqual(b['measured_shapes']['qkv_proj']['shape'],['10448x10752'])
        self.assertEqual(b['measured_shapes']['fc1']['shape'],['10448x14336'])
    def test_r169_error_is_admitted(self):
        b=json.loads((ROOT/'ceiling170.json').read_text())
        self.assertIn('10752',b['what_r169_got_wrong'])
        self.assertIn('19.4',b['what_r169_got_wrong'])
    def test_free_fa_is_not_enough(self):
        b=json.loads((ROOT/'ceiling170.json').read_text())
        self.assertGreater(b['ten_second_arithmetic']['if_fa_were_free_s'],10.0)
    def test_required_rate_exceeds_measured(self):
        b=json.loads((ROOT/'ceiling170.json').read_text())
        r=b['the_required_fa_rate']
        self.assertGreater(r['fa3x_tflops_per_card'],r['best_measured_gemm_tflops_per_card'])
    def test_conclusion_is_honest(self):
        b=json.loads((ROOT/'ceiling170.json').read_text())
        self.assertIn('likely unreachable',b['honest_conclusion'])
        self.assertIn('11.4 s',b['honest_conclusion'])
    def test_audit_direction_closed(self):
        b=json.loads((ROOT/'ceiling170.json').read_text())
        self.assertEqual(len(b['closing_of_the_audit_direction']),3)

if __name__=='__main__':unittest.main()
