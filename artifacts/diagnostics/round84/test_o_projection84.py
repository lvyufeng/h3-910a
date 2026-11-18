"""Guard actual attention row-projection evidence without declaring full-model acceptance."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class OProjectionEvidence(unittest.TestCase):
    def test_missing_model_row_shape_was_actually_executed(self):
        r=json.loads((ROOT/'results/o_row01/blocked_o_row_result83.json').read_text())
        self.assertTrue(r['completed']);self.assertTrue(r['kernel_launch_executed'])
        self.assertEqual(r['logical_geometry'],{'M':2043,'K':1024,'N':5120})
        self.assertTrue(r['kernel_repeat_vs_first']['raw_bits_equal'])
        self.assertEqual(r['padded_output_rows_max_abs'],0)
        self.assertLess(r['kernel_FP32_vs_accumulator_diagnostic']['relative_RMS'],1e-6)
        self.assertEqual(r['build']['dst_dtype'],'float32');self.assertEqual(r['build']['block_dim'],30)
    def test_actual_CPU_half_teacher_and_scope_are_retained(self):
        r=json.loads((ROOT/'results/o_row01/blocked_o_row_result83.json').read_text())
        self.assertIn('Half input,Half weight',r['CPU_reference_operator'])
        self.assertTrue(r['not_whole_encoder_timing']);self.assertFalse(r['full_encoder_precision_accepted'])
        self.assertFalse(r['whole_goal_complete']);self.assertIsNone(r['precision_acceptance_threshold_chosen'])


if __name__=='__main__':unittest.main()
