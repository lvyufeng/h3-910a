"""Guard the failed full TP8 baseline evidence;do not normalize it into a pass."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class FailedTP8Evidence(unittest.TestCase):
    def test_all8_receipts_complete_and_failed_on_repeat(self):
        summary=json.loads((ROOT/'failed_summary80.json').read_text())
        self.assertEqual(len(summary['all_ranks']),8)
        for rank,item in enumerate(summary['all_ranks']):
            p=ROOT/'ranks'/('tp8_rank%d.json'%rank);r=json.loads(p.read_text())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['receipt_sha256'])
            self.assertEqual(r['rank'],rank);self.assertEqual(len(r['language_layers']),50)
            self.assertIn('not bit-identical',r['failure']['message'])
            self.assertFalse(item['repeat_bits_equal']);self.assertTrue(item['all_finite'])
    def test_CPU_reference_and_positions_are_real_but_numeric_gap_not_waived(self):
        summary=json.loads((ROOT/'failed_summary80.json').read_text())
        self.assertTrue(summary['actual_full_TP8_first_pass_executed'])
        self.assertTrue(summary['warmed_encoder_latency_not_measured'])
        self.assertFalse(summary['full_encoder_precision_accepted']);self.assertFalse(summary['whole_goal_complete'])
        for item in summary['all_ranks']:
            self.assertAlmostEqual(item['CPU_selected50_relative_RMS'],.07569804692654716)
            self.assertTrue(item['positions_bits_equal']);self.assertTrue(item['visual_masks_bits_equal'])
            self.assertAlmostEqual(item['repeat_relative_RMS'],.009011207199745425)
    def test_source_not_mutated_after_execution_and_workers_gone(self):
        summary=json.loads((ROOT/'failed_summary80.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'encode_fp16_tp8_80.py').read_bytes()).hexdigest(),summary['source_sha256'])
        status=json.loads((ROOT/'failed_run_state80.json').read_text());self.assertEqual(status['matching_processes'],[])
        self.assertEqual(status['run_finished']['exit_code'],1)
    def test_no_false_production_batch_shape_fix(self):
        text=(ROOT/'encode_fp16_tp8_80.py').read_text();self.assertNotIn('documented production defect',text)
        self.assertIn('removes the singleton batch correctly',text)


if __name__=='__main__':unittest.main()
