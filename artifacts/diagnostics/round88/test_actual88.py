"""Actual real-head72 attention proofs do not imply whole-vision acceptance."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class ActualHead72Evidence(unittest.TestCase):
    def test_all_four_real_vision_cases_repeat_and_native_recovery(self):
        r=json.loads((ROOT/'results/vision01/vision88.json').read_text())
        self.assertTrue(r['completed']and r['original_parameters_unchanged']and r['original_pixels_unchanged']and r['CPU_reference_SHA_unchanged'])
        self.assertEqual(set(r['cases']),{'native','bias_only','accurate_PV_only','accurate_PV_and_bias'})
        for c in r['cases'].values():
            self.assertEqual(len(c['raw27_layers']),27)
            self.assertTrue(c['merged_vs_CPU']['finite'])
            self.assertTrue(all(x['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in x['deepstack'])for x in c['repeat_bits']))
        self.assertTrue(r['native_recovery']['merged']['raw_bits_equal'])
        self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'vision_probe88.py').read_bytes()).hexdigest())
    def test_actual_CPU_Half_attention_leaf_improves_but_whole_path_slow(self):
        s=json.loads((ROOT/'summary88.json').read_text())
        for c in s['actual_leafs']:
            self.assertGreater(c['native_vs_CPU_Half_MATH_relative_RMS']/c['custom_vs_CPU_Half_MATH_relative_RMS'],5)
            self.assertTrue(c['actual_padding_prefix_and_tail_checks']and c['QKV_SHA_unchanged']and c['repeat_bits'])
        joint=s['cases']['accurate_PV_and_bias'];native=s['cases']['native']
        self.assertAlmostEqual(joint['merged_relative_RMS'],.01193380823019713)
        self.assertGreater(joint['resident_vision_median_s'],2*native['resident_vision_median_s'])
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
        self.assertTrue(s['native_and_bias_only_reproduced_R87_output_SHA'])
    def test_actual_10464_launches_retained_Float_before_Half(self):
        s=json.loads((ROOT/'summary88.json').read_text());stats=s['engine_stats']
        self.assertEqual(stats['calls_enabled'],436);self.assertEqual(stats['kernel_launches'],10464)
        self.assertEqual(stats['calls_rejected'],0)
        self.assertEqual(stats['float_before_half_chunks'],3488)
        self.assertEqual(stats['half_output_chunks'],3488)
        src=json.loads((ROOT/'executed_sources88.json').read_text())
        self.assertEqual(src['files']['accurate_vision_attention88.py']['sha256'],hashlib.sha256((ROOT/'attention/accurate_vision_attention88.py').read_bytes()).hexdigest())
        self.assertEqual(src['files']['FrozenVisionQKV88.pt']['sha256'],s['frozen_capture_SHA256'])
    def test_original_reference_and_workers_verified(self):
        v=json.loads((ROOT/'final_verification88.json').read_text())
        self.assertEqual(len(v['production']),23);self.assertEqual(len(v['installed_api_sources']),13)
        self.assertEqual(v['remaining_workers'],[])
        self.assertTrue(v['all_production_reference_and_launcher_unchanged'])


if __name__=='__main__':unittest.main()
