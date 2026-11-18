"""Actual one-cast vision bias evidence and best native-row encoder clock."""
from pathlib import Path
import json
import hashlib
import statistics
import unittest
ROOT=Path(__file__).resolve().parent


class ActualVisionBiasEvidence(unittest.TestCase):
    def test_vision_real_values_repeated_and_native_recovered(self):
        r=json.loads((ROOT/'results/vision01/vision87.json').read_text())
        self.assertTrue(r['completed']and r['original_parameters_unchanged']and r['original_pixels_unchanged'])
        self.assertTrue(r['CPU_reference_SHA_unchanged'])
        self.assertTrue(r['native_recovery']['merged']['raw_bits_equal'])
        self.assertTrue(all(x['raw_bits_equal']for x in r['native_recovery']['deepstack']))
        self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'vision_probe87.py').read_bytes()).hexdigest())
        for case in r['cases'].values():
            self.assertEqual(len(case['raw27_layers']),27)
            self.assertTrue(all(x['merged']['raw_bits_equal']for x in case['repeat_bits']))
    def test_actual_biased_leaf_error_reduced_but_vision_not_complete(self):
        s=json.loads((ROOT/'summary87.json').read_text())
        for leaf in s['actual_vision_biased_leafs'].values():
            self.assertGreater(leaf['native_CPU_Half_relative_RMS']/leaf['custom_CPU_Half_relative_RMS'],30)
            self.assertTrue(leaf['padding_controls']['original_prefix_bit_exact'])
            self.assertTrue(leaf['padding_controls']['zero_only_M_K_padding'])
        self.assertAlmostEqual(s['vision_cases']['FP32_bias_one_Half']['merged_relative_RMS'],.012887294160849183)
        self.assertGreater(s['vision_cases']['FP32_bias_one_Half']['resident_vision_median_s'],s['vision_cases']['native']['resident_vision_median_s'])
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
        self.assertEqual(s['vision_adapter']['cached_weights'],116)
        self.assertEqual(s['vision_adapter']['kernel_launches'],470)
        self.assertEqual(s['vision_adapter']['FP32_weight_shadows'],0)
    def test_best_text_actual_8rank_warm_not_a_precomputed_condition_clock(self):
        rows=[]
        for rank in range(8):
            r=json.loads((ROOT/'results/best_text01/ranks'/('tp8_rank%d.json'%rank)).read_text())
            self.assertTrue(r['completed']and r['frozen_inputs_unchanged'])
            self.assertTrue(r['row_adapter_not_installed']);self.assertNotIn('row_adapter_final',r)
            self.assertTrue(r['stock_encode_ids_timed']and r['timing']['warm_outputs_bit_equal'])
            self.assertTrue(r['native_after_row_AB']['raw_bits_equal']);rows.append(r)
        clocks=[max(r['timing']['samples_synchronized_wall_s_including_d2h'][i]for r in rows)for i in range(3)]
        self.assertAlmostEqual(statistics.median(clocks),2.898743180092424)
        self.assertGreater(statistics.median(clocks),2)
    def test_worker_and_CPU_payload_invariance_verified(self):
        v=json.loads((ROOT/'final_verification87.json').read_text())
        self.assertEqual(len(v['production']),23);self.assertEqual(len(v['installed_api_sources']),13)
        self.assertFalse(v['remaining_workers']);self.assertTrue(v['all_production_reference_and_launcher_unchanged'])
        self.assertTrue(v['/tmp/h3_round78_hf_fp1601/FP16_vision.pt']['unchanged'])


if __name__=='__main__':unittest.main()
