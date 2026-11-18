"""Actual CPU-only late propagation gates and preserved layer43 precision cliff."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class ActualSuffixEvidence(unittest.TestCase):
    def receipt(self):return json.loads((ROOT/'results/suffix01/CPU_suffix_reference93.json').read_text())
    def test_baseline_all25_raw_boundary_SHAs_exact_and_no_finalnorm(self):
        r=self.receipt();old=json.loads((ROOT.parent/'round78/reference.json').read_text())
        self.assertTrue(r['completed']and r['baseline_pass']['matches_R78_all_25_boundaries'])
        self.assertEqual(r['baseline_pass']['mismatched_layers'],[])
        for i in range(25,50):self.assertEqual(r['baseline_pass']['boundaries'][str(i)]['output']['sha256'],old['language_layers'][i]['output']['sha256'])
        self.assertTrue(r['selected50']['baseline_matches_R78_truth_bits'])
        self.assertTrue(r['no_final_norm_after_layer49']and r['no_DeepStack_in_suffix'])
    def test_actual_changed_input_twice_all25_boundaries_identical(self):
        r=self.receipt();self.assertTrue(r['corrected_repeat_bits_equal_all_25_boundaries'])
        a,b=r['corrected_passes'];self.assertEqual(len(a['boundaries']),25)
        for i in range(25,50):
            self.assertEqual(a['boundaries'][str(i)]['output']['sha256'],b['boundaries'][str(i)]['output']['sha256'])
            self.assertTrue(a['boundaries'][str(i)]['output']['finite'])
        self.assertEqual(r['corrected_case']['input_SHA256'],'e232f61b4260b81b08bb0cf3eb96a7df6d0a56ed3c0c82ab9d0ec0f15004a528')
    def test_CPUonly_originalHalf_explains_large_scale_not_NPU_arithmetic_claim(self):
        r=self.receipt();self.assertTrue(r['contains_no_NPU_arithmetic_factor'])
        self.assertTrue(r['no_Float_linear_weights_or_teacher']and r['no_BF16_gold_output_cast'])
        curve=r['output_error_growth']['per_layer'];jump=max(curve,key=lambda row:row['growth_over_previous_boundary'])
        self.assertEqual(jump['layer'],43)
        self.assertAlmostEqual(jump['growth_over_previous_boundary'],15.818678750012088)
        self.assertAlmostEqual(r['selected50']['corrected_vs_R78_truth']['relative_rms'],.17161828722753078)
        self.assertAlmostEqual(r['output_error_growth']['selected50_over_input_amplification'],67.81330741809693)
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete']);self.assertIsNone(r['numerical_tolerance'])
    def test_parameters_inputs_and_all_sources_gates_unchanged(self):
        r=self.receipt();f=r['frozen_after']
        for name in('parameters_unchanged','cpu_raw24_unchanged','r78_receipt_unchanged','r78_multimodal_inputs_unchanged','r78_selected50_unchanged','inputs_file_unchanged','helper_unchanged','source_unchanged'):
            self.assertTrue(f[name])
        self.assertEqual(hashlib.sha256((ROOT/'cpu/suffix93.py').read_bytes()).hexdigest(),r['source_sha256'])
        self.assertEqual(r['original_suffix']['retained_parameter_tensors'],275)
        self.assertEqual(r['original_suffix']['layer_count'],25)
    def test_final_worker_and_old_gold_source_verification(self):
        r=json.loads((ROOT/'final_verification93.json').read_text())
        self.assertEqual(r['remaining_workers'],[]);self.assertEqual(len(r['production']),23)
        self.assertEqual(len(r['installed_api_sources']),13)
        self.assertTrue(r['all_production_reference_originalHF_and_launcher_unchanged'])


if __name__=='__main__':unittest.main()
