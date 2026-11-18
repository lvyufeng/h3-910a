"""Actual GELU mode, corrected Float guard and complete vision evidence."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class ActualErfEvidence(unittest.TestCase):
    def test_native_None_equals_tanh_but_CPU_semantics_are_distinct(self):
        p=json.loads((ROOT/'results/preflight01/gelu_preflight90.json').read_text())
        self.assertTrue(p['completed'])
        for c in p['leafs'].values():
            self.assertTrue(c['native_none_vs_native_tanh']['raw_bits_equal'])
            self.assertFalse(c['CPU_tanh_vs_CPU_Half_erf']['raw_bits_equal'])
            self.assertTrue(c['original_CPU_Half_output_matches_R89_SHA']and c['original_CPU_and_NPU_inputs_SHA_unchanged'])
    def test_failed_unsupported_Float_relu_is_preserved(self):
        r=json.loads((ROOT/'results/kernel01/kernel_probe90.json').read_text())
        self.assertFalse(r['completed']);self.assertFalse(r['cases'])
        self.assertIn("doesn't support float32 for Instruction vec_relu",(ROOT/'results/kernel01/kernel.log').read_text())
        self.assertNotIn('vec("vec_relu", "k_f", "k_f")',(ROOT/'erf_gelu_fixed90.py').read_text())
    def test_real_erf_kernel_improves_both_CPUHalf_leaves_and_repeats(self):
        p=json.loads((ROOT/'results/preflight01/gelu_preflight90.json').read_text())
        k=json.loads((ROOT/'results/vision02/kernel_probe90.json').read_text())
        self.assertTrue(k['completed']and k['improves_both_real_CPU_erf_leafs'])
        for name,c in k['cases'].items():
            self.assertTrue(c['repeat_bits']['raw_bits_equal']and c['input_SHA_unchanged'])
            self.assertTrue(c['CPU_Half_erf']['finite'])
            if name in p['leafs']:
                self.assertLess(c['CPU_Half_erf']['relative_rms'],p['leafs'][name]['native_none_vs_CPU_Half_erf']['relative_rms'])
    def test_six_full_vision_cases_repeated_params_and_native_recovered(self):
        v=json.loads((ROOT/'results/vision02/vision90.json').read_text())
        self.assertTrue(v['completed']and v['original_parameters_unchanged']and v['pixels_unchanged']and v['CPU_reference_SHA_unchanged'])
        self.assertEqual(len(v['cases']),6)
        for c in v['cases'].values():
            self.assertEqual(len(c['raw27_layers']),27)
            self.assertTrue(all(x['merged']['raw_bits_equal']and all(y['raw_bits_equal']for y in x['deepstack'])for x in c['repeat_bits']))
        self.assertTrue(v['native_recovery']['merged']['raw_bits_equal'])
        self.assertEqual(v['TIK_erf_final']['calls'],32);self.assertEqual(v['TIK_erf_final']['kernel_launches'],32)
        self.assertEqual(v['TIK_erf_final']['patched_module_count'],4)
        self.assertEqual(v['TIK_erf_final']['tanh_gelu_modules_left_untouched'],27)
        self.assertEqual(v['source_SHA256'],hashlib.sha256((ROOT/'vision_probe_fixed90.py').read_bytes()).hexdigest())
        self.assertFalse(v['full_encoder_precision_accepted']);self.assertFalse(v['whole_goal_complete'])
    def test_final_production_CPUgold_and_workers_verified(self):
        v=json.loads((ROOT/'final_verification90.json').read_text())
        self.assertEqual(len(v['production']),23);self.assertEqual(len(v['installed_api_sources']),13)
        self.assertFalse(v['remaining_workers']);self.assertTrue(v['all_production_reference_originalHF_frozeninputs_and_launcher_unchanged'])


if __name__=='__main__':unittest.main()
