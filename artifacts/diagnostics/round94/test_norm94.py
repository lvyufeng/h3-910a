"""Same-input Half RMSNorm audit keeps negative Newton controls and no teacher."""
from pathlib import Path
import ast
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class RMSContracts(unittest.TestCase):
    def test_original_Half_inputs_gamma_and_two_rounds(self):
        s=(ROOT/'norm_probe94.py').read_text();ast.parse(s)
        self.assertIn("weight_key='model.language_model.layers.25.input_layernorm.weight'",s)
        self.assertIn('CPU.weight.dtype==NPU.weight.dtype==torch.float16',s)
        self.assertIn('(xf*y).to(torch.float16)*NPU.weight',s)
        self.assertIn("'no_Linear_teacher_or_BF_outputcasts':True",s)
        self.assertNotIn('nn.Linear(',s)
    def test_actual_cases_native_NR1_NR2_are_all_preserved(self):
        r=json.loads((ROOT/'norm94.json').read_text());self.assertTrue(r['completed'])
        self.assertEqual(set(r['cases']),{'CPU_raw24','corrected_vision_native_text'})
        for c in r['cases'].values():
            self.assertEqual(set(c['modes']),{'native','Newton1','Newton2'})
            self.assertTrue(all(v['repeat_bit_identical']for v in c['modes'].values()))
            self.assertGreaterEqual(c['modes']['Newton1']['vs_original_CPU_Half_norm']['relative_rms'],c['modes']['native']['vs_original_CPU_Half_norm']['relative_rms'])
            self.assertGreaterEqual(c['modes']['Newton2']['vs_original_CPU_Half_norm']['relative_rms'],c['modes']['native']['vs_original_CPU_Half_norm']['relative_rms'])
    def test_same_input_native_rsqrt_is_CPU_exact_no_newton_fix_claim(self):
        r=json.loads((ROOT/'norm94.json').read_text())
        for c in r['cases'].values():
            self.assertTrue(c['rsqrt_on_exact_CPU_variance']['raw_bits_equal'])
            self.assertEqual(c['rsqrt_on_exact_CPU_variance']['relative_rms'],0.0)
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertIsNone(r['numerical_tolerance']);self.assertTrue(r['not_full_encoder_timing'])
    def test_actual_executed_source_Half_gamma_and_input_invariants(self):
        r=json.loads((ROOT/'norm94.json').read_text())
        self.assertTrue(r['input_and_Half_gamma_unchanged'])
        self.assertEqual(hashlib.sha256((ROOT/'norm_probe94.py').read_bytes()).hexdigest(),r['source_SHA256'])


if __name__=='__main__':unittest.main()
