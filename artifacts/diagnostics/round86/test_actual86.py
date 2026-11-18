"""Actual CPU rebasing and same-operand lateHalf leaf results,without a pass waiver."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class ActualRebaseEvidence(unittest.TestCase):
    def test_all8_exact_CPUstate_and_native_recovery(self):
        for rank in range(8):
            r=json.loads((ROOT/'results/late01/ranks'/('tp8_rank%d.json'%rank)).read_text())
            self.assertTrue(r['completed']);self.assertTrue(r['CPU_layer24_transfer']['raw_bits_equal'])
            self.assertTrue(r['native_recovered']['raw_bits_equal']);self.assertTrue(r['original_parameters_unchanged'])
            self.assertTrue(r['original_CPU_layer24_input_unchanged']);self.assertTrue(r['CPU_reference_receipt_unchanged'])
            self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'late_rebase86.py').read_bytes()).hexdigest())
            for case in r['cases'].values():
                self.assertEqual([x['layer']for x in case['raw25to49']],list(range(25,50)))
                self.assertTrue(case['repeat_bits']['raw_bits_equal'])
                self.assertTrue(case['no_DeepStack_or_final_norm_added'])
    def test_reset_reduces_error_but_does_not_make_precision_pass(self):
        s=json.loads((ROOT/'summary86.json').read_text());c=s['rebased_late25to49_cases']
        self.assertAlmostEqual(c['native']['relative_RMS'],.0052134548541348455)
        self.assertAlmostEqual(c['accurate_attention_native_rows']['relative_RMS'],.00721398028774451)
        self.assertAlmostEqual(c['accurate_attention_FP32_rows']['relative_RMS'],.005450426963116193)
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
        self.assertTrue(s['not_full_encoder_timing'])
    def test_Float_SiLU_diagnostic_does_not_improve_actual_native_gate(self):
        s=json.loads((ROOT/'summary86.json').read_text());leaf=s['leaf_rank0']
        a=leaf['silu_on_actual_native_gate']
        self.assertEqual(a['native_vs_cpu_half_on_same_native_gate']['npu']['sha256'],a['float_activation_diagnostic_vs_cpu_half']['npu']['sha256'])
        self.assertLess(a['native_vs_cpu_half_on_same_native_gate']['relative_rms'],1e-6)
        self.assertLess(leaf['input_layernorm']['native_vs_cpu_half']['relative_rms'],1e-5)
        self.assertFalse(leaf['down_proj_local_linear']['collective_executed'])
    def test_parameters_and_leaf_operands_frozen_on_all8(self):
        for rank in range(8):
            r=json.loads((ROOT/'results/late01/ranks'/('tp8_rank%d.json'%rank)).read_text())
            leaf=r['block25_leaf_controls'];self.assertTrue(leaf['frozen_unchanged'])
            self.assertTrue(leaf['same_original_half_source']);self.assertTrue(leaf['no_fp32_weight_shadow'])
            self.assertIsNone(leaf['numerical_tolerance']);self.assertFalse(leaf['full_encoder_precision_accepted'])


if __name__=='__main__':unittest.main()
