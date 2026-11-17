"""Validate actual FP32-output TIK micro execution and the separate bias rejection."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class ActualFp32Output(unittest.TestCase):
    def test_new_kernel_really_built_registered_and_executed(self):
        r=json.loads((ROOT/'runtime06'/'probe_fp32_out.json').read_text())
        self.assertTrue(r['build']['built']);self.assertTrue(r['registration']['handle_nonnull'])
        self.assertTrue(r['kernel_launch_executed']);self.assertTrue(r['completed'])
        self.assertEqual(r['build']['dst_dtype'],'float32');self.assertEqual(r['build']['GM_argument_count'],5)
        self.assertTrue(r['cases']['repeat_vs_first']['raw_bits_equal'])
        self.assertEqual(hashlib.sha256((ROOT/'tik_gemm_legacy5_82.py').read_bytes()).hexdigest(),r['builder_sha256'])
        self.assertEqual(hashlib.sha256((ROOT/'run_tik_legacy_probe82.py').read_bytes()).hexdigest(),r['source_sha256'])
    def test_FP32_precision_and_original_CPU_Half_control_are_separate(self):
        r=json.loads((ROOT/'runtime06'/'probe_fp32_out.json').read_text())
        self.assertAlmostEqual(r['cases']['vs_fp32_golden']['relative_rms'],4.081047061834728e-7)
        self.assertAlmostEqual(r['cases']['kernel_cast_to_half_vs_CPU_FP16_linear']['relative_rms'],1.3247272195004748e-5)
        self.assertEqual(r['cases']['kernel_cast_to_half_vs_CPU_FP16_linear'],r['cases']['native_half_vs_CPU_FP16_linear'])
        self.assertTrue(r['FP32_numpy_golden_is_only_accumulator_diagnostic'])
        self.assertFalse(r['full_encoder_precision_accepted']);self.assertTrue(r['not_a_speed_claim'])
    def test_bias_API_rejection_is_not_called_an_executed_kernel_or_pass(self):
        r=json.loads((ROOT/'runtime06'/'probe_fp32_bias_fp16_out.json').read_text())
        self.assertFalse(r['kernel_launch_executed']);self.assertNotIn('completed',r)
        self.assertEqual(r['failure']['type'],'SystemExit')
        self.assertIn('matmul param bias, not support Ascend910',(ROOT/'runtime06'/'bias.log').read_text())
    def test_source_library_reuse_did_not_change_the_production_launcher(self):
        r=json.loads((ROOT/'runtime06'/'probe_fp32_out.json').read_text())
        self.assertTrue(r['launcher']['reused_existing']);self.assertFalse(r['launcher']['compiled_by_this_probe'])
        self.assertEqual(r['launcher']['sha256'],'bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043')
        self.assertEqual(r['build']['unused_aux_inputs'],2)


if __name__=='__main__':unittest.main()
