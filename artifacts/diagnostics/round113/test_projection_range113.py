"""Test overflow controls without executing any tensor runtime."""
from pathlib import Path
import ast
import json
import unittest
ROOT=Path(__file__).resolve().parent


class Range113Contracts(unittest.TestCase):
    def test_cpu_original_failed_at_projection_not_a_reference_pass(self):
        r=json.loads((ROOT.parent/'round111/results/cpu03/CPU_conditioning_failure111.json').read_text())
        self.assertFalse(r['completed']or r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertEqual(r['failure']['stage'],'forward')
        self.assertIn('stage projected contains non-finite',r['failure']['message'])
        self.assertEqual(r['parameter_dtypes'],['torch.float16'])
        self.assertEqual(len(r['parameters']),19)
    def test_original_only_Half_weights_with_actual_NPU_projection_class(self):
        s=(ROOT/'projection_range113.py').read_text();ast.parse(s)
        self.assertIn('params_dtype=torch.float16,quant_config=None',s)
        self.assertIn('original_module(x)[0]',s)
        self.assertIn('bits_sha(original)==bits_sha(F.linear(x,w,b))',s)
        self.assertNotIn('w.float()',s);self.assertNotIn('w.to(torch.float32)',s)
        self.assertNotIn('torch.bfloat16',s)
    def test_power2_control_records_lost_inputs_bias_and_finite_recast_not_teacher(self):
        s=(ROOT/'projection_range113.py').read_text()
        self.assertIn('for exponent in(1,2,3,4):',s)
        self.assertIn('projected=F.linear(scaled_x,w,scaled_b)',s)
        self.assertIn('restored=projected.float()*scale',s)
        self.assertIn('Half_input_values_changed_by_divide_restore',s)
        self.assertIn('Half_bias_values_changed_by_divide_restore',s)
        self.assertIn('finite_original_Half_recast_bits_equal',s)
    def test_no_clamp_or_numerical_waiver_and_capture_pins_unchanged(self):
        s=(ROOT/'projection_range113.py').read_text()
        self.assertNotIn('nan_to_num',s);self.assertNotIn('.clamp(',s)
        self.assertIn("'full_encoder_precision_accepted':False",s)
        self.assertIn("'numerical_tolerance':None",s)
        self.assertIn("'controls_are_NOT_original_four_stage_reference':True",s)
        self.assertIn("file_sha(pin['path'])==pin['file_SHA256']",s)
    def test_nonfinite_data_are_counted_not_erased_and_sign_gate_required(self):
        s=(ROOT/'projection_range113.py').read_text()
        self.assertIn('positive_Inf_count',s);self.assertIn('negative_Inf_count',s);self.assertIn('NaN_count',s)
        self.assertIn('torch.isposinf(original_cpu)&torch.isposinf(recast)',s)
        self.assertIn('torch.isneginf(original_cpu)&torch.isneginf(recast)',s)
    def test_actual_input_payloads_and_geometry_do_not_substitute_cached_hidden(self):
        pins=json.loads((ROOT.parent/'round111/expected_payload_inputs111.json').read_text())
        self.assertEqual(len(pins['inputs']),3)
        self.assertEqual(pins['inputs']['CPU_original']['tensor_key'],'pre_final_norm')
        self.assertEqual(pins['inputs']['corrected_VisionTP8_native_text']['tensor_key'],'selected50')
        s=(ROOT/'projection_range113.py').read_text();self.assertIn('x.shape==(1,2043,5120)',s)
        self.assertNotIn('encode_ids',s)


if __name__=='__main__':unittest.main()
