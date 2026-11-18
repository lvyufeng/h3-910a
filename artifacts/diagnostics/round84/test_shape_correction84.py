"""The official checkpoint geometry is authoritative,not an old planning assumption."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('row_fp32_fixed84_shape_tests',ROOT/'row_fp32_adapter_fixed84.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ActualShapeControls(unittest.TestCase):
    def test_original_checkpoint_config_and_tensors_confirm_actual_down_width(self):
        r=json.loads((ROOT/'actual_shape_receipt_fixed84.json').read_text())
        self.assertEqual(r['config_sha256'],'d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427')
        self.assertEqual(r['text_config']['intermediate_size'],25600)
        self.assertEqual(r['weights']['model.language_model.layers.0.mlp.down_proj.weight']['shape'],[5120,25600])
        self.assertEqual(module.SUPPORTED_ROW_SHAPES,((1024,5120),(3200,5120)))
        self.assertEqual(module.EXPECTED_CACHED_WEIGHT_BYTES,module.expected_cached_weight_bytes())
        self.assertEqual(module.expected_cached_weight_bytes(),2162688000)
    def test_failclosed_attempt_and_its_executed_sources_remain_preserved(self):
        for rank in range(8):
            r=json.loads((ROOT/'results/row_fp3201/ranks'/('tp8_rank%d.json'%rank)).read_text())
            self.assertIn('K=3200',r['failure']['message'])
            self.assertIn('3456',r['failure']['message'])
            self.assertEqual(r['row_adapter_final']['stats']['kernel_launches'],0)
        old=ROOT/'results/row_fp3201/full_row_fp3284.py'
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),r['source_sha256'])
        state=json.loads((ROOT/'failed_state84.json').read_text());self.assertFalse(state['matching_processes'])
    def test_actual_down_primitive_really_executed_and_repeated(self):
        r=json.loads((ROOT/'results/actual_down01/blocked_actual_down_result83.json').read_text())
        self.assertEqual(r['logical_geometry'],{'M':2043,'K':3200,'N':5120})
        self.assertTrue(r['completed']and r['kernel_launch_executed'])
        self.assertEqual(r['build']['accumulation']['k_tiles'],25)
        self.assertTrue(r['kernel_repeat_vs_first']['raw_bits_equal']);self.assertEqual(r['padded_output_rows_max_abs'],0)
        self.assertLess(r['kernel_FP32_vs_accumulator_diagnostic']['relative_RMS'],2e-6)
    def test_corrected_driver_uses_unique_root_and_real_activation_packing_controls(self):
        text=(ROOT/'full_row_fp32_fixed84.py').read_text();ast.parse(text)
        self.assertIn('/tmp/h3_round84_row_fp3202',text)
        self.assertIn('from row_fp32_adapter_fixed84 import install',text)
        self.assertIn("('down_proj',encoder.text_model.layers[0].mlp.down_proj,3200)",text)
        self.assertIn('row_adapter.controls(',text)
        self.assertIn("'native_after_row_AB'",text)


if __name__=='__main__':unittest.main()
