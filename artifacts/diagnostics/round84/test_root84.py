"""Guard real FP32 row A/B tests and unchanged CPU FP16/old native controls."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class FullRowABContracts(unittest.TestCase):
    def test_original_R82_driver_is_immutable_and_new_root_unique(self):
        old=ROOT.parent/'round82/condition_isolation82.py'
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),'7ea7d7e22bf359a8c47eb16d9f576015172eb461c1d2ef5168aa6fb4fa9f15b9')
        text=(ROOT/'full_row_fp3284.py').read_text();ast.parse(text)
        self.assertIn('/tmp/h3_round84_row_fp3201',text)
    def test_one_real_CPU_input_raw_layer_is_compared_before_all50(self):
        text=(ROOT/'full_row_fp3284.py').read_text()
        self.assertIn('layer0 = encoder.text_model.layers[0]',text)
        self.assertIn("'native_Half_partial_vs_CPU': compare(native_layer0,layer_references[0])",text)
        self.assertIn("'FP32_partial_vs_CPU': compare(fp32_layer0,layer_references[0])",text)
        self.assertIn('rawdecoderlayer0beforeDeepStack0',text)
    def test_same_CPU_conditions_and_native_recovery_hashes_are_required(self):
        text=(ROOT/'full_row_fp3284.py').read_text()
        for name in('native_rows_CPU_condition','FP32_rows_CPU_condition','FP32_rows_native_condition'):
            self.assertIn(name,text)
        self.assertIn('851a626fac934eb95ac9e370d6dd24f9ab936711ecef73d0961ed694fa7dc059',text)
        self.assertIn('e34340b57f0a800cfd3c479a91dc7ea55ee86be934d07a87d91d296eb9678a5e',text)
        self.assertIn('native encoder output did not recover after FP32 row experiment',text)
    def test_enabled_stock_path_is_compared_to_enabled_reference(self):
        text=(ROOT/'full_row_fp3284.py').read_text()
        self.assertIn('first_hidden_for_stock = fp32_native_reference',text)
        self.assertIn('compare(stock, first_hidden_for_stock)',text)
        self.assertNotIn('compare(stock, first_hidden).get(',text)
        self.assertIn('private FP16 candidate encode_ids with FP32 row GEMM+HCCL',text)
    def test_caches_outside_warm_and_repetition_does_not_become_accuracy_pass(self):
        text=(ROOT/'full_row_fp3284.py').read_text()
        self.assertLess(text.index('row_adapter.prime(encoder.text_model)'),text.index('for call in range(WARM_CALLS)'))
        self.assertIn("'full_encoder_precision_accepted': False",text)
        self.assertIn("'whole_goal_complete': False",text)
        self.assertIn('row_adapter.uninstall()',text)
    def test_o_projection_probe_adds_only_one_real_supported_shape(self):
        text=(ROOT/'probe_o_row84.py').read_text();ast.parse(text)
        self.assertIn("'o_row':(2043,1024,5120)",text)
        self.assertIn('import blocked_dynamic_steps83 as kernel',text)
        self.assertIn('cpu_half=torch.nn.functional.linear(ac,wc)',text)


if __name__=='__main__':unittest.main()
