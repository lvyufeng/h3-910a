"""Guard full-model KV-only padding experiment before hardware execution."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class KVFullDriverContracts(unittest.TestCase):
    def test_failed_baseline_source_is_unchanged_and_new_root_is_unique(self):
        p=ROOT.parent/'round80'/'tp8'/'encode_fp16_tp8_80.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'6e07e427d4a2afa00b52aedfa7b42930541d0d5376fbbc5fbeddb3cbb9681d89')
        text=(ROOT/'encode_fp16_padded_kv81.py').read_text();ast.parse(text)
        self.assertIn("ROOT = Path('/tmp/h3_round81_full_fp16_pad01')",text)
    def test_original_CPU_FP16_graph_weights_and_outputs_are_fixed(self):
        text=(ROOT/'encode_fp16_padded_kv81.py').read_text()
        self.assertIn('2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78',text)
        self.assertIn('0137a434148366c3dcbf825970274ad0faa036666561d7aecc0170cf136b3fe4',text)
        self.assertIn('repeat_pre_final_norm_bits_equal',text)
        self.assertIn('repeat_post_final_norm_bits_equal',text)
        self.assertIn('FP16_vision.pt',text)
        self.assertNotIn('fresh_encoder_reference.pt',text)
    def test_padding_primed_before_warm_timing_and_removed_in_cleanup(self):
        text=(ROOT/'encode_fp16_padded_kv81.py').read_text()
        self.assertIn('kv_adapter = install(candidate)',text)
        self.assertLess(text.index('kv_adapter.prime(encoder.text_model)'),text.index('for call in range(WARM_CALLS)'))
        self.assertIn('kv_adapter.uninstall()',text)
        self.assertIn("r['kv_padding_final'] = kv_adapter.snapshot()",text)
    def test_embedding_numeric_and_signedzero_check_are_not_hash_only(self):
        text=(ROOT/'encode_fp16_padded_kv81.py').read_text()
        self.assertIn("cmp['embedding_full_numeric']",text)
        self.assertIn("cmp['embedding_sign_zero_differences']",text)
        self.assertIn("if tensor_info(cpu_embedding)['sha256'] != embedding_gold['sha256']",text)
    def test_diagnostic_range_subsets_not_pass_threshold(self):
        text=(ROOT/'encode_fp16_padded_kv81.py').read_text()
        self.assertIn("cmp['range_subset_is_diagnostic_not_an_acceptance_threshold'] = True",text)
        self.assertIn("'text_nonvisual'",text);self.assertIn("'visual'",text)
        self.assertIn("'full_encoder_precision_accepted': False",text)
        self.assertIn("'whole_goal_complete': False",text)
    def test_real_encode_ids_timing_retains_singleton_batch_contract(self):
        text=(ROOT/'encode_fp16_padded_kv81.py').read_text()
        self.assertIn('hidden = stock_forward() if use_stock else run_equivalent',text)
        self.assertIn('removes the singleton batch correctly',text)
        self.assertNotIn('documented production defect',text)
        self.assertNotIn('F.scaled_dot_product_attention =',text)


if __name__=='__main__':unittest.main()
