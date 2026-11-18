"""Late-stage precision isolation contracts,not full encode or a new teacher."""
import ast
import hashlib
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent


class RebaseContracts(unittest.TestCase):
    def test_exact_original_CPUlayer24_to_layer25_and_final49(self):
        text=(ROOT/'late_rebase86.py').read_text();ast.parse(text)
        self.assertIn("input_file=previous.CPU_ROOT/'FP16_layer24.pt'",text)
        self.assertIn("gold_file=previous.CPU_ROOT/'FP16_layer49.pt'",text)
        self.assertIn("r['CPU_layer24_transfer']['raw_bits_equal']",text)
        self.assertIn('for i in range(25,50)',text)
        self.assertNotIn('._deepstack_process(',text);self.assertNotIn('.vision(',text)
    def test_parameters_only_original_late_names_and_Half(self):
        text=(ROOT/'late_rebase86.py').read_text()
        self.assertIn("if 25<=int(name.split('.')[3])<50",text)
        self.assertIn('p.dtype==torch.float16 for p in encoder.parameters()',text)
        self.assertIn('expected_layer_count=50',text)
        self.assertIn('cfg.text_config.intermediate_size==25600',text)
    def test_three_independent_cases_repeat_and_native_recovery(self):
        text=(ROOT/'late_rebase86.py').read_text()
        self.assertIn("('native',False,False)",text)
        self.assertIn("('accurate_attention_native_rows',True,False)",text)
        self.assertIn("('accurate_attention_FP32_rows',True,True)",text)
        self.assertIn("repeat['raw_bits_equal']",text)
        self.assertIn("r['native_recovered']['raw_bits_equal']",text)
    def test_actual_MLP_capture_before_leaf_and_parameter_hash_after(self):
        text=(ROOT/'late_rebase86.py').read_text()
        self.assertIn("layers['25'].mlp.register_forward_pre_hook(capture_mlp)",text)
        self.assertIn("run_controls(encoder.text_model.layers['25'],inp,capture['mlp_input']",text)
        self.assertIn("r['original_parameters_unchanged']",text)
        self.assertIn("'not_full_encoder_timing':True",text)
        self.assertIn("'full_encoder_precision_accepted':False",text)
    def test_previous_executed_sources_stay_immutable(self):
        pins={ROOT.parent/'round85/attention_engine85.py':'ead89fe1247e24019ccaa5d7c4467d484691e18017aaf856f59f4d2cdba2ceb1',
              ROOT.parent/'round84/row_fp32_adapter_fixed84.py':'a169e7a89a187450a52182ae5f66fee6d9d4fdbe2d2ef080803a2b51e2d5b664',
              ROOT.parent/'round79/candidate/candidate_encoder79.py':'2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78'}
        for p,want in pins.items():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),want)


if __name__=='__main__':unittest.main()
