"""Guard actual vision shapes and the no-row-cache full-encode measurement."""
from pathlib import Path
import ast
import importlib.util
import unittest
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('vision_adapter87_contract',ROOT/'vision_adapter87.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class VisionAndTimingContracts(unittest.TestCase):
    def test_real_six_vision_geometries_and_zero_only_MLP_padding(self):
        self.assertEqual(m.SHAPES[(4304,1152)],(8064,1152,4352))
        self.assertEqual(m.SHAPES[(1152,4304)],(8064,4352,1152))
        self.assertEqual(m.SHAPES[(4608,4608)],(2016,4608,4608))
        self.assertEqual(m.SHAPES[(5120,4608)],(2016,4608,5120))
        self.assertEqual(len(m.SHAPES),6)
    def test_original_Half_parameters_and_exact_bias_value_contract(self):
        text=(ROOT/'vision_adapter87.py').read_text();ast.parse(text)
        self.assertIn('module.weight.dtype==module.bias.dtype==torch.float16',text)
        self.assertIn('module.bias.detach().float()',text)
        self.assertNotIn('module.weight.float()',text)
        self.assertIn("[a,entry['w'],entry['bias'],self.aux,out]",text)
        self.assertIn('signature(layer.weight)',text)
        self.assertIn('len(modules)==116',text)
    def test_actual_leaf_operator_references_are_CPUHalf_not_Float_weight(self):
        text=(ROOT/'vision_probe87.py').read_text();ast.parse(text)
        self.assertIn('input_.cpu(),module.weight.detach().cpu(),module.bias.detach().cpu()',text)
        self.assertIn('adapter.controls(module,input_)',text)
        self.assertIn("r['original_parameters_unchanged']",text)
        self.assertIn("r['native_recovery']",text)
    def test_best_text_keeps_native_row_without_packed_row_cache(self):
        text=(ROOT/'best_text87.py').read_text();ast.parse(text)
        self.assertNotIn('from row_fp32_adapter_fixed84 import',text)
        self.assertNotIn('row_adapter.prime(',text)
        self.assertNotIn('row_adapter.set_enabled(',text)
        self.assertNotIn('row_adapter.controls(',text)
        self.assertIn('accurate_attention_native_rows_native_condition',text)
        self.assertIn('accurate batch attention+native row GEMM+HCCL',text)
        self.assertIn('compare(stock, first_hidden_for_stock)',text)
    def test_both_drivers_new_roots_and_no_precision_waiver(self):
        for name,root in(('vision_probe87.py','/tmp/h3_round87_vision01'),('best_text87.py','/tmp/h3_round87_best_text01')):
            text=(ROOT/name).read_text();self.assertIn(root,text)
            self.assertIn("'full_encoder_precision_accepted':False" if name.startswith('vision')else"'full_encoder_precision_accepted': False",text)


if __name__=='__main__':unittest.main()
