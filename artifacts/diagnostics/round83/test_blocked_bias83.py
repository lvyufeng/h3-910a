"""Guard full-sized one-cast bias epilogue derived from the verified micro kernel."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class BlockedBiasContracts(unittest.TestCase):
    def test_bias_is_added_in_FP32_after_all_K_blocks(self):
        source=(ROOT/'blocked_bias_epilogue83.py').read_text();ast.parse(source)
        self.assertIn('init_l1out=(kt==0),bias=None',source)
        self.assertIn('t.tensor_mov(c32,c0,',source)
        self.assertIn('t.vec_add(16,c32[group,:,:]',source)
        self.assertEqual(source.count("t.h_cast(c16,c32,'none')"),1)
        self.assertLess(source.index('t.vec_add('),source.index('t.h_cast('))
    def test_Half_store_units_are_not_FP32_stride_units(self):
        source=(ROOT/'blocked_bias_epilogue83.py').read_text()
        self.assertIn('c16,0,nb//16,mb,0,mp-mb)',source)
        self.assertNotIn('c16,0,nb//16,2*mb',source)
        self.assertIn("'padded_output_rows_are_bias_not_zero':True",source)
    def test_fiveGM_arg_order_and_original_bias_value_contract(self):
        source=(ROOT/'blocked_bias_epilogue83.py').read_text()
        self.assertIn('inputs=[ag,bg,biasg,aux],outputs=[cg]',source)
        probe=(ROOT/'probe_blocked_bias83.py').read_text();ast.parse(probe)
        self.assertIn('bias=bc.float().to(dev)',probe)
        self.assertIn('cpu_half=torch.nn.functional.linear(ac,wc,bc)',probe)
        self.assertIn('reshape(mp,n)[:m].cpu()',probe)
        self.assertIn('GPU_operands_unchanged',probe)
        self.assertIn("'full_encoder_precision_accepted':False",probe)
    def test_tile_step_loop_is_dynamic_but_K_initialization_is_static(self):
        source=(ROOT/'blocked_bias_epilogue83.py').read_text()
        self.assertIn('with t.for_range(0,steps)as step:',source)
        self.assertIn('for kt in range(k//kb):',source)
        self.assertIn("BASE_SHA256='78896adadaf47801dd92d69bdfce394d2628f7885be643dd568647c8b4c8c2ee'",source)


if __name__=='__main__':unittest.main()
