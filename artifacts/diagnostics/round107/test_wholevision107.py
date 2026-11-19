"""Whole vision TP8 stage includes real inputs, all blocks, Float HCCL and 5 full clocks."""
from pathlib import Path
import ast
import unittest
ROOT=Path(__file__).resolve().parent


class WholeVision107Contracts(unittest.TestCase):
    def test_four_cases_full27_DS_merger_and_original_half_gold(self):
        s=(ROOT/'vision_tp_probe107.py').read_text();ast.parse(s)
        for token in('assert size==8','assert len(layers)==27',"('corrected_TP8_BC128',True,False,True,None)",
                     'assert all(p.dtype==torch.float16 for p in vision.parameters())',
                     "'full_encoder_precision_accepted':False","'numerical_tolerance':None"):
            self.assertIn(token,s)
        self.assertIn("value=h.get_tensor(name).to(torch.float16)",s)
        self.assertIn("proof['parameter_conversion']['visual.'+key]['reference_sha256']",s)
    def test_first_forward_counts_map_api_names_and_do_not_reprime108_weights(self):
        s=(ROOT/'vision_tp_probe107.py').read_text()
        self.assertIn("fields={'attention_calls':'enabled_attention_calls','mlp_calls':'enabled_mlp_calls'}",s)
        self.assertIn("if key!='primed_weight_modules'",s)
        self.assertIn("delta['primed_weight_modules']=after_stats['primed_weight_modules']",s)
        self.assertIn('assert delta==expected_counts',s)
    def test_stock_vision_timers_include_gpu_group_work_no_hooks_or_fake_inputs(self):
        s=(ROOT/'vision_tp_probe107.py').read_text()
        self.assertIn('torch.distributed.barrier();torch.npu.synchronize();begin=time.perf_counter()',s)
        self.assertIn('with torch.inference_mode():out,out_ds=vision(pixels,grids)',s)
        self.assertIn("'native_recovery'",s)
        self.assertIn('before==previous.parameters_sha(vision)',s)
        self.assertNotIn('weight.float()',s)
    def test_prebuild_small_and_full_shapes_sequential_and_float_row_destination(self):
        s=(ROOT/'prebuild107.py').read_text();ast.parse(s)
        self.assertIn('for m in(128,8064):',s)
        self.assertIn("('row',256,1152)",s);self.assertIn("('row',640,1152)",s)
        self.assertIn("meta['dst_dtype']==('float16'if mode=='column'else'float32')",s)
    def test_typed_collectives_and_cpu_original_half_reference_in_primitive(self):
        s=(ROOT/'gemm_tp_probe107.py').read_text();ast.parse(s)
        self.assertIn('assert partial.dtype==torch.float32',s)
        self.assertIn('summed=group.all_reduce(partial);assert summed.dtype==torch.float32',s)
        self.assertIn('gold=torch.nn.functional.linear(x,w,bias)',s)
        self.assertIn('group=group.cpu_group',s)
        self.assertNotIn('w.float()',s)


if __name__=='__main__':unittest.main()
