"""Actual original-Half two-head and Float-HCCL GEMM evidence before complete TP vision."""
from pathlib import Path
import json
import unittest
ROOT=Path(__file__).resolve().parent


class Actual107Primitive(unittest.TestCase):
    def test_head2_all16_shards_bitwise_match_originalBC128_and_reference(self):
        r=json.loads((ROOT/'results/head2_01/head2_probe107.json').read_text())
        self.assertTrue(r['completed']and r['global_candidate_helper_unchanged']);self.assertEqual(len(r['cases']),16)
        self.assertEqual(r['head2_prepare']['kernel']['block_dim'],30)
        for c in r['cases']:
            self.assertTrue(c['vs_original_BC128_head_slice']['raw_bits_equal']and c['all_repeats_and_inputs_unchanged'])
            self.assertEqual(c['vs_original_CPU_Half_MATH']['npu']['shape'],[1,2,4032,72]);self.assertEqual(len(c['clock_s']),5)
    def test_all8_complete_Half_cpu_controls_float_collectives_and_output_identity(self):
        for rank in range(8):
            r=json.loads((ROOT/('results/gemm_tp01/gemm_tp_rank%d.json'%rank)).read_text());self.assertTrue(r['completed'])
            for name,c in r['cases'].items():
                self.assertTrue(c['all_repeats_and_original_inputs_weights_unchanged']);self.assertTrue(c['vs_CPU_Half_Linear']['finite'])
                self.assertEqual(len(c['clock_s']),5)
                if c['mode']=='row':
                    self.assertEqual(c['kernel']['dst_dtype'],'float32');self.assertEqual(c['counters']['Float_collectives'],7)
                    self.assertEqual(c['counters']['Half_output_casts_after_collective'],7);self.assertTrue(c['all8_output_SHA_equal'])
                else:self.assertEqual(c['kernel']['dst_dtype'],'float16');self.assertEqual(c['counters']['Float_collectives'],0)
            self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete']);self.assertIsNone(r['numerical_tolerance'])
    def test_actual_checkpoint_shape_and_original_compiler_alias_no_A2_fake(self):
        r=json.loads((ROOT/'actual_preflight_fixed107.json').read_text())
        self.assertEqual(r['vision_config']['intermediate_size'],4304);self.assertEqual(r['vision_config']['hidden_size'],1152)
        self.assertEqual(r['soc']['Ascend910B']['SHORT_SOC_VERSION'],'Ascend910')
        self.assertEqual(r['soc']['Ascend910B']['CORE_NUM'],30);self.assertEqual(r['soc']['Ascend910']['CORE_NUM'],32)
        self.assertEqual(r['weight_shapes']['model.visual.blocks.0.attn.qkv.weight'],[3456,1152])


if __name__=='__main__':unittest.main()
