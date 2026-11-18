"""Full TP8 encoder control source and real-R97 visual SHA gates."""
from pathlib import Path
import ast
import hashlib
import json
import unittest
ROOT=Path(__file__).resolve().parent


class FullGraph98Contracts(unittest.TestCase):
    def test_original_Half_full27plus50_graph_and_stock_clock(self):
        s=(ROOT/'full_encoder98.py').read_text();ast.parse(s)
        for token in('assert size==8','assert all(p.dtype==torch.float16 for p in encoder.parameters())',
                     'assert len(layers)==50 and len(vislayers)==27','encoder.encode_ids(',
                     "'full_encoder_precision_accepted':False","'numerical_tolerance':None",
                     "'CPU_text_RoPE_cache_not_installed':True"):
            self.assertIn(token,s)
        self.assertNotIn('FloatRowAdapter',s)
        self.assertIn('torch.distributed.barrier()\n                torch.npu.synchronize();start=time.perf_counter()',s)
        self.assertIn('with torch.inference_mode():warm=stock()',s)
        self.assertIn("'not_full_encoder_timing':True",s)
    def test_all_newvision_boundaries_DeepStack_and_merger_match_actual_R97(self):
        s=(ROOT/'full_encoder98.py').read_text()
        self.assertIn("assert case['vision_merged_vs_CPU']['npu']['sha256']==control['merged_SHA256']",s)
        self.assertIn("==control['raw27_SHA256']",s)
        self.assertIn("==control['DeepStack_SHA256']",s)
        expected=json.loads((ROOT/'expected_vision97.json').read_text())
        actual=json.loads((ROOT.parent/'round97/results/leaf01/vision97.json').read_text());self.assertTrue(actual['completed'])
        for name,row in expected.items():
            self.assertEqual(row['merged_SHA256'],actual['cases'][name]['merged_vs_CPU']['npu']['sha256'])
            self.assertEqual(len(row['raw27_SHA256']),27);self.assertEqual(len(row['DeepStack_SHA256']),3)
    def test_four_repeated_actual_cases_and_native_restore_original_parameters(self):
        s=(ROOT/'full_encoder98.py').read_text()
        for name in('native','corrected_vision_native_text','corrected_NZ_vision_native_text','corrected_NZ_vision_precise_text'):
            self.assertIn("('"+name+"',",s)
        self.assertIn('for _ in range(5):',s)
        self.assertIn("assert case['repeat_vs_first']['raw_bits_equal']and case['stock_vs_equivalent']['raw_bits_equal']",s)
        self.assertIn("assert r['native_recovery']['raw_bits_equal']",s)
        self.assertIn("assert r['parameter_and_fixture_SHA_unchanged']",s)
        self.assertIn('if vision_attention is not None:vision_attention.uninstall()',s)
    def test_18_cached_objects_prebuild_no_modelinputs(self):
        s=(ROOT/'prebuild98.py').read_text();ast.parse(s)
        self.assertIn("'vision_softmax':vision_softmax",s)
        self.assertIn("'no_FP16_model_input_executed':True",s)
        self.assertIn('for meta in text+vision+[erf]+vision_qk+vision_pv+vision_softmax:',s)
        self.assertIn("assert meta['block_dim']==30",s)


if __name__=='__main__':unittest.main()
