"""Full streaming encoder source-only tests with measured visual identity controls."""
from pathlib import Path
import ast
import json
import unittest
ROOT=Path(__file__).resolve().parent


class Full100Contracts(unittest.TestCase):
    def test_true_Half_full27plus50_graph_stock_barrier_outside_timer(self):
        s=(ROOT/'full_encoder100.py').read_text();ast.parse(s)
        for token in('assert size==8','assert all(p.dtype==torch.float16 for p in encoder.parameters())','assert len(layers)==50 and len(vislayers)==27',
                     'with torch.inference_mode():warm=stock()',"'full_encoder_precision_accepted':False","'numerical_tolerance':None",'CPU_text_RoPE_cache_not_installed'):
            self.assertIn(token,s)
        self.assertIn('torch.distributed.barrier()\n                torch.npu.synchronize();start=time.perf_counter()',s)
        self.assertIn('from flash_engine99 import FlashVisionAttention99',s)
    def test_actual_R99_all27_DeepStack_merger_SHA_gate(self):
        expected=json.loads((ROOT/'expected_vision99.json').read_text())
        actual=json.loads((ROOT.parent/'round99/results/vision01/vision99.json').read_text());self.assertTrue(actual['completed'])
        for name,row in expected.items():
            self.assertEqual(row['merged_SHA256'],actual['cases'][name]['merged_vs_CPU']['npu']['sha256'])
            self.assertEqual(len(row['raw27_SHA256']),27);self.assertEqual(len(row['DeepStack_SHA256']),3)
        s=(ROOT/'full_encoder100.py').read_text()
        self.assertIn("==control['merged_SHA256']",s);self.assertIn("==control['raw27_SHA256']",s);self.assertIn("==control['DeepStack_SHA256']",s)
    def test_four_repeat_recovery_cases_and_thirteen_binaries(self):
        s=(ROOT/'full_encoder100.py').read_text()
        for name in('native','corrected_vision_native_text','corrected_stream_vision_native_text','corrected_stream_vision_precise_text'):
            self.assertIn("('"+name+"',",s)
        self.assertIn('for _ in range(5):',s)
        self.assertIn("assert r['native_recovery']['raw_bits_equal']",s)
        self.assertIn("assert r['parameter_and_fixture_SHA_unchanged']",s)
        pre=(ROOT/'prebuild100.py').read_text();ast.parse(pre)
        self.assertIn("'all13objects_SHA_verified_before_8rank_model_priming':True",pre)
        self.assertIn('for meta in text+vision+[erf,flash]:',pre)


if __name__=='__main__':unittest.main()
