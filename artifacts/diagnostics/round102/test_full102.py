"""Prepared full BC128 TP8 graph: actual pixels, oracle identity, no timed hidden cache."""
from pathlib import Path
import ast
import unittest
ROOT=Path(__file__).resolve().parent


class Full102Contracts(unittest.TestCase):
    def test_Half_full27plus50_entry_and_barrier_before_stockclock(self):
        s=(ROOT/'full_encoder102.py').read_text();ast.parse(s)
        for token in('assert size==8','assert all(p.dtype==torch.float16 for p in encoder.parameters())','assert len(layers)==50 and len(vislayers)==27',
                     'with torch.inference_mode():warm=stock()',"'full_encoder_precision_accepted':False","'numerical_tolerance':None",
                     'from flash_engine101 import FlashVisionAttention101','CPU_text_RoPE_cache_not_installed'):
            self.assertIn(token,s)
        self.assertIn('torch.distributed.barrier()\n                torch.npu.synchronize();start=time.perf_counter()',s)
    def test_output_saved_before_repeat_stock_never_used_in_inference(self):
        s=(ROOT/'full_encoder102.py').read_text()
        save=s.index('torch.save(');clock=s.index('start=time.perf_counter()')
        self.assertLess(save,clock)
        self.assertIn("'selected50':first.detach().contiguous()",s)
        self.assertIn('if rank==0:',s)
        self.assertIn('never used',s)
        self.assertNotIn('torch.load(saved_path',s)
        self.assertIn("'actual_NPU_output_saved_outside_timer'",s)
    def test_R101_visual_hashes_fullrepeat_and_native_recovery_guards(self):
        s=(ROOT/'full_encoder102.py').read_text()
        for token in("==control['merged_SHA256']","==control['raw27_SHA256']","==control['DeepStack_SHA256']",
                     "assert r['native_recovery']['raw_bits_equal']","assert r['parameter_and_fixture_SHA_unchanged']"):
            self.assertIn(token,s)
        self.assertIn('for _ in range(5):',s)
        for name in('native','corrected_vision_native_text','corrected_bc128_vision_native_text','corrected_bc128_vision_precise_text'):
            self.assertIn("('"+name+"',",s)
    def test_serial13objectprebuild_and_same5GMkernel(self):
        s=(ROOT/'prebuild102.py').read_text();ast.parse(s)
        self.assertIn('import flash_vision101',s)
        self.assertIn("'all13objects_SHA_verified_before_8rank_model_priming':True",s)
        self.assertIn('for meta in text+vision+[erf,flash]:',s)
        self.assertIn("assert meta['block_dim']==30",s)


if __name__=='__main__':unittest.main()
