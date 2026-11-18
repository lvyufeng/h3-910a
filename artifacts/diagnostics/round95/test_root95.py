"""Full TP8 CPU-exact RoPE table A/B does not cache actual imageconditioning."""
from pathlib import Path
import ast
import unittest
ROOT=Path(__file__).resolve().parent


class FullEncoderContracts(unittest.TestCase):
    def test_CPU_Rope_prime_metadata_only_and_native_baseline_gates(self):
        s=(ROOT/'full_encoder95.py').read_text();ast.parse(s)
        self.assertIn('metadata_x=torch.empty((1,2043,5120)',s)
        self.assertIn('rope_cache.prime(metadata_x,metadata_positions)',s)
        self.assertIn("['extra_bytes']==1046016",s)
        self.assertIn("['native_probe']['matches_R94_observed_native_SHA']",s)
        self.assertIn("['teacher_weights']==0",s)
        self.assertIn('902ed7eff738cdbce35509ebe8f97e21322db6e2b20c54feff70a05b16191198',s)
        self.assertIn('68d2d166ab3cd03eca41e179fc3ac4419b617448efa8389555f9125759fbc11a',s)
    def test_six_paired_modes_no_NR_or_FP32_row_default(self):
        s=(ROOT/'full_encoder95.py').read_text()
        for pair in("('native',False,False,False)","('native_CPU_rope',False,False,True)","('corrected_vision_native_text',True,False,False)","('corrected_vision_native_text_CPU_rope',True,False,True)","('corrected_vision_precise_text',True,True,False)","('corrected_vision_precise_text_CPU_rope',True,True,True)"):
            self.assertIn(pair,s)
        self.assertNotIn('row_fp32_adapter',s);self.assertNotIn('Newton',s)
        self.assertIn("'bit_native_geometry_cache_always':True",s)
    def test_true_stock_encoder_inside_clock_not_CPUcondition_controls(self):
        s=(ROOT/'full_encoder95.py').read_text()
        start=s.index('            for _ in range(5):');end=s.index("            case['median_stock_s']",start)
        block=s[start:end]
        self.assertIn('torch.distributed.barrier()',block)
        self.assertIn('with torch.inference_mode():warm=stock()',block)
        self.assertNotIn('cpu_on_npu',block)
        self.assertIn('return encoder.encode_ids(payload[\'input_ids\'],pixel_values=',s)
        self.assertIn('len(layers)==50 and len(vislayers)==27',s)
    def test_geometry_unchanged_cacheSHA_repeat_and_native_recovery(self):
        s=(ROOT/'full_encoder95.py').read_text()
        self.assertIn('layer_hashes[name][50:]==layer_hashes[old][50:]',s)
        self.assertIn("case['repeat_vs_first']['raw_bits_equal']and case['stock_vs_equivalent']['raw_bits_equal']",s)
        self.assertIn("r['native_recovery']['raw_bits_equal']",s)
        self.assertIn("r['CPU_rope_final']['cached'].values()",s)
        self.assertIn("'full_encoder_precision_accepted':False",s)
        self.assertIn("'numerical_tolerance':None",s)


if __name__=='__main__':unittest.main()
