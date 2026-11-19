"""Root backend shapes, Float partial-sum order and standalone head2 contracts."""
from pathlib import Path
import ast
import unittest
import vision_gemm_backend107 as backend
import head2_attention107 as head2
ROOT=Path(__file__).resolve().parent


class Root107Contracts(unittest.TestCase):
    def test_original_logical_shards_versus_k128_kernel_padding(self):
        expected=[(1152,432,1152,512),(144,1152,256,1152),(1152,538,1152,640),(538,1152,640,1152)]
        for k,n,kp,np in expected:
            g=backend.geometry(8064,k,n);self.assertEqual((g['m'],g['mp'],g['kp'],g['np']),(8064,8064,kp,np))
            # 8064 is63*128 and needs no additionalMpadding for GEMM; attentionQ4032 remainsunrounded.
            self.assertEqual(g['k'],k);self.assertEqual(g['n'],n)
    def test_row_bias_not_in_each_rank_and_float_GM_before_reduction(self):
        s=(ROOT/'vision_gemm_backend107.py').read_text();ast.parse(s)
        self.assertIn("assert mode!='row'or bias is None",s)
        self.assertIn("dtype=torch.float16 if mode=='column'else torch.float32",s)
        self.assertIn("[a,entry['weight'],self.aux,self.aux,c]",s)
        self.assertIn("'Float_weight_shadows':0",s)
        self.assertIn('flat[:entry[\'m\'],:entry[\'k\']]=input_',s)
        self.assertNotIn('weight.float()',s)
    def test_head2_no_global_candidate_helper_patch_and_strict_shapes(self):
        s=(ROOT/'head2_attention107.py').read_text();ast.parse(s)
        self.assertNotIn('setattr(',s)
        self.assertIn('build_kernel(batch=2,m=4032,n=4032,d=72',s)
        self.assertIn('q.shape==k.shape==v.shape==(1,2,4032,72)',s)
        self.assertIn('padded_v.reshape(2,256,16,80)',s)
        self.assertIn('contiguous().unsqueeze(0)',s)
        self.assertEqual(head2.MAX_SCORE_BOUND,16384.0)
    def test_cheap_probe_covers_all16_heads_both_images_before_full_model(self):
        s=(ROOT/'head2_probe107.py').read_text();ast.parse(s)
        self.assertIn('for rank in range(8):',s)
        self.assertIn("assert same['raw_bits_equal']",s)
        self.assertIn('candidate._vision_sdpa_bf16_compatible is original_helper',s)
        self.assertIn('first.shape==gold.shape==old_slice.shape==(1,2,4032,72)',s)
        self.assertIn("'numerical_tolerance':None",s)


if __name__=='__main__':unittest.main()
