"""Source-only BC256 host and poison-padding smoke contracts."""
from pathlib import Path
import ast
import unittest
import flash_engine103 as engine
ROOT=Path(__file__).resolve().parent


class Root103Contracts(unittest.TestCase):
    def test_query4032_stays_unpadded_KV4096_exactHalf(self):
        s=(ROOT/'flash_engine103.py').read_text();ast.parse(s)
        self.assertIn('rows=4032 if len(packed)==0 else 4096',s)
        self.assertIn('padded_v.reshape(16,256,16,80).permute(0,1,3,2)',s)
        self.assertIn("'KV_tile':256,'query_tile':32",s)
        self.assertIn('contiguous().unsqueeze(0)',s)
        self.assertEqual(engine.MAX_SCORE_BOUND,16384.0)
    def test_cheap_smoke_true192_pad256_query64_and_poison_keys(self):
        s=(ROOT/'bc256_smoke103.py').read_text();ast.parse(s)
        for token in('batch=2,m=64,n=192,d=72','(1,2,64,72)',
                     'kp[:,192:,:72].fill_(20)','vp[:,192:,:72].fill_(-23)',
                     "assert c['poison_vs_zero_pad']['raw_bits_equal']","if case=='constant_uniform':assert metric['raw_bits_equal']"):
            self.assertIn(token,s)
    def test_real_leaf_oldBC128_SHA_strictshape_and_metadata_deepcopy(self):
        s=(ROOT/'leaf_compare103.py').read_text()
        self.assertIn('from flash_engine101 import FlashVisionAttention101',s)
        self.assertIn('from flash_engine103 import FlashVisionAttention103',s)
        self.assertIn("expected['R101_BC128_leaf_output_SHAs'][image]",s)
        self.assertIn('candidate_out.shape==old_repeat.shape==new_repeat.shape==(1,16,4032,72)',s)
        self.assertIn('copy.deepcopy(old.prepare())',s)
        self.assertIn("'numerical_tolerance':None",s)


if __name__=='__main__':unittest.main()
