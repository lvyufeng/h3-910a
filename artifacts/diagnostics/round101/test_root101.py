"""Root host packing/shape/metadata contracts; source only, no Torch/CANN run."""
from pathlib import Path
import ast
import unittest
import flash_engine101 as engine
ROOT=Path(__file__).resolve().parent


class Root101Contracts(unittest.TestCase):
    def test_only_key_value_padding_not_query_and_strict_output_shape(self):
        s=(ROOT/'flash_engine101.py').read_text();ast.parse(s)
        self.assertIn('rows=4032 if len(packed)==0 else 4096',s)
        self.assertIn('padded[:,:4032,:72]=original[0]',s)
        self.assertIn('padded_v.reshape(16,256,16,80).permute(0,1,3,2)',s)
        self.assertIn('q.shape==k.shape==v.shape==(1,16,4032,72)',s)
        self.assertIn('contiguous().unsqueeze(0)',s)
        self.assertIn("'KV_tile':128",s)
    def test_guard_no_halfcumulative_or_hidden_teacher_or_defaultenable(self):
        s=(ROOT/'flash_engine101.py').read_text()
        self.assertEqual(engine.MAX_SCORE_BOUND,16384.0)
        self.assertIn('reshape(-1).amax(dim=0).item()',s)
        self.assertIn('dense.storage_offset()!=0',s)
        self.assertNotIn('clamp(',s);self.assertNotIn('.weight.float()',s)
        self.assertIn('copy.deepcopy(self._meta)',s)
    def test_disabled_delegation_activation_guard_and_witness_survives_uninstall(self):
        class Candidate:
            def __init__(self):self._vision_sdpa_bf16_compatible=lambda q,k,v:('original',q,k,v)
        class Runtime:device=0
        candidate=Candidate();e=engine.FlashVisionAttention101(candidate,Runtime(),ROOT/'unused_source_test')
        self.assertEqual(e.forward(1,2,3),('original',1,2,3))
        with self.assertRaises(RuntimeError):e.set_enabled(True)
        e._meta={'object':{'SHA':'before'}};w=e.snapshot();e._meta['object']['SHA']='after'
        self.assertEqual(w['kernel']['object']['SHA'],'before')
        after=e.uninstall();self.assertEqual(after['kernel']['object']['SHA'],'after')
        self.assertFalse(after['full_encoder_precision_accepted']or after['whole_goal_complete'])
        self.assertIsNone(after['numerical_tolerance'])
    def test_smoke_masks_nonzero_phantomkeys_and_known_exact_constant_result(self):
        s=(ROOT/'bc128_smoke101.py').read_text();ast.parse(s)
        for token in('batch=2,m=128,n=192,d=72','kp[:,192:,:72].fill_(20)','vp[:,192:,:72].fill_(-23)',
                     "assert c['poison_vs_zero_pad']['raw_bits_equal']","if case=='constant_uniform':assert metric['raw_bits_equal']"):
            self.assertIn(token,s)
        self.assertIn('v[0,0].fill_(0.25);v[0,1].fill_(-0.5)',s)
    def test_real_leaf_oldSHA_strictshape_and_no_newtolerance(self):
        s=(ROOT/'leaf_compare101.py').read_text()
        self.assertIn('candidate_out.shape==old_repeat.shape==new_repeat.shape==(1,16,4032,72)',s)
        self.assertIn("expected['R99_streaming_leaf_output_SHAs'][image]",s)
        self.assertIn("'numerical_tolerance':None",s)
        self.assertIn('copy.deepcopy(old.prepare())',s)


if __name__=='__main__':unittest.main()
