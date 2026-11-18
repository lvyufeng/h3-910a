"""Root single-launch engine contracts, no Torch or CANN execution."""
from pathlib import Path
import ast
import unittest
import flash_engine99 as engine
ROOT=Path(__file__).resolve().parent


class Engine99Contracts(unittest.TestCase):
    def test_no_query_row_pad_and_correct_original_value_packing(self):
        s=(ROOT/'flash_engine99.py').read_text();ast.parse(s)
        self.assertIn('padded.reshape(16,4032,5,16).permute(0,2,1,3)',s)
        self.assertIn('padded_v.reshape(16,252,16,80).permute(0,1,3,2)',s)
        self.assertNotIn('pack_a(',s)
        self.assertIn('q.shape==k.shape==v.shape==(1,16,4032,72)',s)
        self.assertIn('contiguous().unsqueeze(0)',s)
    def test_softmax_shift_bound_16k_not_unsafe65k_no_clamp(self):
        s=(ROOT/'flash_engine99.py').read_text()
        self.assertEqual(engine.MAX_SCORE_BOUND,16384.0)
        self.assertIn('math.isfinite(bound)',s);self.assertIn('torch.isfinite(v).all()',s)
        self.assertIn('qnorm*knorm*(72**-0.5)*(1+1e-4)',s)
        self.assertNotIn('clamp(',s)
    def test_disabled_delegation_enable_guard_and_uninstall_metadata_witness(self):
        class Candidate:
            def __init__(self):self._vision_sdpa_bf16_compatible=lambda q,k,v:('original',q,k,v)
        class Runtime:device=0
        c=Candidate();e=engine.FlashVisionAttention99(c,Runtime(),ROOT/'unused_source_test')
        self.assertEqual(e.forward(1,2,3),('original',1,2,3))
        with self.assertRaises(RuntimeError):e.set_enabled(True)
        e._meta={'nested':{'SHA':'original'}}
        witness=e.snapshot();e._meta['nested']['SHA']='mutated'
        self.assertEqual(witness['kernel']['nested']['SHA'],'original')
        result=e.uninstall();self.assertEqual(result['kernel']['nested']['SHA'],'mutated')
        self.assertIsNone(e._meta);self.assertFalse(result['full_encoder_precision_accepted']or result['whole_goal_complete'])
        self.assertIsNone(result['numerical_tolerance'])
    def test_only_private_helper_patched_not_Torch_or_model_weights(self):
        s=(ROOT/'flash_engine99.py').read_text()
        self.assertIn("HELPER='_vision_sdpa_bf16_compatible'",s)
        self.assertIn('copy.deepcopy(self._meta)',s)
        self.assertNotIn('nn.functional.',s);self.assertNotIn('.weight.float()',s)


if __name__=='__main__':unittest.main()
