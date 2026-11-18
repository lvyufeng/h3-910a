"""Actual true4032-key softmax padding controls and exact-shape vision composition."""
from pathlib import Path
import ast
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class DirectNZContracts(unittest.TestCase):
    def test_real910capabilities_no_A2_Floatvcmax(self):
        r=json.loads((ROOT/'actual_sdk_reduction97.json').read_text())
        self.assertEqual(r['compile_soc'],'Ascend910');self.assertEqual(r['core_num'],32)
        self.assertFalse(r['intrinsic_checks']['Intrinsic_vcmax:float32'])
        self.assertTrue(r['intrinsic_checks']['Intrinsic_vmax:float32']and r['intrinsic_checks']['Intrinsic_vcadd:float32'])
        s=(ROOT/'softmax/softmax_pack_padded_fixed97.py').read_text()
        self.assertNotIn('t.vec_reduce_max(',s)
        self.assertIn('launched_usable_cores',s)
    def test_failed_roots_preserved_and_actual_complete_smoke(self):
        one=json.loads((ROOT/'results/nz01/softmax_smoke97.json').read_text())
        two=json.loads((ROOT/'results/nz02/softmax_smoke97.json').read_text())
        three=json.loads((ROOT/'results/nz03/softmax_smoke97.json').read_text())
        self.assertFalse(one['completed']or two['completed']);self.assertTrue(three['completed'])
        for m in('512','448'):
            r=three['cases'][m]
            self.assertTrue(r['key_tail64_zero_and_padded_M_rows_zero']and r['repeat_bits_equal']and r['input_SHA_unchanged'])
            self.assertTrue(r['recovered_float_probabilities_vs_CPU']['finite'])
            self.assertEqual(r['meta']['block_dim'],30);self.assertEqual(r['meta']['GM_argument_count'],5)
            self.assertEqual(len(r['runtime_s']),5)
            self.assertEqual(r['meta']['source_sha256'],hashlib.sha256((ROOT/'softmax/softmax_pack_padded_fixed97.py').read_bytes()).hexdigest())
        self.assertFalse(three['full_encoder_precision_accepted']or three['whole_goal_complete'])
    def test_no_Torch_score_unpack_probability_pack_and_batchshape_exact(self):
        s=(ROOT/'accurate_vision_attention_nz97.py').read_text();ast.parse(s)
        start=s.index('    def forward(');end=s.index('    def snapshot(');forward=s[start:end]
        self.assertNotIn('softmax(',forward);self.assertNotIn('unpack_c(',forward);self.assertNotIn('pack_a(p',forward)
        self.assertIn('return result.unsqueeze(0)',forward)
        self.assertIn("[scores,auxf,auxf,hi,lo]",forward)
        self.assertIn("[hi,lo,packed_v,auxf,out]",forward)
        self.assertIn('candidate_out.shape==old_repeat.shape==new_repeat.shape==(1,16,4032,72)',(ROOT/'leaf_compare97.py').read_text())
    def test_no_posthoc_tolerance_or_hiddenconditioning_fullvision_gate(self):
        s=(ROOT/'vision_probe97.py').read_text()
        self.assertIn('out.shape==repeat.shape==(1,16,4032,72)',s)
        self.assertIn("expected['R90_current_joint_vision_merged_SHA256']",s)
        self.assertIn("'not_full_encoder_timing':True",s)
        self.assertIn("'full_encoder_precision_accepted':False",s)
        self.assertIn("'numerical_tolerance':None",s)


if __name__=='__main__':unittest.main()
