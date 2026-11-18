"""Exact shape gates and bit-preserving corrected fused-PV vision optimization."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class FusedActualEvidence(unittest.TestCase):
    def test_shape_regression_failed_fullvision_preserved(self):
        r=json.loads((ROOT/'results/vision02/vision96.json').read_text())
        self.assertTrue(r['completed']);self.assertGreater(r['cases']['current_patch_bias_erf_accurate_attention']['merged_vs_CPU']['relative_rms'],1)
        old=(ROOT/'pv/accurate_vision_attention_pv96.py').read_text()
        fixed=(ROOT/'pv/accurate_vision_attention_pv_fixed96.py').read_text()
        self.assertIn('        return result\n',old)
        self.assertIn('        return result.unsqueeze(0)\n',fixed)
        self.assertIn('candidate_out.shape==old_repeat.shape==new_repeat.shape==(1,16,4032,72)',(ROOT/'leaf_compare_fixed96.py').read_text())
    def test_real_frozen_leaf_exactshape_and_bits_same_after_fusion(self):
        r=json.loads((ROOT/'results/fixed01/leaf_compare96.json').read_text());self.assertTrue(r['completed'])
        for image in r['images']:
            self.assertTrue(image['fused_vs_old']['raw_bits_equal'])
            self.assertEqual(image['fused_vs_old']['npu']['shape'],[1,16,4032,72])
            self.assertEqual(image['fused_vs_old']['cpu']['shape'],[1,16,4032,72])
            self.assertTrue(image['all_warm_repeats_and_QKV_unchanged'])
            self.assertEqual(image['old_vs_CPU_Half_MATH']['relative_rms'],image['fused_vs_CPU_Half_MATH']['relative_rms'])
    def test_corrected_fused_fullvision_and_all27_DStack_bits_same_old(self):
        old=json.loads((ROOT/'results/vision01/vision96.json').read_text())
        new=json.loads((ROOT/'results/fixed01/vision96.json').read_text());self.assertTrue(new['completed'])
        key='current_patch_bias_erf_accurate_attention';a=old['cases'][key];b=new['cases'][key]
        self.assertEqual(a['merged_vs_CPU']['npu']['sha256'],b['merged_vs_CPU']['npu']['sha256'])
        for x,y in zip(a['DeepStack_vs_CPU'],b['DeepStack_vs_CPU']):self.assertEqual(x['npu']['sha256'],y['npu']['sha256'])
        self.assertEqual([x['output']['sha256']for x in a['raw27_layers']],[x['output']['sha256']for x in b['raw27_layers']])
        self.assertTrue(new['original_parameters_unchanged']and new['pixels_unchanged']and new['CPU_reference_SHA_unchanged'])
        self.assertTrue(new['native_recovery']['merged']['raw_bits_equal'])
        self.assertFalse(new['full_encoder_precision_accepted']or new['whole_goal_complete'])
    def test_compiled_twoL0C_smoke_matches_original_fullK_arithmetic(self):
        r=json.loads((ROOT/'results/fused01/pv_smoke96.json').read_text());self.assertTrue(r['completed'])
        self.assertTrue(r['fused_vs_old_two_products']['raw_bits_equal']and r['fused_repeat']['raw_bits_equal'])
        self.assertTrue(r['all_inputs_unchanged']);self.assertEqual(r['newmeta']['GM_argument_count'],5)
        self.assertTrue(r['newmeta']['single_final_half_cast'])


if __name__=='__main__':unittest.main()
