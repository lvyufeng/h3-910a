"""All actual encoder ranks include true VisionTP8, original CPU Half gold, real stock clocks."""
from pathlib import Path
import json
import unittest
import collect108
ROOT=Path(__file__).resolve().parent


class Actual108Evidence(unittest.TestCase):
    def test_all8_complete_originalhalf_full50_text_plus27vision_recovery(self):
        r=collect108.collect(ROOT/'results/full01')
        self.assertTrue(r['completed']and r['all8_native_recovery_parameters_fixture_originalCPUgold_unchanged'])
        for c in r['cases'].values():
            self.assertTrue(c['all8_selected50_and_all50_text_boundary_SHAs_equal'])
            self.assertTrue(c['all8_repeat_stock_equivalent_and_R107_visual_SHA_verified'])
            self.assertEqual(c['selected50_relativeRMS_all8_min_max'][0],c['selected50_relativeRMS_all8_min_max'][1])
            self.assertEqual(len(c['paired_8rankmax_clock_s']),5)
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertIsNone(r['numerical_tolerance'])
    def test_summary_reproduces_exact_paired_rankmax_samples(self):
        self.assertEqual(json.loads((ROOT/'summary108.json').read_text()),collect108.collect(ROOT/'results/full01'))
    def test_real_TP8_has_H2_attention_and_actual_float_collectives_in_allranks(self):
        for rank in range(8):
            r=json.loads((ROOT/('results/full01/tp8_rank%d.json'%rank)).read_text());self.assertTrue(r['completed'])
            for name in('corrected_VisionTP8_native_text','corrected_VisionTP8_precise_text'):
                c=r['cases'][name];self.assertTrue(c['true_VisionTP8'])
                self.assertEqual(c['head2_snapshot']['local_heads'],2)
                snap=c['VisionTP8_snapshot'];self.assertEqual(snap['geometry']['local_mlp_channels'],538)
                self.assertEqual(snap['stats']['primed_weight_modules'],108)
                self.assertGreater(snap['stats']['all_reduce_calls'],0)
                self.assertEqual(snap['stats']['column_runs'],snap['stats']['row_runs'])
                self.assertEqual(snap['backend_snapshot']['Float_weight_shadows'],0)
    def test_serial18_kernel_objects_not_recompiled_concurrently(self):
        r=json.loads((ROOT/'results/full01/prebuild108.json').read_text())
        self.assertTrue(r['all18objects_SHA_verified_before_8rank_model_priming'])
        metas=r['text_kernels']+r['vision_kernels']+[r['erf_kernel'],r['flash_kernel'],r['local_head2_kernel']]+r['TP_GEMM_kernels']
        self.assertEqual(len(metas),18)
        for meta in metas:self.assertEqual(meta['block_dim'],30);self.assertTrue(meta['obj_sha256'])


if __name__=='__main__':unittest.main()
