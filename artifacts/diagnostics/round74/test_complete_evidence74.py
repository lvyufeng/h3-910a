"""Final complete original-BF16 HF reference/repeat gates with retained failed quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from summarize_reference74 import aggregate,summarize
from summarize_capture74 import summarize as capture_summary

ROOT=Path(__file__).resolve().parent


class CompleteHFReferenceEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first=json.loads((ROOT/'hf_reference01.json').read_text());cls.repeat=json.loads((ROOT/'language_repeat01.json').read_text())
        cls.modal=json.loads((ROOT/'modal_comparison01.json').read_text());cls.vision=json.loads((ROOT/'vision_repeat01.json').read_text())
        cls.stability=json.loads((ROOT/'reference_stability01.json').read_text())
    def test_original_bf16_fullmultimodal_reference_complete_and_all50repeatstable(self):
        r=summarize();self.assertTrue(r['independent_full_multimodal_reference_completed'])
        self.assertTrue(r['original_bf16_reference_language_repeat_bits_stable']);self.assertTrue(r['original_bf16_reference_vision_vs_retained_independent_reference_bits_stable'])
        self.assertEqual(r['loaded_HF_parameter_tensors'],903);self.assertEqual(r['loaded_HF_parameter_bytes'],51506202080)
        self.assertEqual(r,json.loads((ROOT/'summary.json').read_text()))
    def test_actual_quality_difference_is_not_stableexperimental_identity_or_officialpass(self):
        r=summarize();p=r['selected50_full_and_modal_comparisons']
        self.assertAlmostEqual(p['full']['relative_rms'],0.4095109240532696,places=12)
        self.assertAlmostEqual(p['text_positions']['relative_rms'],0.001680392965248786,places=12)
        self.assertAlmostEqual(p['visual_positions']['relative_rms'],0.4386801550918832,places=12)
        self.assertEqual(p['full']['max_abs_error'],15140)
        self.assertFalse(r['selected50_hidden_states_equal_HF']);self.assertFalse(r['official_quality_accepted'])
        self.assertFalse(r['whole_service_under2s_proven']);self.assertFalse(r['official_goal_complete']);self.assertFalse(r['production_deployed'])
    def test_complete_reference_raw_dtype_and_different_finalnorm_boundaries_are_explicit(self):
        self.assertEqual(self.first['parameter_dtypes'],{'torch.bfloat16':903})
        self.assertEqual(self.first['linear_input_dtypes_observed'],['torch.bfloat16'])
        self.assertTrue(self.stability['pre_final_norm_matches_layer49_identity'])
        self.assertTrue(self.stability['pre_final_norm_repeat']['raw_bits_equal']);self.assertTrue(self.stability['post_final_norm_repeat']['raw_bits_equal'])
        self.assertTrue(self.modal['final_norm_difference_is_excluded_from_h3_comparison'])
        self.assertGreater(self.modal['hf_final_norm_boundary']['relative_rms'],6)
    def test_partial_sharedfeature_F32adapter_inputSHA_and_unstableproducer_cannot_pass(self):
        for mutation in('partial','shared','FP32adapter','dtype','inputsha','unstable'):
            a,b,c,v,s=map(deepcopy,(self.first,self.repeat,self.modal,self.vision,self.stability))
            if mutation=='partial':a['completed']=False
            elif mutation=='shared':a['shared_native_visual_features_used']=True
            elif mutation=='FP32adapter':a['fp32_linear_adapter_used']=True
            elif mutation=='dtype':a['linear_input_dtypes_observed']=['torch.float32']
            elif mutation=='inputsha':b['independent_HF_multimodal_input_sha256']='0'*64
            else:b['layers'][24]['sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):aggregate(a,b,c,v,capture_summary(),s)
    def test_published_stability_modalmetric_and_wrong_integer_positions_are_rejected(self):
        for mutation in('publishedstability','modaltextRR','positions'):
            a,b,c,v,s=map(deepcopy,(self.first,self.repeat,self.modal,self.vision,self.stability))
            if mutation=='publishedstability':s['all50_layer_raw_identities'][0]['repeat_sha256']='f'*64
            elif mutation=='modaltextRR':c['comparisons']['selected50']['text_positions']['relative_rms']=.005
            else:c['comparisons']['text_positions']['raw_bits_equal']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):aggregate(a,b,c,v,capture_summary(),s)
    def test_error_is_reported_on_all_elements_not_removed_as_outlierrepair(self):
        p=self.modal['selected50_outliers']
        self.assertTrue(p['finite']);self.assertGreater(p['visual_error_fraction'],.99999)
        self.assertEqual(len(p['top16_channels']),16);self.assertEqual(len(p['top16_tokens']),16)
        self.assertTrue(all(p['top16_token_is_visual']))
        self.assertAlmostEqual(p['top16_channel_error_fraction'],.9767680963708597,places=12)
        self.assertAlmostEqual(p['top16_token_error_fraction'],.9812110102257036,places=12)
        self.assertGreater(self.modal['comparisons']['selected50']['full']['relative_rms'],.4)
    def test_selected_layer_shape_and_deepstack_boundaries_not_claimed_as_puretextport(self):
        self.assertEqual({p['layer']for p in self.repeat['selected']},{0,1,2,9,24,49})
        for p in self.repeat['selected']:
            self.assertEqual(p['full']['shape'],[1,2043,5120]);self.assertEqual(p['text']['shape'],[27,5120]);self.assertEqual(p['visual']['shape'],[2016,5120])
            self.assertIn('PREmodelDeepStackinjection',p['boundary'])
    def test_final_unchanged_sources_original_refs_and_local_archived_sources_match(self):
        r=json.loads((ROOT/'final_verification.json').read_text());e=json.loads((ROOT/'expected_sources.json').read_text())
        self.assertTrue(r['all_match']);self.assertFalse(r['production_modified']);self.assertEqual(len(r['production']),23)
        for category in('production','diagnostics','installed_sources','fixtures'):
            self.assertEqual(set(r[category]),set(e[category]))
            for path,digest in e[category].items():
                self.assertTrue(r[category][path]['matches']);self.assertEqual(r[category][path]['sha256'],digest)
                local=ROOT/Path(path).name
                if Path(path).name=='candidate_padded_text_encoder.py':local=ROOT.parent/'round65'/'source'/'candidate_padded_text_encoder.py'
                if local.is_file():self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
