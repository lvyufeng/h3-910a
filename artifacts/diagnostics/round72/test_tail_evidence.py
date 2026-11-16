"""Real NPU evidence acceptance and preserved counterexamples for RoPE tail."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import unittest
from summarize72 import aggregate, summarize, validate_fixtures

ROOT=Path(__file__).resolve().parent


class RealEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads((ROOT/'ranks'/f'rope_rank{rank}.json').read_text()) for rank in range(8)]
        cls.previous=[json.loads((ROOT.parent/'round71'/'ranks'/f'vision_rank{rank}.json').read_text()) for rank in range(8)]
    def test_complete_output_and_all_27_rope_layers_without_goal_promotion(self):
        r=summarize()
        self.assertTrue(r['complete_output_identity_accepted'])
        self.assertEqual(r['complete_outputs_compared'],96)
        self.assertEqual(r['actual_rope_pair_checks'],216);self.assertEqual(r['actual_query_key_output_checks'],432)
        self.assertAlmostEqual(r['cases']['tail']['median_complete_encode_ids_s'],6.4630467696115375)
        for key in ('complete_encode_ids_under2s_accepted','whole_service_text_under2s_proven','official_goal_complete','production_deployed','full_packed_gather_fusion_supported'):self.assertFalse(r[key])
    def test_fullpacked_real_fixture_not_dense_replay_and_finite_bounds(self):
        r=validate_fixtures(ROOT)
        self.assertEqual(r['captured_layers'],[0,8,16,26]);self.assertEqual(r['fullpacked_capture_file_bytes'],390178148)
        self.assertLess(r['fullpacked_capture_file_bytes'],512*1024*1024)
        self.assertTrue(r['wrong_pitch_and_alignment_counterexamples_retained'])
        self.assertTrue(r['all_real_standalone_outputs_exact'])
        for layer in (0,8,16,26):
            meta=json.loads((ROOT/'capture'/f'rope_layer{layer:02d}.json').read_text())
            self.assertEqual(meta['query_layout']['stride'],[3456,72,1]);self.assertEqual(meta['key_layout']['storage_offset'],1152)
            self.assertEqual(meta['tensor_identities']['packed_qkv']['bytes'],55738368)
    def test_actual_odd_rows_signed_zero_and_four_fp32_inputs(self):
        r=validate_fixtures(ROOT)
        self.assertEqual(r['runtime_core_budget'],30);self.assertEqual(r['tiny_and_odd_micro_tokens'],[1,5,8,61])
        self.assertTrue(r['multistep_inactive_rows_and_direct_fp32_readonly_verified'])
        micro=json.loads((ROOT/'micro_tail02.json').read_text())['cases']['61']
        self.assertTrue(micro['four_fp32_inputs_readonly']);self.assertTrue(micro['direct_tail_query']['raw_bits_equal'])
        self.assertTrue(micro['multiple_steps_and_inactive_last_rows_exercised'])
    def test_failed_gather_is_retained_never_an_enabled_fallback(self):
        failure=json.loads((ROOT/'gather32_failure.json').read_text());support=json.loads((ROOT/'gather_support.json').read_text())
        self.assertEqual(failure['failure']['type'],'SystemExit');self.assertIn('t.vgather',failure['failure']['message'])
        self.assertEqual(failure['cases'],{});self.assertFalse(support['gather_path_enabled'])
        self.assertFalse(any(support['queries'].values()))
        self.assertNotIn('from vision_rope72 import', (ROOT/'vision_tail_tp8.py').read_text())
        self.assertNotIn('vgather', (ROOT/'vision_rope_tail72.py').read_text())
    def test_source_and_fixture_fingerprints_match_local_expected_and_compiled_objects(self):
        expected=json.loads((ROOT/'expected_sources.json').read_text());actual=json.loads((ROOT/'final_verification.json').read_text())
        self.assertTrue(actual['all_sources_and_fixtures_match'])
        for category in ('production','diagnostics','fixtures'):
            self.assertEqual(set(actual[category]),set(expected[category]))
            for path,digest in expected[category].items():
                self.assertEqual(actual[category][path]['expected'],digest);self.assertEqual(actual[category][path]['sha256'],digest)
                self.assertTrue(actual[category][path]['matches'])
                name=Path(path).name;local=ROOT/name
                if name=='candidate_padded_text_encoder.py':local=ROOT.parent/'round65'/'source'/name
                elif name=='evidence_io72.py':local=ROOT.parent/'round63'/'evidence_io63.py'
                if local.is_file():self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),digest)
        self.assertEqual({row['tokens'] for row in actual['compiled_kernels'].values()},{1,5,8,61,8064})
        self.assertEqual(len(actual['production']),23);self.assertEqual(len(actual['diagnostics']),24);self.assertEqual(len(actual['fixtures']),5)
    def test_corrupt_rope_word_layout_readonly_or_layer_coverage_fails(self):
        for mutation in ('bits','layout','readonly','layer'):
            rows=deepcopy(self.rows);checks=rows[0]['cases']['tail']['rope_checks']
            if mutation=='bits':checks[0]['query']['raw_bits_equal']=False
            elif mutation=='layout':checks[0]['output_layouts'][0]['stride'][0]+=1
            elif mutation=='readonly':checks[0]['packed_and_coefficients_readonly']=False
            else:checks.pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):aggregate(rows,self.previous)
    def test_changed_fresh_vision_output_is_preserved_failed_acceptance(self):
        rows=deepcopy(self.rows)
        rows[0]['cases']['tail']['warm_calls'][0]['identities']['vision_output'][0]['sha256']='0'*64
        r=aggregate(rows,self.previous)
        self.assertFalse(r['complete_output_identity_accepted']);self.assertFalse(r['complete_encode_ids_under2s_accepted'])
    def test_same_call_rank_max_then_median_and_restoration_guards(self):
        r=aggregate(self.rows,self.previous)
        for mode in ('native','tail','restored'):
            maxima=[max(row['cases'][mode]['warm_calls'][i]['seconds'] for row in self.rows) for i in range(3)]
            self.assertEqual(r['cases'][mode]['median_complete_encode_ids_s'],statistics.median(maxima))
        self.assertEqual(r,aggregate(list(reversed(self.rows)),self.previous))
        rows=deepcopy(self.rows);rows[0]['vision_rope_restored']=False
        with self.assertRaises(ValueError):aggregate(rows,self.previous)


if __name__=='__main__':unittest.main()
