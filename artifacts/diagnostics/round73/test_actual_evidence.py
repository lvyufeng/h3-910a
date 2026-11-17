"""Actual-data gates for attention profiling, failed controls and finite strided cast."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from summarize_profile73 import summarize as profile_summary,vector
from summarize_geometry73 import summarize as geometry_summary
from summarize_cast73 import aggregate,summarize
from summarize_controls73 import summarize as control_summary

ROOT=Path(__file__).resolve().parent


class ActualEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths=[ROOT/'ranks'/f'cast_rank{i}.json'for i in range(8)]
        if not all(path.is_file()for path in paths):raise unittest.SkipTest('Corrected actual eight-rank cast A/B receipts not yet collected')
        cls.rows=[json.loads(path.read_text())for path in paths]
        cls.previous=[json.loads((ROOT.parent/'round72'/'ranks'/f'rope_rank{i}.json').read_text())for i in range(8)]
    def test_complete_output_identity_and_all_actual_cast_attention_calls(self):
        r=summarize();self.assertTrue(r['complete_output_identity_accepted'])
        self.assertEqual(r['actual_cast_checks'],1296);self.assertEqual(r['actual_attention_output_checks'],432)
        self.assertEqual(r['complete_outputs_compared'],96)
        self.assertFalse(r['whole_service_text_under2s_proven']);self.assertFalse(r['official_goal_complete']);self.assertFalse(r['production_deployed'])
    def test_failed_global_binding_harness_cannot_enter_success_gate(self):
        failed=[json.loads((ROOT/'failed_ab'/f'cast_rank{i}.json').read_text())for i in range(8)]
        self.assertTrue(all(row['failure']['type']=='AttributeError'for row in failed))
        with self.assertRaises(ValueError):aggregate(failed,self.previous)
    def test_all_qkv_cast_bits_layout_finite_and_actual_output_proof_enforced(self):
        for mutation in('bits','layout','readonly','missing_operand','attention'):
            rows=deepcopy(self.rows);check=rows[0]['cases']['cast']['actual_attention_checks'][0]
            if mutation=='bits':check['input_cast_proofs'][0]['proof']['raw_bits_equal']=False
            elif mutation=='layout':check['input_cast_proofs'][0]['candidate_layout']['stride'][1]+=1
            elif mutation=='readonly':check['input_values_readonly']=False
            elif mutation=='missing_operand':check['input_cast_proofs'].pop()
            else:check['attention_output']['max_abs_error']=.125
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):aggregate(rows,self.previous)
    def test_changed_fresh_vision_stays_failed_whole_identity(self):
        rows=deepcopy(self.rows);rows[0]['cases']['cast']['warm_calls'][0]['identities']['vision_output'][0]['sha256']='0'*64
        r=aggregate(rows,self.previous);self.assertFalse(r['complete_output_identity_accepted']);self.assertFalse(r['complete_encode_ids_under2s_accepted'])
    def test_profile_child_parent_levels_and_samecall_slowest_rank(self):
        r=profile_summary();self.assertEqual(r['profile_events_verified'],15576);self.assertEqual(r['complete_outputs_compared'],80)
        self.assertTrue(r['complete_output_identity_accepted'])
        row=r['representative_median_call']['representative']
        self.assertAlmostEqual(row['child_sum_s'],sum(v for k,v in row['current_stream_category_s'].items()if k.startswith('child.')))
        for v in r['profile_vectors']:
            self.assertEqual(v['representative'],max(v['all_ranks'],key=lambda x:x['complete_wall_s']))
        raw=json.loads((ROOT/'profile'/'attention_rank0.json').read_text())['cases']['profile'][0]['event_records']
        raw.pop()
        with self.assertRaises(ValueError):vector(raw)
    def test_geometry_full_matrices_and_negative_speed_controls_are_not_integrated(self):
        r=geometry_summary();self.assertEqual(r['full_stage_matrix_proofs'],180)
        self.assertFalse(r['production_deployed'])
        for mode,rows in r['modes'].items():
            self.assertEqual(len(rows),4)
            self.assertTrue(all(row['all_full_stages_bits_equal']for row in rows))
            if mode.startswith('chunk'):self.assertTrue(all(row['median_s']>row['same_call_native_s']for row in rows))
    def test_finite_micro_native_layout_oracle_and_retained_exceptional_counterexample(self):
        micro=json.loads((ROOT/'cast_finite_micro01.json').read_text())
        self.assertTrue(micro['completed']);self.assertEqual(len(micro['cases']),8)
        for row in micro['cases']:
            self.assertTrue(row['full_bits_equal']);self.assertTrue(row['integer_oracle_equal']);self.assertTrue(row['all_input_values_finite'])
            self.assertTrue(row['source_readonly']);self.assertEqual(row['candidate_layout'],row['native_layout'])
        dense=next(row for row in micro['cases']if row['tokens']==61 and row['input_pitch']==1152)
        self.assertEqual(dense['distinct_finite_bf16_words'],65280)
        failed=json.loads((ROOT/'cast_micro02_layout_failure.json').read_text())
        self.assertFalse(failed['cases'][-1]['full_bits_equal'])
        single=json.loads((ROOT/'cast_singleton_diagnosis.json').read_text())
        self.assertTrue(single['full_bits_equal']);self.assertTrue(single['acceptance_rejected'])
    def test_independent_original_wrapped_profile_restored_controls_and_all_call_proofs(self):
        r=control_summary()
        self.assertTrue(r['complete_output_identity_accepted']);self.assertTrue(r['all_intermediate_identity'])
        self.assertEqual(r['actual_transcription_checks'],432);self.assertEqual(r['complete_outputs_compared'],112)
        self.assertEqual(r['profile_events_verified'],15576)
        self.assertEqual(set(r['phases']),{'original','wrapped','profile','restored'})
        self.assertFalse(r['official_goal_complete']);self.assertFalse(r['whole_service_text_under2s_proven'])
    def test_final_source_reference_object_receipt_matches_current_local_sources(self):
        expected=json.loads((ROOT/'expected_sources.json').read_text());observed=json.loads((ROOT/'final_verification.json').read_text())
        self.assertTrue(observed['all_match']);self.assertFalse(observed['production_modified'])
        for category in('production','diagnostics','fixtures'):
            self.assertEqual(set(expected[category]),set(observed[category]))
            for path,digest in expected[category].items():
                self.assertEqual(observed[category][path]['expected'],digest);self.assertEqual(observed[category][path]['sha256'],digest)
                self.assertTrue(observed[category][path]['matches'])
                local=ROOT/Path(path).name
                if Path(path).name=='candidate_padded_text_encoder.py':local=ROOT.parent/'round65'/'source'/'candidate_padded_text_encoder.py'
                elif Path(path).name=='evidence_io73.py':local=ROOT.parent/'round63'/'evidence_io63.py'
                if local.is_file():self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),digest)
        self.assertEqual(len(observed['production']),23);self.assertEqual(len(observed['diagnostics']),36);self.assertEqual(len(observed['fixtures']),5)
        self.assertEqual({(row['tokens'],row['pitch'])for row in observed['compiled_casts'].values()},{(n,p)for n in(1,5,61,4032)for p in(1152,3456)})


if __name__=='__main__':unittest.main()
