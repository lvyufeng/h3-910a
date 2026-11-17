"""Protect actual full-step/video/precision scope and retained failed diagnostics."""
import hashlib
import json
from pathlib import Path
import unittest
from timing_rows76 import complete_timing_cases
from summarize_generation76 import summarize

ROOT=Path(__file__).resolve().parent


class ActualGenerationEvidence(unittest.TestCase):
    def test_cold_request_has_complete_rank_rows_but_fails_all49_15s_gate(self):
        text=(ROOT/'progress'/'generation02_snapshot04.log').read_text();case=complete_timing_cases(text,1)[0]
        self.assertEqual(len(case['steps']),49)
        self.assertTrue(all(len(step['rank_observations'])==8 for step in case['steps']))
        self.assertFalse(case['all49_strictly_below15s']);self.assertEqual(case['all49_count_strictly_below15s'],47)
        self.assertGreater(case['first_step_s'],60);self.assertGreater(case['second_step_s'],20)
        self.assertEqual(case['after_first_two_count_strictly_below15s'],47)
        self.assertGreater(case['stage_rank_rows']['decode']['max_s'],120)
    def test_missing_rank_prefix_is_not_silently_reassigned(self):
        text=(ROOT/'progress'/'generation02_snapshot04.log').read_text()
        text=text.replace('(DiffusionWorker_SP1 pid=3448881) [H3PERF] dstep_complete 1','[H3PERF] dstep_complete 1',1)
        with self.assertRaisesRegex(ValueError,'Ambiguous missing-rank prefix'):complete_timing_cases(text,1)
    def test_sealed_final_log_exactly_recovered_not_partial_log_accepted(self):
        meta=json.loads((ROOT/'final_log_metadata.json').read_text());content=(ROOT/'generation_complete.log').read_bytes()
        self.assertEqual(len(content),meta['bytes']);self.assertEqual(hashlib.sha256(content).hexdigest(),meta['sha256'])
        self.assertNotEqual(hashlib.sha256((ROOT/'generation.log').read_bytes()).hexdigest(),meta['sha256'])
    def test_both_cases_thermal_and_encoder_scope_do_not_imply_lossless(self):
        result=summarize();self.assertFalse(result['native_BF16_compute_used']);self.assertFalse(result['live_encoder_in_request']);self.assertFalse(result['official_goal_complete'])
        cold,warm=result['same_call_rank_max_timing_cases']
        self.assertEqual(cold['case'],'optimized');self.assertEqual(warm['case'],'HF_reference_conditioning')
        self.assertFalse(cold['decode_reporter_within20s']);self.assertTrue(warm['decode_reporter_within20s'])
        self.assertTrue(warm['all49_strictly_below15s']);self.assertFalse(cold['all49_strictly_below15s'])
        self.assertTrue(result['stage_rank_maxima_must_not_be_added_to_infer_request_wall'])
        self.assertTrue(result['different_raw_RGB_outputs'])
        self.assertIn('excludes uint8 frame quantization,videoD2H',cold['decode_stage_scope'])
    def test_actual_video_deliverables_have_exact_receipt_hashes_and_124frames(self):
        receipt=json.loads((ROOT/'generation_result.json').read_text());scan=json.loads((ROOT/'video_scan.json').read_text())
        self.assertTrue(scan['completed']);self.assertFalse(scan['official_quality_accepted']);self.assertFalse(scan['optical_flow_or_CLIP_or_OCR_score_computed'])
        for case,row in zip(receipt['cases'],scan['videos']):
            path=ROOT/(case['case']+'.mp4');self.assertEqual(path.stat().st_size,case['video']['bytes'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),case['video']['sha256']);self.assertEqual(row['mp4_sha256'],case['video']['sha256'])
            self.assertEqual(row['label'],case['case']);self.assertTrue(row['all124_frames_decoded']);self.assertEqual([f['index']for f in row['frames']],list(range(124)))
            self.assertEqual((row['stream']['width'],row['stream']['height']),(1344,768));self.assertEqual(row['stream']['audio_tracks'],0)
            self.assertAlmostEqual(row['stream']['duration_s'],124/24,places=5)
            self.assertIn('original_exported_RGB_endpoint_metrics_before_MP4_codec',row)
    def test_VAE_source_restored_and_failed_local_decoder_not_repaired_globally(self):
        restored=json.loads((ROOT/'restoration.json').read_text());self.assertTrue(restored['restored'])
        self.assertEqual(restored['observed_sha256'],restored['baseline_sha256']);self.assertFalse(restored['encoder_and_pipeline_sources_modified'])
        failed=json.loads((ROOT/'optimized_scan.json').read_text());self.assertIn('libtiff.5.dylib',failed['failure']['message'])
        self.assertFalse(failed.get('completed',False));self.assertEqual(failed['videos'],[])
        self.assertEqual(failed['source_sha256'],hashlib.sha256((ROOT/'scan_video_local76.py').read_bytes()).hexdigest())
    def test_quality_is_rejected_and_only_encoder_reference_not_independent_DiT(self):
        from summarize_quality76 import summarize as quality_summary
        r=quality_summary();self.assertFalse(r['official_lossless_accepted']);self.assertFalse(r['official_goal_complete']);self.assertFalse(r['sole_cause_established'])
        self.assertTrue(r['HF_encoder_conditioning_alone_did_not_remove_observed_visual_artifacts'])
        self.assertTrue(all(not case['visual_review']['quality_pass']for case in r['videos']))
        self.assertFalse(r['full_live_encoder_E2E_measured']);self.assertFalse(r['audio_waveform_exported_or_reviewed'])
        self.assertEqual([case['manual_review_time_points']for case in r['videos']],[[0,12,24,36,48,62,74,86,98,110,123]]*2)
    def test_recovered_PNGs_validated_individually_without_accepting_partial_archive(self):
        proof=json.loads((ROOT/'transfer_recovery.json').read_text());self.assertFalse(proof['archive_complete'])
        self.assertTrue(proof['recovered_individual_PNG_hashes_match_original_remote']);self.assertFalse(proof['remote_individual_hash_validation_pending'])
        for name,info in proof['only_complete_individual_PNG_members_recovered'].items():
            p=ROOT/name;self.assertEqual(p.stat().st_size,info['bytes']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),info['sha256'])


if __name__=='__main__':unittest.main()
