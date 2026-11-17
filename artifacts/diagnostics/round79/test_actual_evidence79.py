"""Test real FP16 stage receipts rather than treating source tests as accuracy proof."""
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parent


class ActualFP16Evidence(unittest.TestCase):
    def test_actual_vision_run_completed_with_uniform_FP16_params(self):
        r=json.loads((ROOT/'vision.json').read_text());self.assertTrue(r['completed']);self.assertTrue(r['vision_FP16_finite_and_repeat_stable'])
        self.assertEqual(r['parameter_tensors'],351);self.assertEqual(r['parameter_bytes'],1190533600)
        self.assertEqual(r['parameter_dtype'],'FP16');self.assertEqual(r['actual_input_pixels']['dtype'],'torch.float16')
        self.assertEqual(r['actual_input_grid'],[[1,48,84],[1,48,84]])
        self.assertEqual(len(r['layers']),27)
        self.assertTrue(all(row['output']['finite']and row['output']['dtype']=='torch.float16'for row in r['layers']))
    def test_all_full_NPU_stage_repeats_bit_equal(self):
        r=json.loads((ROOT/'vision.json').read_text());self.assertEqual(len(r['cases']),3)
        for row in r['cases']:
            self.assertTrue(row['merged_repeat']['raw_bits_equal']);self.assertEqual(row['merged_repeat']['elements'],10321920)
            self.assertEqual(len(row['deepstack_repeat']),3)
            self.assertTrue(all(x['raw_bits_equal']and x['finite']for x in row['deepstack_repeat']))
    def test_executed_source_and_candidate_hashes_match(self):
        r=json.loads((ROOT/'vision.json').read_text());m=json.loads((ROOT/'candidate'/'build_manifest79.json').read_text())
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(r['source_sha256'],sha(ROOT/'vision_probe79.py'))
        self.assertEqual(r['candidate_source_sha256'],m['candidate_sha256']);self.assertEqual(m['candidate_sha256'],sha(ROOT/'candidate'/'candidate_encoder79.py'))
        self.assertEqual(r['candidate_manifest_sha256'],sha(ROOT/'candidate'/'build_manifest79.json'))
    def test_CPU_source_shards_are_official_not_anonymous_repack(self):
        r=json.loads((ROOT/'CPU_source_verification79.json').read_text());m=json.loads((ROOT.parent/'round78'/'official_text_manifest_receipt78.json').read_text())
        self.assertEqual(r['official_revision'],'0919fb684102ef115126612f3fbb84d1027e7317')
        self.assertEqual(len(r['used_checkpoint_shards_verified_against_official_LFS']),12)
        for name,item in r['used_checkpoint_shards_verified_against_official_LFS'].items():self.assertEqual(item,m['lfs_shards'][name])
    def test_measured_error_and_median_not_false_full_encoder_pass(self):
        r=json.loads((ROOT/'vision.json').read_text());s=json.loads((ROOT/'summary79.json').read_text())
        self.assertEqual(s['resident_single_rank_vision_median_s'],statistics.median(x['resident_vision_s']for x in r['cases']))
        self.assertEqual(s['CPU_FP16_final_vision_relative_RMS'],r['CPU_FP16_comparison']['merged']['relative_rms'])
        self.assertFalse(r['CPU_FP16_comparison']['merged']['raw_bits_equal'])
        self.assertGreater(s['CPU_FP16_final_vision_relative_RMS'],.01)
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['full_encoder_2s_target_claimed']);self.assertFalse(s['whole_goal_complete'])
    def test_partial_archives_are_excluded_from_versioned_proofs(self):
        self.assertIn('/reference/*.pt',(ROOT/'.gitignore').read_text())
        self.assertFalse((ROOT/'reference'/'vision_reference_receipt79.json').exists())


if __name__=='__main__':unittest.main()
