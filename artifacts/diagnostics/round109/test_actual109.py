"""Actual original CPU FP16 raw42/43 capture extends the frozen oracle, never redefines it."""
from pathlib import Path
import json
import unittest
ROOT=Path(__file__).resolve().parent


class ActualCPU109(unittest.TestCase):
    def test_both25layer_baseline_passes_exact_originalR78_sha_and_pair43_replay(self):
        r=json.loads((ROOT/'results/cpu02/CPU_raw_reference109.json').read_text())
        self.assertTrue(r['completed']);self.assertEqual(r['baseline_passes']['passes'],2)
        self.assertTrue(r['baseline_passes']['all_25_R78_boundaries_matched']and r['baseline_passes']['repeat_bit_identical_all_25'])
        self.assertEqual(len(r['baseline_passes']['boundaries']),25)
        self.assertTrue(r['captured_pair_replay']['comparison']['raw_bits_equal'])
        self.assertTrue(r['original_parameters_unchanged']and r['r78_files_after']['unchanged'])
    def test_plain_Half_capture_file_tensor_identities_are_distinct_and_original(self):
        r=json.loads((ROOT/'results/cpu02/CPU_raw_reference109.json').read_text())
        expected={'42':'58af7295b0544fcf8c68994d769a6495402d30cf24c8b884ff3cf3b67847335f',
                  '43':'7600cf45e5230d4aa707d85f719a2a982bede75632a8fd86fb39f6826eecc23e'}
        for key,item in r['captured_tensors'].items():
            self.assertTrue(item['plain_torch_Tensor']and item['reload_tensor_sha256_matches']and item['matches_R78_boundary_SHA256'])
            self.assertEqual(item['tensor']['dtype'],'torch.float16');self.assertEqual(item['tensor']['shape'],[1,2043,5120])
            self.assertEqual(item['tensor']['sha256'],expected[key]);self.assertNotEqual(item['file_sha256'],item['tensor']['sha256'])
        self.assertFalse(r['whole_goal_complete']or r['full_encoder_precision_accepted']);self.assertIsNone(r['numerical_tolerance'])
    def test_first_optionalHF_NPUimport_environment_failure_preserved_not_partial_gold(self):
        old=json.loads((ROOT/'results/cpu01/CPU_raw_failure109.json').read_text())
        self.assertFalse(old['completed']);self.assertEqual(old['failure']['stage'],'build_suffix')
        self.assertIn('libhccl.so',old['failure']['message']);self.assertNotIn('baseline_passes',old)
        self.assertNotIn('captured_tensors',old)


if __name__=='__main__':unittest.main()
