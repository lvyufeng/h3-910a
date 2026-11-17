"""Validate the signed-zero explanation without treating a hash mismatch as numeric error."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class EmbeddingSignedZeroEvidence(unittest.TestCase):
    def test_original_CPU_embedding_was_reconstructed_and_hash_verified(self):
        r=json.loads((ROOT/'embedding_sign.json').read_text())
        self.assertTrue(r['completed']);self.assertTrue(r['FP16_reference_embedding_unchanged'])
        self.assertEqual(hashlib.sha256((ROOT/'embedding_zero_sign81.py').read_bytes()).hexdigest(),r['source_sha256'])
        reference=json.loads((ROOT.parent/'round78'/'reference.json').read_text())
        self.assertEqual(r['reference_embedding_raw_sha256'],reference['embedding_output']['sha256'])
    def test_one_signed_zero_explains_actual_observed_R80_hash(self):
        r=json.loads((ROOT/'embedding_sign.json').read_text())
        self.assertEqual(r['signed_zero_changed_elements'],1)
        self.assertTrue(r['numeric_equal_before_after']);self.assertFalse(r['raw_bits_equal_before_after'])
        self.assertTrue(r['all_bit_differences_are_signed_zeros'])
        self.assertEqual(r['R80_observed_NPU_embedding_raw_sha256'],r['local_FP32_reduction_model_raw_sha256'])
        observed=json.loads((ROOT.parent/'round80'/'tp8'/'ranks'/'tp8_rank0.json').read_text())
        self.assertEqual(r['R80_observed_NPU_embedding_raw_sha256'],observed['cases']['cpu_fp16_comparison']['embedding']['npu']['sha256'])
    def test_local_collective_model_is_not_called_actual_new_HCCL_or_full_pass(self):
        r=json.loads((ROOT/'embedding_sign.json').read_text())
        self.assertFalse(r['actual_HCCL_reexecuted']);self.assertFalse(r['CPU_reference_changed'])
        self.assertFalse(r['production_changed']);self.assertFalse(r['full_encoder_precision_accepted']);self.assertFalse(r['whole_goal_complete'])


if __name__=='__main__':unittest.main()
