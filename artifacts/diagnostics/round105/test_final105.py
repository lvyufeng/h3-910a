"""Final original-source/object/worker and genuine local-output audit, not precision acceptance."""
from pathlib import Path
import hashlib
import json
import unittest
ROOT=Path(__file__).resolve().parent


class Final105Evidence(unittest.TestCase):
    def test_read_only_all_original_sources_objects_gold_and_workers_audited(self):
        r=json.loads((ROOT/'final_source_verification_cpu105.json').read_text())
        self.assertEqual(len(r['production']),23);self.assertEqual(len(r['installed_api_sources']),13)
        self.assertEqual(len(r['execution_sources']),151);self.assertEqual(len(r['kernel_objects']),24)
        self.assertTrue(r['all_relevant_workers_zero']);self.assertFalse(r['remaining_workers'])
        self.assertTrue(r['all_production_API_execution_source_originalHF_CPUgold_frozenQKV_launcher_and_recorded_binaries_unchanged'])
        self.assertTrue(r['all8_R102_encoder_results_complete_native_recovery_verified'])
        self.assertEqual(len(r['actual_saved_outputs']),4)
    def test_actual_local_Half_output_captures_byte_identity_no_warm_input_cache(self):
        p=ROOT.parent/'round102/captured_outputs102.json';control=json.loads(p.read_text())
        self.assertEqual(len(control),4)
        for item in control.values():
            actual=ROOT.parents[2]/Path(item['path'])
            self.assertTrue(actual.is_file())
            self.assertEqual(hashlib.sha256(actual.read_bytes()).hexdigest(),item['file_SHA256'])
            self.assertTrue(item['captured_outside_warm_clock_never_used_as_input'])
    def test_exact_new_kernel_output_bits_and_negative_speed_preserved(self):
        r=json.loads((ROOT/'results/leaf01/leaf_compare105.json').read_text());self.assertTrue(r['completed'])
        for image in r['images']:
            self.assertTrue(image['flash_vs_old']['raw_bits_equal'])
            self.assertTrue(image['all_warm_repeats_and_QKV_unchanged'])
            self.assertLess(image['old_over_flash_median_speedup'],1)
        s=json.loads((ROOT/'STATE.json').read_text());self.assertFalse(s['whole_goal_complete']or s['full_encoder_precision_accepted'])
        self.assertTrue(s['owned_lease']['released_after_final_audit'])
        self.assertTrue(s['full_vision_and_full106_not_executed_due_negative_leaf_speedup'])


if __name__=='__main__':unittest.main()
