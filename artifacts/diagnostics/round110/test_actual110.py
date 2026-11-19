"""All eight exact-input block43 controls and original CPU-Half identities remain intact."""
from pathlib import Path
import json
import unittest
ROOT=Path(__file__).resolve().parent


class ActualBlock43_110(unittest.TestCase):
    def test_all8_exactoriginalraw42_and_originalHalfweights_before_layer43(self):
        ranks=[json.loads((ROOT/('results/block43_01/block43_rank%d.json'%rank)).read_text())for rank in range(8)]
        self.assertTrue(all(r['completed']for r in ranks))
        for r in ranks:
            self.assertEqual(len(r['originalCPUHalf_full_layer43_weight_SHAs_verified']),11)
            self.assertTrue(r['original_parameters_unchanged']and r['exact_CPU_input_unchanged']and r['original_CPU_gold_unchanged'])
            self.assertTrue(r['native_recovery']['raw_bits_equal'])
        for name in('native','precise_attention'):
            hashes={r['cases'][name]['raw43_vs_original_CPU_Half']['npu']['sha256']for r in ranks}
            self.assertEqual(len(hashes),1)
            for r in ranks:
                c=r['cases'][name];self.assertTrue(c['repeated_bits_equal'])
                self.assertEqual(c['exact_CPU_raw42_input']['sha256'],'58af7295b0544fcf8c68994d769a6495402d30cf24c8b884ff3cf3b67847335f')
        self.assertFalse(ranks[0]['full_encoder_precision_accepted']or ranks[0]['whole_goal_complete'])
        self.assertIsNone(ranks[0]['numerical_tolerance'])
    def test_same_gate_Half_silu_float_variant_same_NPU_bits_not_CPU_alignment(self):
        r=json.loads((ROOT/'results/block43_01/block43_rank0.json').read_text());leaf=r['actual_layer43_MLP_leaf_controls']['controls']
        silu=leaf['silu_on_actual_native_gate']
        self.assertEqual(silu['native_vs_cpu_half_on_same_native_gate']['npu']['sha256'],
                         silu['float_activation_diagnostic_vs_cpu_half']['npu']['sha256'])
        self.assertFalse(silu['native_vs_cpu_half_on_same_native_gate']['raw_bits_equal'])
        self.assertTrue(r['actual_layer43_MLP_leaf_controls']['frozen_unchanged'])
    def test_all_old_and_new_sources_binary_CPUcaptures_and_workers_verified(self):
        r=json.loads((ROOT/'final_source_verification110.json').read_text())
        self.assertEqual(len(r['production']),23);self.assertEqual(len(r['installed_api_sources']),13)
        self.assertEqual(len(r['execution_sources']),230);self.assertEqual(len(r['kernel_objects']),38)
        self.assertTrue(r['all8_block43_exact_originalCPU42_rebase_complete']and r['all_relevant_workers_zero'])
        self.assertFalse(r['remaining_workers']);self.assertEqual(len(r['actual_saved_outputs']),5)
        self.assertTrue(r['all_production_API_execution_source_originalHF_CPUgold_frozenQKV_launcher_and_recorded_binaries_unchanged'])
        self.assertTrue(r['original_CPU_raw42_43_capture']['all_original_25_R78_boundaries_bothpasses_SHA_equal'])
        self.assertEqual(set(r['original_CPU_raw42_43_capture']['captures']),{'42','43'})


if __name__=='__main__':unittest.main()
