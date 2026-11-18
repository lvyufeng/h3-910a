"""Validate actual eight-rank FP32 row experiments including the negative full-graph result."""
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parent


def receipts():
    return [json.loads((ROOT/'results/row_fp3202/ranks'/('tp8_rank%d.json'%rank)).read_text())for rank in range(8)]


class ActualRowEvidence(unittest.TestCase):
    def test_all8_completed_and_original_native_output_recovered_bitwise(self):
        for rank,r in enumerate(receipts()):
            self.assertTrue(r['completed']);self.assertEqual(r['rank'],rank)
            self.assertTrue(r['native_matches_round81_sha256']);self.assertTrue(r['native_after_row_AB']['raw_bits_equal'])
            self.assertTrue(r['frozen_inputs_unchanged']);self.assertEqual(r['self_check_after_move']['parameter_dtype_counts'],{'torch.float16':752})
            self.assertTrue(r['vision_parameter_provenance']['fully_verified'])
            self.assertTrue(r['cleanup']['row_adapter_uninstalled'])
            self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'full_row_fp32_fixed84.py').read_bytes()).hexdigest())
    def test_real_partial_dtype_was_FP32_on_both_sides_of_each_collective(self):
        for r in receipts():
            a=r['row_adapter_final'];stats=a['stats']
            self.assertEqual(stats['kernel_launches'],804)
            self.assertEqual(stats['fp32_collective_count'],804);self.assertEqual(stats['half_cast_count'],804)
            self.assertEqual(stats['fp32_dtype_before_collective'],{'torch.float32':804})
            self.assertEqual(stats['fp32_dtype_after_collective'],{'torch.float32':804})
            self.assertEqual(stats['calls_rejected'],0)
            self.assertEqual(a['weight_cache']['entries'],100)
            self.assertEqual(a['weight_cache']['bytes'],2162688000)
            self.assertEqual(a['weight_cache']['mutations_detected'],0)
            self.assertEqual(a['prime']['fp32_weight_shadows'],0)
            self.assertTrue(all(not k['compiled_now']for k in a['prime']['kernels'].values()))
            for control in r['actual_activation_packing_controls'].values():
                self.assertTrue(control['a_pack_matches_verified_helper'])
                self.assertTrue(control['a_pack_roundtrip_exact']);self.assertTrue(control['a_pack_tail_rows_zero'])
    def test_CPU_condition_improvement_is_real_but_native_condition_regression_not_hidden(self):
        for r in receipts():
            cases=r['row_full_cases'];base=cases['native_rows_CPU_condition']
            self.assertTrue(base['reproduces_R82_CPU_condition_sha256'])
            self.assertAlmostEqual(base['selected50_vs_CPU']['relative_rms'],.024756999879344863)
            self.assertAlmostEqual(cases['FP32_rows_CPU_condition']['selected50_vs_CPU']['relative_rms'],.02190204269269481)
            native_gap=r['cases']['cpu_fp16_comparison']['selected50_pre_final_norm']['relative_rms']
            self.assertGreater(cases['FP32_rows_native_condition']['selected50_vs_CPU']['relative_rms'],native_gap)
            self.assertFalse(r['full_encoder_precision_accepted']);self.assertFalse(r['whole_goal_complete'])
            for case in cases.values():
                self.assertTrue(case['repeat_vs_first']['raw_bits_equal']);self.assertEqual(len(case['raw_layers']),50)
            layer=r['single_actual_CPU_input_layer0']
            self.assertLess(layer['FP32_partial_vs_CPU']['relative_rms'],layer['native_Half_partial_vs_CPU']['relative_rms'])
            self.assertTrue(layer['FP32_repeat_bits']['raw_bits_equal'])
    def test_true_complete_encoder_time_not_relabelled_as_the_old_native_clock(self):
        rows=receipts();values=[max(r['timing']['samples_synchronized_wall_s_including_d2h'][i]for r in rows)for i in range(3)]
        self.assertAlmostEqual(statistics.median(values),2.433647890109569)
        self.assertGreater(statistics.median(values),2)
        for r in rows:
            self.assertTrue(r['stock_encode_ids_timed']);self.assertTrue(r['timing']['warm_outputs_bit_equal'])
            self.assertIn('FP32 row GEMM+HCCL',r['timing']['path'])
            self.assertTrue(r['cases']['stock_encode_ids_probe']['matches_equivalent_path'])
    def test_executed_private_sources_and_cleanup_are_verified(self):
        source=json.loads((ROOT/'executed_sources84.json').read_text())['runs']['/tmp/h3_round84_row_fp3202']
        for name in('row_fp32_adapter_fixed84.py','full_row_fp32_fixed84.py'):
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),source[name]['sha256'])
        v=json.loads((ROOT/'final_verification84.json').read_text())
        self.assertEqual(len(v['production']),23);self.assertEqual(len(v['installed_api_sources']),13)
        self.assertFalse(v['remaining_workers']);self.assertTrue(v['all_production_reference_and_launcher_unchanged'])


if __name__=='__main__':unittest.main()
