"""Actual originalCPUHalf vision-entry and separate patch/nonlinear A/B evidence."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class ActualVisionEntryEvidence(unittest.TestCase):
    def test_original_CPU_entry_matches_R78_block0_and_failed_record_preserved(self):
        r=json.loads((ROOT/'results/vision02/vision89.json').read_text())
        old=json.loads((ROOT/'results/vision01/vision89.json').read_text())
        self.assertTrue(r['completed']and r['original_HF_CPU_entry_reference']['CPU_block0_matches_original_R78_output_SHA'])
        self.assertEqual(r['original_HF_CPU_entry_reference']['CPU_block0']['sha256'],'1b41bf75401dfbcef77dff545c690857ee4ff7b58a8d49c6eadf13c6fbfdd5ef')
        self.assertEqual(r['original_HF_CPU_entry_reference']['CPU_block0']['sha256'],old['original_HF_CPU_entry_reference']['CPU_block0']['sha256'])
        self.assertFalse(old['completed']);self.assertIn('Inference tensors cannot be saved for backward',old['failure']['message'])
        self.assertTrue(r['actual_entry']['positions_vs_original_CPU_Half']['raw_bits_equal'])
    def test_actual_patch_GEMM_closes_Conv_bias_gap_and_uses_14_launches(self):
        s=json.loads((ROOT/'summary89.json').read_text())
        native=s['actual_entry']['patch_vs_original_CPU_Half_Conv3d']['relative_rms']
        custom=s['actual_patch_onecast_GEMM']['custom_vs_original_CPU_Half_Conv3d']['relative_rms']
        self.assertGreater(native/custom,60)
        self.assertEqual(s['patch_final']['kernel_launches'],14)
        self.assertFalse(s['patch_final']['Float_weight_shadow'])
        self.assertTrue(s['actual_patch_onecast_GEMM']['controls']['original_Half_flatten_prefix_bit_exact'])
        joint=s['full_vision_cases']['patch_and_linear_bias']
        self.assertAlmostEqual(joint['merged_relative_RMS'],.0072464766718495196)
        self.assertLess(joint['resident_vision_median_s'],s['full_vision_cases']['native']['resident_vision_median_s'])
        self.assertFalse(s['full_encoder_precision_accepted'])
    def test_Float_widening_has_no_full_graph_gain_and_merger_erf_gap_remains(self):
        r=json.loads((ROOT/'results/vision02/vision89.json').read_text());native=r['cases']['native']
        for name in('Float_norm_only','Float_GELU_only','Float_norm_and_GELU'):
            c=r['cases'][name]
            self.assertEqual(c['merged_vs_CPU']['npu']['sha256'],native['merged_vs_CPU']['npu']['sha256'])
            self.assertEqual([a['npu']['sha256']for a in c['DeepStack_vs_CPU']],[a['npu']['sha256']for a in native['DeepStack_vs_CPU']])
        for name in('deepstack_merger_list.0.act_fn','merger.act_fn'):
            c=r['actual_nonlinear_leaf']['controls'][name]
            self.assertEqual(c['approximate'],'none')
            self.assertGreater(c['native_vs_cpu_half']['relative_rms'],.0003)
            self.assertEqual(c['native_vs_cpu_half']['relative_rms'],c['npu_float_op_then_half_vs_cpu_half']['relative_rms'])
            self.assertLess(c['cpu_float_op_then_half_vs_cpu_half']['relative_rms'],.000002)
    def test_all_11_cases_repeat_parameters_and_native_recovered(self):
        r=json.loads((ROOT/'results/vision02/vision89.json').read_text())
        self.assertEqual(len(r['cases']),11);self.assertEqual(r['rebase_calls'],12)
        for c in r['cases'].values():
            self.assertEqual(len(c['raw27_layers']),27)
            self.assertTrue(all(a['merged']['raw_bits_equal']and all(b['raw_bits_equal']for b in a['deepstack'])for a in c['repeat_bits']))
        self.assertTrue(r['native_recovery']['merged']['raw_bits_equal'])
        self.assertTrue(r['original_parameters_unchanged']and r['pixels_unchanged']and r['CPU_entry_NPU_transfer_and_all_calls_unchanged']and r['CPU_reference_SHA_unchanged'])
        self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'vision_probe_fixed89.py').read_bytes()).hexdigest())
    def test_original_HF_and_binary_provenance_and_worker_cleanup(self):
        v=json.loads((ROOT/'final_verification89.json').read_text())
        self.assertEqual(len(v['production']),23);self.assertEqual(len(v['installed_api_sources']),13)
        self.assertEqual(v['remaining_workers'],[]);self.assertTrue(v['original_HF_modeling_source']['unchanged'])
        src=json.loads((ROOT/'executed_sources89.json').read_text());s=json.loads((ROOT/'summary89.json').read_text())
        self.assertEqual(src['files']['/tmp/h3_round89_vision02/CPUEntryAndActualLeaves89.pt']['sha256'],s['frozen_inputs_SHA256'])
        self.assertTrue(any(x['sha256']==s['patch_final']['kernel_object_SHA256']for x in src['kernel_objects'].values()))


if __name__=='__main__':unittest.main()
