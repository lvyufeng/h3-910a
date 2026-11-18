"""Actual original CPU Half and same-input TP8 one-block proofs."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class ActualBlock25Evidence(unittest.TestCase):
    def test_prefix_all8_three_states_reproduce_R91(self):
        expected=json.loads((ROOT/'expected_raw24_92.json').read_text())['cases']
        for rank in range(8):
            r=json.loads((ROOT/'results/prefix01/ranks'/('prefix_rank%d.json'%rank)).read_text())
            self.assertTrue(r['completed']and r['parameter_and_fixture_SHA_unchanged']and r['CPU_reference_SHA_unchanged'])
            for name,case in r['cases'].items():
                self.assertEqual(case['raw24']['sha256'],expected[name]['sha256'])
                self.assertTrue(case['repeat_bit_identical']and case['no_raw25to49_executed'])
    def test_original_CPU25_exact_R78_raw25_and_no_teacher(self):
        r=json.loads((ROOT/'results/cpu02/CPU_block25_reference92.json').read_text())
        self.assertTrue(r['completed']);self.assertTrue(r['reference_gate']['matches_R78_raw25_SHA256'])
        self.assertTrue(r['reference_gate']['repeat_bits_equal'])
        self.assertEqual(r['reference_gate']['output']['sha256'],'a83f92f8f9108ff05c25d7967eb575ee9a2b93e7c7463a44b9536595b519c01d')
        self.assertTrue(all(c['repeat_bits_equal']for c in r['cases'].values()))
        self.assertEqual(r['original_layer']['parameter_count'],11)
        self.assertEqual(r['original_layer']['parameter_dtypes'],['torch.float16'])
        self.assertEqual(r['cpu_execution']['threads'],16)
    def test_same_input_all8_all4_math_modes_repeat_and_native_recovery(self):
        rows=[json.loads((ROOT/'results/block25_02/ranks'/('block25_rank%d.json'%rank)).read_text())for rank in range(8)]
        for r in rows:
            self.assertTrue(r['completed']and r['original_parameters_unchanged']and r['CPU_reference_SHA_unchanged'])
            self.assertTrue(r['not_full_encoder_timing']and r['no_DeepStack_or_final_norm_inserted'])
            for c in r['cases'].values():
                self.assertTrue(c['native_recovery_and_input_SHA_unchanged'])
                for mode in('native','precise_attention_native_rows','native_attention_FP32_rows','precise_attention_FP32_rows'):
                    self.assertTrue(c[mode]['repeat_bit_identical']and c[mode]['same_input_vs_original_CPU_Half_block25']['finite'])
        for case in rows[0]['cases']:
            for mode in('native','precise_attention_native_rows','native_attention_FP32_rows','precise_attention_FP32_rows'):
                self.assertEqual(len({r['cases'][case][mode]['same_input_vs_original_CPU_Half_block25']['npu']['sha256']for r in rows}),1)
    def test_summary_separates_input_amp_and_same_input_arithmetic_no_speed_claim(self):
        s=json.loads((ROOT/'summary92.json').read_text());cpu=json.loads((ROOT/'results/cpu02/CPU_block25_reference92.json').read_text())
        self.assertTrue(s['CPU_original_raw25_twice_SHA_exact']and s['not_full_encoder_timing'])
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
        for case,c in s['cases'].items():
            self.assertEqual(c['input_only_original_CPU_control'],cpu['input_error_amplification'].get(case))
        self.assertTrue(s['original_R91_fullvision_combination_not_default'])
    def test_isolated_native_and_precise25_reproduce_R91_continuation(self):
        old=json.loads((ROOT.parent/'round91/results/full01/ranks/tp8_rank0.json').read_text())
        now=json.loads((ROOT/'results/block25_02/ranks/block25_rank0.json').read_text())
        for case in('native','corrected_vision_native_text','corrected_vision_precise_text_geometry_cache'):
            expected=next(row['output']['sha256']for row in old['cases'][case]['raw50_layers']if row['layer']==25)
            mode='precise_attention_native_rows'if'precise_text'in case else'native'
            self.assertEqual(now['cases'][case][mode]['same_input_vs_original_CPU_Half_block25']['npu']['sha256'],expected)

    def test_first_CPU_CLI_failure_retained_without_math_gate_waiver(self):
        fail=json.loads((ROOT/'results/cpu01/CPU_block25_failure92.json').read_text())
        self.assertEqual(fail['stage'],'reference_gate');self.assertEqual(fail['error']['type'],'TypeError')
        self.assertFalse(fail['reference_written']or fail['outputs_written'])
        original=ROOT/'cpu/original_cpu_block25_92.py'
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),fail['source_sha256'])


if __name__=='__main__':unittest.main()
