"""Actual CPUFP16 attention evidence and precise full-graph source contracts."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class AttentionEvidence(unittest.TestCase):
    def test_all8_real_QKV_and_CPU_Half_MATH_references_are_repeat_verified(self):
        for rank in range(8):
            r=json.loads((ROOT/'results/capture01/ranks'/('tp8_rank%d.json'%rank)).read_text())
            self.assertTrue(r['completed']);self.assertTrue(r['parameters_unchanged'])
            self.assertTrue(r['CPU_reference_receipt_unchanged']);self.assertTrue(r['raw_decoder0_reproduces_R84_native_CPU_input'])
            self.assertTrue(r['CPU_Half_MATH_repeat']['raw_bits_equal'])
            self.assertTrue(all(x['raw_bits_equal']for x in r['operand_repeat'].values()))
            self.assertEqual(r['capture']['q']['shape'],[1,8,2043,128])
            self.assertEqual(r['capture']['k']['shape'],[1,1,2043,128])
            self.assertEqual(r['capture']['q']['dtype'],'torch.float16')
    def test_PV_still_diverges_when_probabilities_are_exact_CPU_values(self):
        for rank in range(8):
            r=json.loads((ROOT/'results/capture01/ranks'/('tp8_rank%d.json'%rank)).read_text())
            d=r['first_512_query_leaf_diagnostics']
            self.assertGreater(d['PV_exact_CPU_probabilities']['relative_rms'],100*d['scores_unmasked']['relative_rms'])
            self.assertGreater(d['PV_exact_CPU_probabilities']['relative_rms'],50*d['probabilities']['relative_rms'])
    def test_actual_batched_attention_reduces_same_CPU_Half_error(self):
        r=json.loads((ROOT/'results/batch01/attention_result85.json').read_text())
        self.assertTrue(r['completed']and r['kernel_launch_executed'])
        self.assertTrue(r['custom_repeat_bits']['raw_bits_equal']);self.assertTrue(r['original_operands_unchanged'])
        self.assertGreater(r['saved_native_vs_CPU_Half_MATH']['relative_rms']/r['custom_vs_CPU_Half_MATH']['relative_rms'],15)
        self.assertEqual(r['actual_CPU_Half_MATH_reference_sha256'],r['saved_native_vs_CPU_Half_MATH']['cpu']['sha256'])
        self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'probe_batch_attention85.py').read_bytes()).hexdigest())
        self.assertEqual(r['kernel_source_sha256'],hashlib.sha256((ROOT/'batch/batch_gemm85.py').read_bytes()).hexdigest())
        self.assertTrue(r['smoke_repeat_bits']['raw_bits_equal'])
        for c in r['query_chunks']:
            self.assertLess(c['P_reconstruction']['relative_rms'],1e-7)
            self.assertLess(c['PV_vs_CPU_Float']['relative_rms'],1e-6)
        self.assertFalse(r['full_encoder_precision_accepted']);self.assertFalse(r['whole_goal_complete'])
    def test_reusable_engine_does_not_cache_different_input_activations_as_weights(self):
        text=(ROOT/'attention_engine85.py').read_text();ast.parse(text)
        self.assertIn('Once per real attention call',text)
        self.assertIn('high=p.to(torch.float16);low=((p-high.float())*1024)',text)
        self.assertIn('assert out.dtype==torch.float32',text)
        self.assertIn("getattr(current,'__self__',None)is self",text)
    def test_full_graph_source_has_native_and_precise_controls_and_recovery(self):
        text=(ROOT/'full_attention85.py').read_text();ast.parse(text)
        for name in('native_rows_CPU_condition','accurate_attention_native_rows_CPU_condition',
                    'accurate_attention_FP32_rows_CPU_condition','accurate_attention_FP32_rows_native_condition'):
            self.assertIn(name,text)
        self.assertIn('attention_engine.set_enabled(False)\n        with torch.inference_mode():restored_native',text)
        self.assertIn('compare(stock, first_hidden_for_stock)',text)
        self.assertIn('accurate_attention_uninstalled',text)
        self.assertIn('/tmp/h3_round85_full01',text)


if __name__=='__main__':unittest.main()
