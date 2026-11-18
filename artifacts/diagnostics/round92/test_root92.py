"""Actual prefix provenance and one-block CPU/NPU isolation contracts."""
from pathlib import Path
import ast
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class Block25IsolationContracts(unittest.TestCase):
    def test_prefix_stops_after_actual_raw24_and_requires_R91_SHA(self):
        text=(ROOT/'prefix_capture92.py').read_text();ast.parse(text)
        self.assertIn('encoder.text_model.layers[24].register_forward_hook(raw24_post)',text)
        self.assertIn('raise PrefixCaptured()',text)
        self.assertIn("info['sha256']==expected[name]['sha256']",text)
        self.assertIn("'no_raw25to49_executed':True",text)
        self.assertIn('for _ in range(2):',text)
        self.assertIn("'position_ids':cpu_inputs['position_ids']",text)
    def test_only_original_layer25_names_are_constructed_and_loaded(self):
        text=(ROOT/'same_input_block25_92.py').read_text();ast.parse(text)
        self.assertIn("encoder.text_model.layers=torch.nn.ModuleDict({'25':",text)
        self.assertIn("name.startswith('model.language_model.layers.25.')",text)
        self.assertIn('expected_layer_count=2',text)
        self.assertIn('SimpleNamespace(layers=list(encoder.text_model.layers.values()))',text)
        self.assertIn("'no_DeepStack_or_final_norm_inserted':True",text)
        self.assertIn('cfg.text_config.intermediate_size==25600',text)
    def test_four_independent_math_modes_same_original_Half_inputs(self):
        text=(ROOT/'same_input_block25_92.py').read_text()
        for case in("('native',False,False)","('precise_attention_native_rows',True,False)","('native_attention_FP32_rows',False,True)","('precise_attention_FP32_rows',True,True)"):
            self.assertIn(case,text)
        self.assertIn("'same_input_vs_original_CPU_Half_block25':vs_same_CPU",text)
        self.assertIn("'vs_original_CPU_layer25'",text)
        self.assertIn("previous.compare(value,cpu_input)['raw_bits_equal']",text)
        self.assertIn("'native_recovery_and_input_SHA_unchanged'",text)
    def test_original_CPU_block25_gate_is_required_by_NPU_consumer(self):
        text=(ROOT/'same_input_block25_92.py').read_text()
        self.assertIn("proof['language_layers'][25]['output']['sha256']",text)
        self.assertIn("cpu_receipt['outputs_file_SHA256']",text)
        self.assertIn("'not_full_encoder_timing':True",text)
        self.assertIn("'numerical_tolerance':None",text)
    def test_cpu_retry_explicit_callables_no_None_shadow_and_same_oracle_math(self):
        text=(ROOT/'run_cpu_reference_fixed92.py').read_text();ast.parse(text)
        self.assertIn('original.run_reference(root=ROOT,compare=metrics.compare,tensor_info=metrics.tensor_info)',text)
        self.assertIn("ROOT=Path('/tmp/h3_round92_case02')",text)
        self.assertIn('838027c883bed6d1e5a0b839e0c2accde2457bef9cc81d58129fdcad38d65f4b',text)
        self.assertIn("'original_CPU_reference_gate_and_math_not_changed':True",text)

    def test_pinned_R91_raw24_inputs_cover_exact_three_cases(self):
        expected=json.loads((ROOT/'expected_raw24_92.json').read_text())
        self.assertEqual(set(expected['cases']),{'native','corrected_vision_native_text','corrected_vision_precise_text_geometry_cache'})
        for info in expected['cases'].values():
            self.assertEqual(info['shape'],[1,2043,5120]);self.assertEqual(info['dtype'],'torch.float16')
        self.assertEqual(expected['R91source_SHA256'],hashlib.sha256((ROOT.parent/'round91/full_encoder91.py').read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
