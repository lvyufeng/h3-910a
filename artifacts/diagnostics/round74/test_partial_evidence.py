"""Actual captured/partial-reference evidence cannot be promoted to full quality proof."""
import hashlib
import json
from pathlib import Path
import unittest
from summarize_capture74 import summarize
from summarize_reference74 import aggregate

ROOT=Path(__file__).resolve().parent


class ReferenceProgressGates(unittest.TestCase):
    def test_current_experimental_capture_is_complete_and_exact_not_older_layer_fixture(self):
        r=summarize();self.assertEqual(r['complete_outputs_verified'],8);self.assertEqual(r['current_block_fixtures'],6)
        self.assertEqual(r['selected_layers'],[0,1,2,9,24,49]);self.assertEqual(r['total_tensor_bytes'],125521920)
        self.assertEqual(r['complete_output_sha256'],'67ef196e33bcaa495420da1ebc07d1ff6dcaf89ddbaf75a350940ed8ae210eee')
        self.assertTrue(r['all_current_intermediate_identity']);self.assertFalse(r['official_goal_complete'])
    def test_actual_partial_hf_reference_cannot_pass_full_reference_gate(self):
        partial=json.loads((ROOT/'progress'/'hf_snapshot01.json').read_text())
        self.assertFalse(partial['full_multimodal_reference_completed']);self.assertNotEqual(len(partial['language_layers']),50)
        with self.assertRaisesRegex(ValueError,'not complete'):aggregate(partial,{}, {},{},summarize())
    def test_actual_reference_uses_unsharded_original_bf16_weights_not_native_tp8_count(self):
        partial=json.loads((ROOT/'progress'/'hf_snapshot01.json').read_text())
        self.assertEqual(partial['loaded_parameter_tensors'],903);self.assertEqual(partial['parameter_dtypes'],{'torch.bfloat16':903})
        self.assertEqual(partial['complete_parameter_bytes'],51506202080);self.assertEqual(partial['unchanged_linear_modules'],466)
        self.assertTrue(partial['reference_is_original_bf16_HF']);self.assertFalse(partial['shared_native_visual_features_used'])
        self.assertFalse(partial['fp32_linear_adapter_used'])
    def test_fresh_hf_vision_matches_retained_independent_reference_bits_on_full_tensors(self):
        r=json.loads((ROOT/'vision_repeat01.json').read_text())
        self.assertTrue(r['completed']);self.assertTrue(r['all_full_vision_outputs_bits_equal'])
        for proof in[r['comparisons']['merged'],*r['comparisons']['deepstack'],r['comparisons']['positions']]:
            self.assertTrue(proof['raw_bits_equal']);self.assertTrue(proof['finite']);self.assertEqual(proof['max_abs_error'],0)
            self.assertEqual(proof['fresh_tensor_sha256'],proof['retained_tensor_sha256'])
    def test_fresh_experimental_vision_gap_does_not_become_quality_pass(self):
        r=json.loads((ROOT/'progress'/'hf_snapshot01.json').read_text())
        self.assertTrue(r['embedding_vs_experimental']['raw_bits_equal'])
        self.assertTrue(r['multimodal_text_input_vs_experimental']['position_ids']['raw_bits_equal'])
        self.assertTrue(r['multimodal_text_input_vs_experimental']['visual_pos_masks']['raw_bits_equal'])
        self.assertFalse(r['vision_vs_experimental']['merged']['raw_bits_equal'])
        self.assertGreater(r['vision_vs_experimental']['merged']['relative_rms'],.05)
        self.assertFalse(r['official_goal_complete'])
    def test_actual_installed_code_input_and_executed_source_hashes(self):
        r=json.loads((ROOT/'provenance01.json').read_text());p=json.loads((ROOT/'progress'/'hf_snapshot01.json').read_text())
        self.assertEqual(r['installed_sources']['transformers.models.qwen3_vl.modeling_qwen3_vl']['sha256'],hashlib.sha256((ROOT/'source'/'modeling_qwen3_vl.py').read_bytes()).hexdigest())
        self.assertEqual(p['source_sha256'],hashlib.sha256((ROOT/'hf_multimodal74.py').read_bytes()).hexdigest())
        self.assertEqual(p['input_file_sha256'],r['reference_fixtures']['/tmp/h3_round54_capture01/encoder_input.pt']['sha256'])
        self.assertEqual(r['excluded_checkpoint_language_layers'],list(range(50,64)))
        self.assertTrue(r['exclusions_are_selected_hidden_state_boundary_not_missing_weights'])


if __name__=='__main__':unittest.main()
