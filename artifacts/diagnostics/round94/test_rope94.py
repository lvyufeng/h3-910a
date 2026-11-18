"""Real original geometry/FP16 rotary-only audit, no full-encoder acceptance."""
from pathlib import Path
import json
import hashlib
import ast
import unittest
ROOT=Path(__file__).resolve().parent


class SourceContracts(unittest.TestCase):
    def test_no_linear_model_or_teacher_created(self):
        s=(ROOT/'rope_probe94.py').read_text();ast.parse(s)
        self.assertNotIn('nn.Linear(',s);self.assertNotIn('Qwen3VLModel(',s)
        self.assertIn("'no_Linear_weight_or_teacher':True",s)
        self.assertIn('CPUrot=hf.Qwen3VLTextRotaryEmbedding(',s)
        self.assertIn('hf.Qwen3VLVisionRotaryEmbedding(36)',s)
    def test_original_real_positions_and_grid_used(self):
        s=(ROOT/'rope_probe94.py').read_text()
        self.assertIn("previous.CPU_ROOT/'FP16_multimodal_inputs.pt'",s)
        self.assertIn('positions.shape==(3,1,2043)',s)
        self.assertIn('[[1,48,84],[1,48,84]]',s)
        self.assertIn("r['vision_frequency_vs_CPU']['raw_bits_equal']",s)
        self.assertIn("r['text_inv_freq_CPU_vs_candidate']['raw_bits_equal']",s)
    def test_no_speed_or_precision_waiver(self):
        s=(ROOT/'rope_probe94.py').read_text()
        self.assertIn("'not_full_encoder_timing':True",s)
        self.assertIn("'full_encoder_precision_accepted':False",s)
        self.assertIn("'numerical_tolerance':None",s)
        self.assertIn("'R93_CPU_control_concurrent_no_speed_comparison':True",s)


class ActualEvidence(unittest.TestCase):
    def test_repeats_original_sources_and_geometry_gates(self):
        r=json.loads((ROOT/'rope94.json').read_text())
        self.assertTrue(r['completed']and r['original_positions_SHA_unchanged']and r['original_HF_candidate_CPUgold_unchanged'])
        for name in('text_actual_Half','vision_actual_Float'):
            self.assertTrue(r[name]['cos_repeat']['raw_bits_equal']and r[name]['sin_repeat']['raw_bits_equal'])
        self.assertTrue(r['vision_frequency_vs_CPU']['raw_bits_equal'])
        self.assertTrue(r['text_inv_freq_CPU_vs_candidate']['raw_bits_equal'])
        self.assertEqual(hashlib.sha256((ROOT/'rope_probe94.py').read_bytes()).hexdigest(),r['source_SHA256'])
    def test_no_math_acceptance_or_unmeasured_encoder_speed(self):
        r=json.loads((ROOT/'rope94.json').read_text())
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertIsNone(r['numerical_tolerance']);self.assertTrue(r['not_full_encoder_timing'])


if __name__=='__main__':unittest.main()
