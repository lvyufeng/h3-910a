#!/usr/bin/env python3
"""Portable contracts for the explicitly approved local numeric candidate test."""
import ast
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_approval_local_only(self):
        a=json.loads((ROOT/'approval134.json').read_text());self.assertEqual(a['selected'],'仅做局部候选试验 (Recommended)')
        for k in ('baseline_adoption','new_numeric_tolerance','full50_regression','generation49_or_decoded_media','copy_candidate'):self.assertFalse(a[k])
    def test_Half_parameters_and_single_output_narrowing(self):
        s=(ROOT/'fp32_bias_candidate134.py').read_text()
        self.assertIn('x.dtype==layer.weight.dtype==torch.float16',s)
        self.assertIn('torch.mm(x,layer.weight.t(),out_dtype=torch.float32)',s)
        self.assertIn('output.add_(bias.float())',s)
        self.assertEqual(s.count('output.to(torch.float16)'),1)
        self.assertNotIn('bfloat16',s);self.assertNotIn('weight.data=',s);self.assertNotIn('clamp',s)
    def test_original_candidate_restored_cases(self):
        s=(ROOT/'candidate_probe134.py').read_text()
        self.assertIn("for case in ('original','candidate','restored')",s)
        self.assertIn("active_apply=candidate.apply if case=='candidate' else old",s)
        self.assertIn('range(32)',s);self.assertIn('range(128)',s)
        self.assertIn('candidate.calls==160',s)
    def test_original_same_input_and_reference_gate(self):
        s=(ROOT/'candidate_probe134.py').read_text()
        self.assertIn('assert actual==ref',s);self.assertIn('source_input_SHA_verified',s)
        self.assertIn('original_vs_frozen_input',s);self.assertIn('same_python_default_dispatch_as_R132_audio_head',s)
        self.assertIn('set_current_vllm_config(native_vllm)',s)
    def test_numeric_modes_and_physical_restore_separate(self):
        s=(ROOT/'candidate_probe134.py').read_text()
        self.assertIn('output_vs_case_first',s);self.assertIn('output_vs_original_first',s)
        self.assertIn('physical_restore',s);self.assertIn('parameters_after',s)
        self.assertIn('wrapper_installed',s);self.assertIn("delattr(method,'apply')",s)
        self.assertIn('torch.use_deterministic_algorithms(initial,warn_only=warn)',s)
    def test_allrank_gates_no_fullmodel(self):
        s=(ROOT/'candidate_probe134.py').read_text()
        self.assertIn('cohort.run',s);self.assertIn('publish_fatal128',s)
        self.assertNotIn('MiniMaxH3DiTModel',s);self.assertNotIn('.generate(',s)
        self.assertNotIn('NativeCopyAdapter',s)
    def test_PIDfd_deadline_and_uncertain_replay_guard(self):
        s=(ROOT/'launch_candidate134.py').read_text();abort=(ROOT/'private_abort134.py').read_text()
        self.assertIn('deadline=time.monotonic()+1200',s);self.assertIn('not receipt.exists()',s)
        self.assertIn('candidate_rank*134.json',s);self.assertIn('approval134.json',s)
        self.assertIn('pidfd_send_signal',abort);self.assertNotIn('os.kill(',abort)
        self.assertIn('/tmp/h3_adaln13401',abort)


if __name__=='__main__':unittest.main()
