#!/usr/bin/env python3
"""Verify gate-only replay data, exact leaf scope and no adoption/composition."""
import ast
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent

def load(name):
    return json.loads((HERE/name).read_text())

class Contracts(unittest.TestCase):
    def test_gate_source_only_returns_original(self):
        tree=ast.parse((HERE/'gate_stage219.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        attention=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_attention')
        values=[n.value for n in ast.walk(attention) if isinstance(n,ast.Return)]
        self.assertTrue(values)
        self.assertTrue(all(isinstance(v,ast.Name) and v.id=='reference' for v in values))
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='enable_measurement' for n in cls.body))

    def test_same_kernel_as_prior_candidate(self):
        summary=load('gate_summary219.json')
        self.assertTrue(summary['all_step0_48_gate_passed'])
        for row in summary['candidate_kernel']:
            self.assertEqual(row['sha256'],'052c49f2df2235bac2136b4630a05a71bd1b9b454d88c354bbbf5e254ab6e327')
            self.assertEqual((row['instance']['Tr'],row['instance']['Tc']),(327,138))
            self.assertFalse(row['instance']['has_attn_mask'])

    def test_actual_calls_and_element_coverage(self):
        summary=load('gate_summary219.json')
        self.assertEqual(summary['total_compared_calls'],800)
        self.assertEqual(summary['total_compared_elements'],29956505600)
        self.assertEqual(summary['total_differing_words'],0)
        self.assertEqual(summary['including_prior_step25_calls'],1200)
        self.assertEqual(summary['including_prior_step25_elements'],44934758400)
        self.assertEqual({(g['rank'],g['step']) for g in summary['gates']},
                         {(rank,step) for rank in range(8) for step in (0,48)})
        for gate in summary['gates']:
            self.assertEqual(gate['strict_checks'],155)
            self.assertTrue(gate['original_outputs_returned'])
            self.assertEqual(gate['candidate_calls'],50)
            self.assertEqual(gate['different_words'],0)

    def test_no_adoption_or_new_speed_claim(self):
        summary=load('gate_summary219.json')
        self.assertEqual(summary['candidate_outputs_used_by_model'],0)
        self.assertFalse(summary['candidate_default_enabled'])
        self.assertTrue(summary['no_adoption_or_composition'])
        self.assertFalse(summary['whole_goal_complete'])
        self.assertIn('Awaiting',summary['decision'])
        self.assertAlmostEqual(summary['previous_measured_step_gain_s_not_remeasured'],0.5682868957519531)

    def test_extreme_and_boundary_results_are_honest(self):
        # A failure here is evidence, not permission to relax word equality.
        edge=load('edge_summary219.json')
        self.assertFalse(edge['candidate_enabled_or_adopted'])
        self.assertFalse(edge['nonfinite_input_injection'])
        self.assertTrue(edge['stop_on_word_mismatch_or_nonfinite'])
        self.assertGreater(len(edge['baseline_cases']),0)
        self.assertGreater(len(edge['candidate_cases']),0)
        for case in edge['candidate_cases']:
            if edge['all_executed_cases_match']:
                self.assertEqual(case['differing_words'],0)
                self.assertEqual(case['nonfinite_words'],0)

    def test_cleanup_and_restore(self):
        receipt=load('cleanup219.json')
        self.assertTrue(receipt['original_sources_unchanged'])
        self.assertFalse(receipt['private_model_or_test_processes'])
        self.assertTrue(receipt['candidate_private_caches_removed'])
        self.assertTrue(receipt['host_only_tensor_outputs_removed'])
        self.assertTrue(receipt['no_device_or_sdk_reset'])
        self.assertIn('released=true',receipt['lease_release'])

    def test_sources_parse(self):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(),filename=str(path))

if __name__=='__main__':
    unittest.main()
