#!/usr/bin/env python3
"""Portable creation-contract tests; never perform API, model or resource actions."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SPEC = json.loads((ROOT / 'roadmap_spec.json').read_text())


class RoadmapContracts(unittest.TestCase):
    def test_target_is_existing_authorized_repository(self):
        self.assertEqual(SPEC['repo'], 'lvyufeng/h3-910a')
        self.assertEqual(SPEC['owner'], 'lvyufeng')

    def test_issue_keys_and_titles_are_unique(self):
        self.assertEqual(len(SPEC['issues']), 13)
        for field in ('key', 'title'):
            self.assertEqual(len({i[field] for i in SPEC['issues']}), 13)

    def test_dependency_order_is_acyclic(self):
        seen = set()
        for item in SPEC['issues']:
            self.assertTrue(set(item['dependencies']).issubset(seen), item['key'])
            seen.add(item['key'])
        self.assertEqual(sum(len(i['dependencies']) for i in SPEC['issues']), 21)

    def test_approximate_branch_is_not_an_integration_dependency(self):
        items = {i['key']: i for i in SPEC['issues']}
        self.assertEqual(items['optional-numeric']['status'], 'Awaiting approval')
        self.assertEqual(items['optional-numeric']['precision'], '先批准再执行')
        self.assertNotIn('optional-numeric', items['integration']['dependencies'])

    def test_only_previously_completed_baseline_is_done(self):
        self.assertEqual([i['key'] for i in SPEC['issues'] if i['status'] == 'Done'], ['baseline'])
        self.assertEqual(SPEC['issues'][0]['status'], 'In Progress')

    def test_every_issue_has_acceptance_and_source_evidence(self):
        for item in SPEC['issues']:
            self.assertGreaterEqual(len(item['work']), 4, item['key'])
            self.assertGreaterEqual(len(item['acceptance']), 3, item['key'])
            self.assertTrue(item['sources'], item['key'])
            for source in item['sources']:
                if not source.startswith('https://'):
                    self.assertTrue((WORKSPACE / source).is_file(), source)

    def test_known_scopes_and_precision_contracts_are_retained(self):
        analysis = (ROOT / 'PLAN.md').read_text()
        for term in ('TP2', 'USP4', 'TP8', '155', '60760', '19.372', '17.556',
                     '142.143', '20.565', '2.783', '1.057', '4090', '原生'):
            self.assertIn(term, analysis)

    def test_scripts_parse_and_default_to_no_writes(self):
        for filename in ('create_issues.py', 'create_project.py'):
            text = (ROOT / filename).read_text()
            ast.parse(text, filename=filename)
            self.assertIn("'--apply'", text)
            self.assertIn('if not args.apply:', text)

    def test_issue_script_reconciles_before_creating_native_edges(self):
        source = (ROOT / 'create_issues.py').read_text()
        self.assertIn("matches = [v for v in existing if marker(item['key'])", source)
        self.assertIn("if entries[key]['id'] not in current:", source)
        self.assertIn('Re-read GitHub state before any retry', source)
        self.assertIn('state_reason', source)

    def test_project_does_not_reset_recorded_live_status(self):
        source = (ROOT / 'create_project.py').read_text()
        self.assertIn("if name == 'Status' and item['key'] in recorded_items:", source)
        self.assertIn('MARKER in', source)
        self.assertIn('BOARD_LAYOUT', source)

    def test_workspace_rules_are_persistent(self):
        agent = (WORKSPACE / 'AGENTS.md').read_text()
        rule = (WORKSPACE / '.cursor/rules/github-execution.mdc').read_text()
        self.assertIn('GitHub Project + Issues', agent)
        self.assertIn('alwaysApply: true', rule)
        self.assertLess(len(rule.splitlines()), 50)
        self.assertIn('TP stays ON', agent)


if __name__ == '__main__':
    unittest.main()
