#!/usr/bin/env python3
"""Contracts for the rewritten consolidated status document."""
import re,unittest
from pathlib import Path
ROOT=Path('/Users/lvyufeng/Projects/h3-910a/artifacts/diagnostics')
DOC=ROOT/'PERFORMANCE_STATUS.md'

class Tests(unittest.TestCase):
    def test_exists_and_is_current(self):
        text=DOC.read_text()
        self.assertIn('截至 R167',text)
        self.assertNotIn('截至 R139',text)
    def test_dit_target_is_ten_seconds(self):
        text=DOC.read_text()
        self.assertIn('<10 s/step',text)
        self.assertIn('上调',text)
    def test_the_audit_finding_is_prominent(self):
        text=DOC.read_text()
        self.assertIn('2.87 s',text)
        self.assertIn('15.035 s',text)
        self.assertIn('验证脚手架',text)
    def test_contract_versus_model_distinction_is_made(self):
        text=DOC.read_text()
        self.assertIn('合同口径',text)
        self.assertIn('不合规',text)
    def test_feasibility_scenario_reaches_below_ten(self):
        text=DOC.read_text()
        self.assertIn('8.3 s',text)
        self.assertIn('69 TFLOPS',text)
    def test_retired_hypotheses_are_listed(self):
        text=DOC.read_text()
        for marker in ('拷贝停顿','hook 开销','分配器','tiling'):
            self.assertIn(marker,text)
    def test_do_not_repeat_section_keeps_the_lessons(self):
        text=DOC.read_text()
        self.assertIn('synchronize()',text)
        self.assertIn('位模式',text)
        self.assertIn('可加分解',text)
    def test_candidates_have_states(self):
        text=DOC.read_text()
        self.assertIn('Awaiting approval',text)
        self.assertIn('INCONCLUSIVE',text)

if __name__=='__main__':unittest.main()
