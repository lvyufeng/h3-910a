#!/usr/bin/env python3
"""Contracts for the rewritten consolidated status document."""
import re,unittest
from pathlib import Path
ROOT=Path('/Users/lvyufeng/Projects/h3-910a/artifacts/diagnostics')
DOC=ROOT/'PERFORMANCE_STATUS.md'

class Tests(unittest.TestCase):
    def test_exists_and_is_current(self):
        import re
        text=DOC.read_text()
        # version-agnostic: the header must name a round number, and it must not be the stale R139 one
        m=re.search(r'截至 R(\d+)',text)
        self.assertIsNotNone(m,'the header must state which round it is current to')
        self.assertGreaterEqual(int(m.group(1)),167)
        self.assertNotIn('截至 R139',text)

    def test_kernel_phase_is_recorded(self):
        text=DOC.read_text()
        self.assertIn('内核阶段',text)
        self.assertIn('0.351 s',text)
        self.assertIn('2.52 s',text)
        self.assertIn('overlay179.sh',text)
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
    def test_both_chains_are_stated(self):
        text=DOC.read_text()
        # the diagnostic chain may reach below ten, the compliant one must be shown as still short
        self.assertIn('诊断链',text)
        self.assertIn('合规链',text)
        self.assertIn('11.84 s',text)
        self.assertIn('仍差 1.84 s',text)
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
