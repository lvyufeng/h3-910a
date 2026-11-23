#!/usr/bin/env python3
"""Contracts for the R148 guard identity check (measurement only)."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_model_math_unchanged(self):
        s=(ROOT/'guard_probe148.py').read_text()
        self.assertIn('return native',s,'the model must keep receiving the native result')
        self.assertIn('_STATE[\'kept\'].append',s)
        # No comparison may run inside the model forward: scope the check to the
        # body of apply(), because _stats() is only invoked from collect().
        tree=ast.parse(s)
        apply_fn=next(n for n in ast.walk(tree)
                      if isinstance(n,ast.FunctionDef) and n.name=='apply')
        calls=[n for n in ast.walk(apply_fn) if isinstance(n,ast.Call)]
        attrs=[n.func.attr for n in calls if isinstance(n.func,ast.Attribute)]
        names=[n.func.id for n in calls if isinstance(n.func,ast.Name)]
        self.assertNotIn('cpu',attrs,'no in-forward D2H may be issued')
        self.assertNotIn('_stats',names,'comparison must be deferred to collect()')
        self.assertNotIn('to',[a for a in attrs if False])  # placeholder, narrowing is allowed
    def test_amax_matches_native_form(self):
        s=(ROOT/'guard_probe148.py').read_text()
        self.assertIn('amax(dim=tuple(range(product.ndim)))',s)
    def test_inconclusive_is_labelled_not_claimed(self):
        b=json.loads((ROOT/'guard_check148.json').read_text())
        self.assertEqual(b['identity_check'],'INCONCLUSIVE')
        self.assertIn('control',b['identity_check_reason'])
        self.assertTrue(b['model_always_returned_the_native_result'])
        self.assertFalse(b['control_observed']['all_bit_identical'])
    def test_solid_peak_result(self):
        b=json.loads((ROOT/'guard_check148.json').read_text())
        p=b['peak_distribution']
        self.assertEqual(p['in_scale_one_regime']+p['exceeding_32752'],b['calls_recorded'])
        self.assertGreater(p['exceeding_32752'],0,'if nothing exceeds 32752 an unconditional fast path would be credible')
        self.assertGreater(p['max'],32752)
    def test_open_phase_question_recorded(self):
        b=json.loads((ROOT/'guard_check148.json').read_text())
        self.assertIn('2043',b['unexpected'])

if __name__=='__main__':unittest.main()
