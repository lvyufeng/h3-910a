"""The hardware-corrected guard replaces only unsupported Float vec_relu."""
from pathlib import Path
import importlib.util
import math
import unittest
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('erf_fixed90_contract',ROOT/'erf_gelu_fixed90.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class FixedFloatGuard(unittest.TestCase):
    def test_emission_has_no_unsupported_relu_and_keeps_one_final_Half(self):
        ops=m.emission_steps(m.plan_tiling(8192))
        self.assertNotIn('vec_relu',[x['op']for x in ops])
        self.assertEqual(len(m.narrowing_steps(m.plan_tiling(8192))),1)
        k=[x for x in ops if x['op']=='vec_abs'and x['dst']=='e_f']
        self.assertEqual(len(k),1)
    def test_geometry_abi_and_scalar_model_unchanged(self):
        p=m.plan_tiling(9289728)
        self.assertEqual(p['grid']['cores'],30)
        self.assertEqual(p['grid']['tasks'],2268)
        for x in(-65504.,-100.,-8.,0.,2**-24,1.,8.,65504.):
            self.assertTrue(math.isfinite(m.scalar_emulation(x)))
        self.assertEqual(m.scalar_emulation(-65504.),0.)
    def test_fixed_consumers_use_distinct_new_root_and_module(self):
        for name in('kernel_probe_fixed90.py','vision_probe_fixed90.py'):
            text=(ROOT/name).read_text()
            self.assertIn('/tmp/h3_round90_vision02',text)
            self.assertIn('erf_gelu_fixed90',text)
        self.assertIn('/tmp/h3_round90_vision01',(ROOT/'kernel_probe90.py').read_text())


if __name__=='__main__':unittest.main()
