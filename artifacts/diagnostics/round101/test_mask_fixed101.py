"""Kernel phantom-mask correction targets the full Float buffer, not a one-element slice."""
from pathlib import Path
import importlib.util
import unittest
ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('bc128fixed101',ROOT/'flash_vision_mask_fixed101.py')
KERNEL=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(KERNEL)


class MaskRegression101(unittest.TestCase):
    def test_mask_addresses_all4096_phantom_elements_from_original_base(self):
        class Base:
            def __getitem__(self,offset):return offset
        class Recorder:
            def __init__(self):self.calls=[]
            def vector_dup(self,*args):self.calls.append(args)
        plan=KERNEL.plan_kernel();t=Recorder()
        count=KERNEL._mask_phantom_scores(t,Base(),plan,-3.4028234663852886e38)
        self.assertEqual(count,64)
        self.assertEqual([a[1]for a in t.calls],list(range(4096,8192,64)))
        self.assertTrue(all(a[0]==64 and a[3:]==(1,1,1)for a in t.calls))
        self.assertEqual(t.calls[-1][1]+64,8192)
    def test_no_mask_if_key_count_fills_complete_blocks(self):
        class Recorder:
            def vector_dup(self,*args):raise AssertionError('no phantom write expected')
        self.assertEqual(KERNEL._mask_phantom_scores(Recorder(),None,KERNEL.plan_kernel(batch=2,m=128,n=256),0.0),0)
    def test_original_failure_and_new_direct_base_source_are_retained(self):
        original=(ROOT/'flash_bc128/flash_vision101.py').read_text()
        fixed=(ROOT/'flash_vision_mask_fixed101.py').read_text()
        self.assertIn('_fill_flat(t, s_f_flat[start], value',original)
        self.assertNotIn('_fill_flat(t, s_f_flat[start], value',fixed)
        self.assertIn('t.vector_dup(F32_MASK, s_f_flat[start+offset], value, 1, 1, 1)',fixed)
        self.assertEqual(KERNEL.plan_kernel()['buffers']['ub_total_bytes'],190848)


if __name__=='__main__':unittest.main()
