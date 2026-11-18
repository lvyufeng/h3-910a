"""Real CPUHalfGELU semantics,existingreferenceSHA and safe adapter sequencing."""
from pathlib import Path
import ast
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class ErfControlsContracts(unittest.TestCase):
    def test_preflight_original_CPUHalf_output_SHA_and_modes_not_assumed(self):
        text=(ROOT/'gelu_preflight90.py').read_text();ast.parse(text)
        self.assertIn("expected=old['actual_nonlinear_leaf']['controls'][name]['native_vs_cpu_half']['cpu']['sha256']",text)
        self.assertIn("'native_none_vs_native_tanh'",text)
        self.assertIn("'CPU_tanh_vs_CPU_Half_erf'",text)
        self.assertIn("torch.erf(xn.float()*(2**-.5))",text)
        self.assertIn('with torch.inference_mode():',text)
        self.assertIn('FROZEN_SHA256',text)
    def test_only_four_original_erf_modules_and_no_tanh_module_patch(self):
        text=(ROOT/'torch_erf_adapter90.py').read_text();ast.parse(text)
        self.assertIn("module.approximate=='none'",text)
        for name in('merger.act_fn','deepstack_merger_list.0.act_fn','deepstack_merger_list.1.act_fn','deepstack_merger_list.2.act_fn'):
            self.assertIn(name,text)
        self.assertNotIn('blocks.',text)
        self.assertNotIn('torch.nn.GELU.forward=',text)
        self.assertIn('MethodType(function,module)',text)
    def test_Torch_adapter_restored_before_child_TIK_installed(self):
        text=(ROOT/'vision_probe90.py').read_text();ast.parse(text)
        self.assertLess(text.index('explicit.uninstall()'),text.index('erf=ErfGeluAdapter90('))
        self.assertIn("run_case('native',False,False,False)",text)
        self.assertIn("run_case('patch_and_linear_bias',True,False,False)",text)
        self.assertIn("run_case('TIK_erf_patch_and_bias',True,False,True)",text)
        self.assertIn("'no_CPU_precomputed_entry_in_real_path':True",text)
        self.assertIn("'full_encoder_precision_accepted':False",text)
    def test_kernel_probe_preserves_sameHalf_and_signed_extremes(self):
        text=(ROOT/'kernel_probe90.py').read_text();ast.parse(text)
        self.assertIn('65504.,-65504.',text)
        self.assertIn('0.,-0.,2**-24,-2**-24',text)
        self.assertIn('runtime.launch(meta,[x,aux,aux,aux,out]',text)
        self.assertIn("'CPU_Half_erf':compare",text)
        self.assertNotIn('weight.float()',text)
    def test_R89_executed_patch_and_R87_bias_sources_are_immutable(self):
        for path,pin in((ROOT.parent/'round89/patch_bias89.py','2eb5ff1b6ac741d3bb27e523524a917aeebda791525ac23dffb20b906d45e7f2'),
                         (ROOT.parent/'round87/vision_adapter87.py','8393f6b9c379e0df8cc3c1393ff2b839dab4c535fddd4be14782cec615c8db19')):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pin)


if __name__=='__main__':unittest.main()
