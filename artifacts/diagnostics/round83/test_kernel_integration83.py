"""Pin the reviewed stride, accumulation and workspace policies before integration."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class KernelIntegrationContracts(unittest.TestCase):
    def test_blocked_compile_workspaces_are_retained_without_path_deletes_or_moves(self):
        text=(ROOT/'blocked'/'blocked_fp32_gemm83.py').read_text()
        self.assertIn('shutil.copyfile(built_obj, obj)',text)
        self.assertNotIn('shutil.rmtree(',text)
        self.assertNotIn('built_obj.replace(',text)
        self.assertIn('unexpected build workspace',text)
    def test_blocked_C0_spans_all_K_and_never_rounds_partials(self):
        text=(ROOT/'blocked'/'blocked_fp32_gemm83.py').read_text()
        self.assertIn('t.matmul(c0, al, bl, mb, kb, nb, init_l1out=(kt == 0))',text)
        self.assertNotIn('t.h_cast(',text)
        self.assertIn('scope=tik.scope_cc',text)
        self.assertIn('2 * (m_pad - mb)',text)
    def test_bias_is_FP32_UB_add_followed_by_one_Half_cast(self):
        text=(ROOT/'bias'/'tik_bias_epilogue83.py').read_text();ast.parse(text)
        self.assertIn('init_l1out=True, bias=None',text)
        self.assertIn('t.tensor_mov(c32, acc_l0c',text)
        self.assertIn('t.vec_add(16,',text)
        self.assertEqual(text.count('t.h_cast(c16, c32, "none")'),1)
        self.assertLess(text.index('t.vec_add(16,'),text.index('t.h_cast(c16, c32, "none")'))
    def test_original_CPU_reference_path_and_BF16_policy_unchanged(self):
        for name in('probe_bias83.py','probe_blocked83.py'):
            text=(ROOT/name).read_text()
            self.assertNotIn('torch.bfloat16',text)
            self.assertIn('cpu_half=torch.nn.functional.linear(',text)
            self.assertIn("'whole_goal_complete':False",text)


if __name__=='__main__':unittest.main()
