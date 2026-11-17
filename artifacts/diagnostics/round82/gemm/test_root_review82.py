"""Root-review guards for actual-profile query,stream ABI and original Half CPU teacher."""
import ast
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class ReviewedGemmContracts(unittest.TestCase):
    def test_profile_is_selected_before_intrinsic_capability(self):
        source=(ROOT/'gemm_fp16_fp32out82.py').read_text()
        self.assertLess(source.index('compile_soc = set_compile_soc()'),source.index('l0c_fp32 = platform.intrinsic_check_support'))
        self.assertIn("int(report['core_num']) != 30",source)
        self.assertNotIn("int(report['core_num']) != 32",source)
    def test_compiler_core_profile_not_memory_size_or_A2_assumption(self):
        profiles=json.loads((ROOT/'profile_compare82.json').read_text())
        counts={item['requested_compile_soc']:item['CORE_NUM']for item in profiles['rows']}
        self.assertEqual(counts,{'Ascend910A':32,'Ascend910B':30})
        self.assertTrue(all(item['Intrinsic_mmad_f162f32']for item in profiles['rows']))
    def test_launcher_uses_known_torch_stream_not_a_guessed_C_ABI(self):
        source=(ROOT/'run_gemm_probe82.py').read_text()
        self.assertIn('ctypes.c_void_p(torch.npu.current_stream().npu_stream)',source)
        self.assertNotIn('lib.gemm82_current_stream()',source)
        self.assertNotIn('aclrtGetCurrentStream();',(ROOT/'gemm_launcher82.c').read_text())
        self.assertIn('-lascendcl',source);self.assertIn('-lruntime',source)
    def test_original_Half_bias_and_CPU_Half_operator_are_directly_compared(self):
        source=(ROOT/'run_gemm_probe82.py').read_text()
        self.assertIn('bias_np = bias_half_np.astype(np.float32)',source)
        self.assertIn('torch.nn.functional.linear(ac,bc,bh)',source)
        self.assertIn("'kernel_cast_to_half_vs_CPU_FP16_linear'",source)
        self.assertIn("'FP32_numpy_golden_is_only_accumulator_diagnostic'",source)
    def test_actual_launch_not_inferred_from_source_or_build(self):
        source=(ROOT/'run_gemm_probe82.py').read_text()
        self.assertIn("receipt['kernel_launch_executed'] = False",source)
        self.assertIn("receipt['kernel_launch_executed'] = True",source)
        self.assertIn('kernel repeat or finite control failed',source)
        ast.parse(source)


if __name__=='__main__':unittest.main()
