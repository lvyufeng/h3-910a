"""Contract tests for the R83 root probe harnesses,without torch or NPU imports."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class RootProbeContracts(unittest.TestCase):
    def test_all_root_modules_compile_without_execution(self):
        for name in('runtime5gm83.py','probe_bias83.py','probe_blocked83.py'):
            text=(ROOT/name).read_text();ast.parse(text);compile(text,str(ROOT/name),'exec')
    def test_current_stream_five_GM_ABI_and_once_verified_library(self):
        text=(ROOT/'runtime5gm83.py').read_text()
        self.assertIn('torch.npu.current_stream().npu_stream',text)
        self.assertIn('if len(buffers)!=5:',text)
        self.assertIn('value.storage_offset()!=0',text)
        self.assertIn('if key in self.cache:return self.cache[key]',text)
        self.assertIn('bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043',text)
        self.assertNotIn('aclrtGetCurrentStream',text)
    def test_bias_teacher_is_actual_CPUHalf_and_only_originalHalfbias(self):
        text=(ROOT/'probe_bias83.py').read_text()
        self.assertIn('cpu_half=torch.nn.functional.linear(ac,wc,bc)',text)
        self.assertIn('bias=bc.float().to(dev)',text)
        self.assertIn("'FP32_diagnostic_not_a_new_teacher'",text)
        self.assertIn('GPU_operands_unchanged',text)
        self.assertIn("'full_encoder_precision_accepted':False",text)
    def test_blocked_GEMM_has_no_Half_partial_and_reports_layout_cost(self):
        text=(ROOT/'probe_blocked83.py').read_text()
        self.assertIn("if meta['dst_dtype']!='float32'",text)
        self.assertIn("'resident_end_to_end_with_A_pack_and_output_unpack_s'",text)
        self.assertIn("'cached_Half_weight_bytes'",text)
        self.assertIn('first.half(),cpu_half',text)
        self.assertIn("'not_whole_encoder_timing':True",text)
        self.assertIn("'smoke':(128,256,128)",text)
    def test_repeat_and_predeclared_wrong_layout_guards_precede_timing(self):
        text=(ROOT/'probe_blocked83.py').read_text()
        self.assertLess(text.index('Blocked GEMM repeat/finite/padding/frozen input guard failed'),text.index('packed_times=[]'))
        self.assertLess(text.index('predeclared wrong-layout sanity guard'),text.index('packed_times=[]'))
        self.assertIn("'precision_acceptance_threshold_chosen':None",text)
        self.assertIn("'predeclared_wrong_layout_guard_RMS':0.01",text)
    def test_write_once_results_no_same_root_replay(self):
        for name in('probe_bias83.py','probe_blocked83.py'):
            text=(ROOT/name).read_text();self.assertIn(".open('x')",text)
            self.assertIn('Refusing to replay or overwrite',text)


if __name__=='__main__':unittest.main()
