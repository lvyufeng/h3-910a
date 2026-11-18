"""Real vision72/key4032 CPUHalf oracle and separate bias/PV A/B contracts."""
from pathlib import Path
import ast
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class VisionAttentionDriverContracts(unittest.TestCase):
    def test_teacher_is_same_real_Half_QKV_unmasked_CPU_MATH(self):
        text=(ROOT/'vision_probe88.py').read_text();ast.parse(text)
        self.assertIn('q.shape==k.shape==v.shape==(1,16,4032,72)',text)
        self.assertIn('q.dtype==k.dtype==v.dtype==torch.float16',text)
        self.assertIn('sdpa_kernel(SDPBackend.MATH)',text)
        self.assertIn('is_causal=False',text);self.assertNotIn('is_causal=True',text)
        self.assertIn("before_leaf={name:previous.tensor_info(x)['sha256']",text)
        self.assertIn('assert leaf_unchanged',text)
    def test_real_capture_precedes_engine_install_and_separate_full_cases(self):
        text=(ROOT/'vision_probe88.py').read_text()
        self.assertLess(text.index('candidate._vision_sdpa_bf16_compatible=capture_sdpa'),text.index('engine=AccurateVisionAttention88('))
        for mode in("('native',False,False)","('bias_only',True,False)","('accurate_PV_only',False,True)","('accurate_PV_and_bias',True,True)"):
            self.assertIn(mode,text)
        self.assertIn("r['native_recovery']['merged']['raw_bits_equal']",text)
    def test_padded_columns_and_actual_prefix_are_checked(self):
        text=(ROOT/'vision_probe88.py').read_text()
        self.assertIn('engine.padding_selfcheck(qn,kn,vn)',text)
        self.assertIn("'q_prefix_exact','q_width_tail_zero','k_key_tail_zero'",text)
        self.assertIn('scale=72**-.5',text)
        self.assertIn('torch.matmul(cp.to(\'npu:0\'),vn.float())',text)
        self.assertIn("'not_full_encoder_timing':True",text)
        self.assertIn("'full_encoder_precision_accepted':False",text)
    def test_executed_R87_sources_remain_unchanged(self):
        for path,pin in((ROOT.parent/'round87/vision_adapter87.py','8393f6b9c379e0df8cc3c1393ff2b839dab4c535fddd4be14782cec615c8db19'),
                         (ROOT.parent/'round87/vision/vision_bias87.py','f8655b5fa35aecec0753d3e657f16fd2e235f402a57e441eb8959f4c057f0379')):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pin)


if __name__=='__main__':unittest.main()
