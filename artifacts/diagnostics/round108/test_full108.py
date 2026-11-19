"""Full encoder TP8 contract: same CPU Half oracle and actual HCCL/packing/D2H clocks."""
from pathlib import Path
import ast
import unittest
ROOT=Path(__file__).resolve().parent


class Full108Contracts(unittest.TestCase):
    def test_private_originalHalf_TP_priming_and_standalone_H2_no_teacher(self):
        s=(ROOT/'full_encoder108.py').read_text();ast.parse(s)
        for token in('assert size==8','assert all(p.dtype==torch.float16 for p in encoder.parameters())',
                     'vision_tp=VisionTPAdapter107(candidate,group,backend,head2.forward,m=8064)',
                     "'FP32_text_row_adapter_not_installed':True","'full_encoder_precision_accepted':False","'numerical_tolerance':None"):
            self.assertIn(token,s)
        self.assertNotIn('weight.float()',s)
    def test_five_cases_full50plus27_HCCL_scope_not_vision_only(self):
        s=(ROOT/'full_encoder108.py').read_text()
        for name in('native','corrected_vision_native_text','corrected_replica_BC128_native_text','corrected_VisionTP8_native_text','corrected_VisionTP8_precise_text'):
            self.assertIn("('"+name+"',",s)
        self.assertIn('assert len(layers)==50 and len(vislayers)==27',s)
        self.assertIn('with torch.inference_mode():warm=stock()',s)
        self.assertIn('torch.distributed.barrier()\n                torch.npu.synchronize();start=time.perf_counter()',s)
    def test_actual_fullvision_identity_and_native_recovery_before_acceptance(self):
        s=(ROOT/'full_encoder108.py').read_text()
        for token in("==control['merged_SHA256']","==control['raw27_SHA256']","==control['DeepStack_SHA256']",
                     "case['matches_actual_R107_vision_merged_and_all27_boundary_SHA']=True", "assert r['native_recovery']['raw_bits_equal']"):
            self.assertIn(token,s)
        self.assertIn('vision_tp.set_enabled(False);head2.set_enabled(False)',s)
    def test_18objects_serialprebuild_before_any_model(self):
        s=(ROOT/'prebuild108.py').read_text();ast.parse(s)
        self.assertIn("'all18objects_SHA_verified_before_8rank_model_priming':True",s)
        self.assertIn('for meta in text+vision+[erf,flash,head2]+tp:',s)
        self.assertIn('batch=2,m=4032,n=4032,d=72',s)
    def test_genuine_output_capture_before_timed_calls_never_loaded_as_input(self):
        s=(ROOT/'full_encoder108.py').read_text()
        self.assertLess(s.index('torch.save('),s.index('start=time.perf_counter()'))
        self.assertNotIn('torch.load(saved_path',s)
        self.assertIn("'actual_NPU_output_saved_outside_timer'",s)


if __name__=='__main__':unittest.main()
