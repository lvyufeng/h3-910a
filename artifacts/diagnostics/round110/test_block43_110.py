"""One true block43 precision isolation: frozen CPU raw42/43, not full-encoder timing."""
from pathlib import Path
import ast
import unittest
ROOT=Path(__file__).resolve().parent


class Block43Controls110(unittest.TestCase):
    def test_real_originalHalf_weightbytes_gate_before_NPU_mapper_and_No_FloatShadow(self):
        s=(ROOT/'block43_verified_weights110.py').read_text();ast.parse(s)
        self.assertIn("converted_sha==reference['parameter_conversion'][key]['reference_sha256']",s)
        self.assertIn("len(r['originalCPUHalf_full_layer43_weight_SHAs_verified'])==11",s)
        self.assertIn("value.to(torch.float16)",s)
        self.assertNotIn('value.float()',s)

    def test_exact_savedCPU_raw42_input_and_originalR78_raw43_identity(self):
        s=(ROOT/'block43_probe110.py').read_text();ast.parse(s)
        for token in("CPU_CAPTURE/'CPU_raw42_109.pt'","CPU_CAPTURE/'CPU_raw43_109.pt'",
                     "previous.tensor_info(cpu_input)['sha256']==truths[42]","previous.tensor_info(cpu_gold)['sha256']==truths[43]",
                     'cpu_input.dtype==cpu_gold.dtype==torch.float16'):
            self.assertIn(token,s)
        self.assertIn('assert previous.compare(inp,cpu_input)[\'raw_bits_equal\']',s)
        self.assertIn("capture_receipt['baseline_passes']['all_25_R78_boundaries_matched']",s)
        self.assertIn("capture_receipt['captured_pair_replay']['comparison']['raw_bits_equal']",s)
        self.assertIn("capture_receipt['captured_tensors'][str(i)]['file_sha256']",s)
    def test_only_original_Half_layer43_native_and_precise_no_late_norm_or_DeepStack(self):
        s=(ROOT/'block43_probe110.py').read_text()
        self.assertIn("layers=torch.nn.ModuleDict({'43':",s)
        self.assertIn('assert len(names)==11',s)
        self.assertIn('assert all(p.dtype==torch.float16 for p in encoder.parameters())',s)
        self.assertNotIn('weight.float()',s);self.assertNotIn('final_norm(',s);self.assertNotIn('deepstack_visual_embeds',s)
        self.assertIn("'not_full_encoder_timing':True",s)
    def test_repeats_native_recovery_and_actual_mlp43_inputs_preserve_reference(self):
        s=(ROOT/'block43_probe110.py').read_text()
        self.assertIn("assert previous.compare(repeated,out)['raw_bits_equal']",s)
        self.assertIn("assert r['native_recovery']['raw_bits_equal']",s)
        self.assertIn("run_controls(encoder.text_model.layers['43'],inp,capture['mlp_input']",s)
        self.assertIn("'numerical_tolerance':None",s)
        self.assertNotIn('time.perf_counter(',s)


if __name__=='__main__':unittest.main()
