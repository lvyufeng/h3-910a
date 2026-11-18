"""Original CPU Half entry and unwaived nonlinear A/B source contracts."""
from pathlib import Path
import ast
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class VisionEntryContracts(unittest.TestCase):
    def test_CPU_entry_uses_original_HF_half_parameters_and_original_block0_gold(self):
        text=(ROOT/'original_hf_entry89.py').read_text();ast.parse(text)
        self.assertIn('cpu=hf.Qwen3VLVisionModel(cfg)',text)
        self.assertIn('cpu.load_state_dict(state,strict=True,assign=True)',text)
        self.assertIn('cpu.patch_embed(payload[\'pixel_values\'])',text)
        self.assertIn('cpu.fast_pos_embed_interpolate',text)
        self.assertIn('sdpa_kernel(SDPBackend.MATH)',text)
        self.assertIn("proof['vision_layers'][0]['output']",text)
        self.assertIn("observed['sha256']==expected['sha256']",text)
        self.assertIn("torch.set_num_threads(proof['CPU_num_threads'])",text)
    def test_actual_patch_CPU_linear_uses_Half_not_float_weight(self):
        text=(ROOT/'vision_probe89.py').read_text();ast.parse(text)
        self.assertIn('flat_weight.dtype==patch_bias.dtype==torch.float16',text)
        self.assertIn('torch.nn.functional.linear(payload[\'pixel_values\'],flat_weight,patch_bias)',text)
        self.assertNotIn('patch_weight.float()',text)
        self.assertIn("'CPU_Half_Linear_vs_original_CPU_Half_Conv3d'",text)
    def test_actual_leaves_precede_optional_adapters_and_all_cases_preserved(self):
        text=(ROOT/'vision_probe89.py').read_text()
        self.assertLess(text.index("r['actual_nonlinear_leaf']=run_controls("),text.index('nonlinear=VisionNonlinearAdapter89('))
        self.assertIn("('CPU_entry_only',True,False,False,False)",text)
        self.assertIn("('CPU_entry_bias',True,True,False,False)",text)
        self.assertIn("('Float_norm_only',False,False,True,False)",text)
        self.assertIn("('Float_GELU_only',False,False,False,True)",text)
        self.assertIn("('CPU_entry_norm_GELU_bias',True,True,True,True)",text)
        self.assertIn("'numerical_tolerance':None",text)
    def test_rebase_is_a_diagnostic_not_a_full_encoder_speed_claim(self):
        text=(ROOT/'vision_probe89.py').read_text()
        self.assertIn("'CPU_rebase_not_full_encoder_timing':True",text)
        self.assertIn('entry_npu=cpu[\'entry\'].to(\'npu:0\')',text)
        self.assertIn("previous.compare(entry_npu,cpu['entry'])['raw_bits_equal']",text)
        self.assertIn("'R88_slow_attention_not_installed':True",text)
        self.assertNotIn('accurate_vision_attention88',text)
        self.assertIn("r['native_recovery']['merged']['raw_bits_equal']",text)
    def test_fixed_leaf_call_is_inside_inference_mode(self):
        text=(ROOT/'vision_probe_fixed89.py').read_text();tree=ast.parse(text)
        found=False
        for node in ast.walk(tree):
            if isinstance(node,ast.With)and any('torch.inference_mode()'==ast.unparse(item.context_expr)for item in node.items):
                found=found or any(isinstance(c,ast.Call)and isinstance(c.func,ast.Name)and c.func.id=='run_controls'for body in node.body for c in ast.walk(body))
        self.assertTrue(found)
        self.assertIn('/tmp/h3_round89_vision02',text)
        self.assertEqual(hashlib.sha256((ROOT/'vision_probe89.py').read_bytes()).hexdigest(),'13eaf186c7a66fbc934745668d33bb9ecdb00d69efc54ecbd58e18fc1c89a2a4')
    def test_patch_kernel_has_the_original_flatten_and_one_voxel_contract(self):
        text=(ROOT/'patch_bias89.py').read_text();ast.parse(text)
        self.assertIn('p.kernel_size)==tuple(p.stride)==(2,16,16)',text)
        self.assertIn('kernel.build_kernel(m=8064,k=1536,n=1152',text)
        self.assertIn('out.permute(1,0,2).reshape(8064,1152)',text)
        self.assertIn('[packed_a,self.packed_w,self.packed_bias,self.aux,out]',text)
        self.assertIn('dtype=torch.float16',text)
        self.assertNotIn('p.weight.float()',text)
        self.assertIn('self.module.forward',text)
    def test_immutable_R87_bias_and_R78_HF_baseline_pins(self):
        self.assertEqual(hashlib.sha256((ROOT.parent/'round87/vision_adapter87.py').read_bytes()).hexdigest(),'8393f6b9c379e0df8cc3c1393ff2b839dab4c535fddd4be14782cec615c8db19')
        self.assertIn('fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe',(ROOT/'original_hf_entry89.py').read_text())


if __name__=='__main__':unittest.main()
