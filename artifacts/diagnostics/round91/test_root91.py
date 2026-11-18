"""Actual full encoder warm and only deterministic geometry cached contracts."""
from pathlib import Path
import ast
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class FullEncoderContracts(unittest.TestCase):
    def test_real_stock_encoder_call_is_inside_warm_timer_not_CPUconditions(self):
        text=(ROOT/'full_encoder91.py').read_text();ast.parse(text)
        self.assertIn('return encoder.encode_ids(payload[\'input_ids\'],pixel_values=payload[\'pixel_values\'],image_grid_thw=payload[\'image_grid_thw\'])',text)
        start=text.index('            for _ in range(5):')
        end=text.index("            case['median_stock_s']",start)
        warm=text[start:end]
        self.assertIn('torch.distributed.barrier()',warm)
        self.assertIn('with torch.inference_mode():warm=stock()',warm)
        self.assertIn('time.perf_counter()-start',warm)
        self.assertNotIn('cpu_on_npu',warm)
        self.assertLess(end,text.index("r['CPU_condition_controls']"))
    def test_corrected_vision_scope_and_native_rows_are_explicit(self):
        text=(ROOT/'full_encoder91.py').read_text()
        self.assertIn('bias=VisionBiasAdapter87(',text)
        self.assertIn('patch=PatchBiasAdapter89(',text)
        self.assertIn('erf=ErfGeluAdapter90(',text)
        self.assertNotIn('row_fp32_adapter',text)
        self.assertIn("'FP32_row_adapter_not_installed':True",text)
        self.assertIn('encoder.vision(payload[\'pixel_values\'].to(device,torch.float16),grid)',text)
    def test_all77_boundary_and_complete_stock_equivalence_gates(self):
        text=(ROOT/'full_encoder91.py').read_text()
        self.assertIn('len(layers)==50 and len(vislayers)==27',text)
        self.assertIn("case['repeat_vs_first']['raw_bits_equal']and case['stock_vs_equivalent']['raw_bits_equal']",text)
        self.assertIn("layer_hashes[name]==layer_hashes['native']",text)
        self.assertIn("layer_hashes[name]==layer_hashes['corrected_vision_native_text']",text)
        self.assertIn("r['native_recovery']['raw_bits_equal']",text)
    def test_original_cpu_boundary_and_no_accuracy_waiver(self):
        text=(ROOT/'full_encoder91.py').read_text()
        self.assertIn("['pre_final_norm']",text)
        self.assertIn("'full_encoder_precision_accepted':False",text)
        self.assertIn("'numerical_tolerance':None",text)
        self.assertIn("if not precise:assert previous.tensor_info(out)['sha256']=='851a626fac934eb95ac9e370d6dd24f9ab936711ecef73d0961ed694fa7dc059'",text)
    def test_previous_executed_sources_kept_immutable(self):
        for p,sha in((ROOT.parent/'round90/erf_gelu_fixed90.py','5a5e85aaad5697a7a890ecb0b5b724bc3acd93d819978ab55d5eb68038cd0552'),
                      (ROOT.parent/'round89/patch_bias89.py','2eb5ff1b6ac741d3bb27e523524a917aeebda791525ac23dffb20b906d45e7f2'),
                      (ROOT.parent/'round87/best_text87.py','be449afe470a593d5381bcf8c08a1c0c3654010e1790351b59d8dae88d909409')):
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),sha)


if __name__=='__main__':unittest.main()
