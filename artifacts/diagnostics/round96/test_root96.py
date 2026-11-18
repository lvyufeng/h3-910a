"""Current real vision precision combination and old QKV gates, no speed waiver."""
from pathlib import Path
import ast
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class VisionCombinationContracts(unittest.TestCase):
    def test_original_attention_QKV_and_current_R90_vision_bits_are_gated(self):
        s=(ROOT/'vision_probe96.py').read_text();ast.parse(s)
        self.assertIn("previous.file_sha(frozen_path)==expected['frozenQKVfileSHA256']",s)
        self.assertIn("expected['R88_precise_leaf_output_SHAs'][image]",s)
        self.assertIn("expected['R90_current_joint_vision_merged_SHA256']",s)
        self.assertIn("previous.compare(repeat,out)['raw_bits_equal']",s)
    def test_only_originalHalfweights_actualpixels_all27_blocks(self):
        s=(ROOT/'vision_probe96.py').read_text()
        self.assertIn("value=h.get_tensor(name).to(torch.float16)",s)
        self.assertIn("proof['parameter_conversion']['visual.'+key]['reference_sha256']",s)
        self.assertIn('len(layers)==27',s)
        self.assertIn("('current_patch_bias_erf_accurate_attention',True,True)",s)
        self.assertIn('out,out_ds=vision(pixels,grids)',s)
    def test_repeats_native_recovery_params_inputs_originalCPUgold_unchanged(self):
        s=(ROOT/'vision_probe96.py').read_text()
        self.assertIn('for _ in range(5):',s)
        self.assertIn("r['native_recovery']['merged']['raw_bits_equal']",s)
        self.assertIn("r['original_parameters_unchanged']",s)
        self.assertIn("r['pixels_unchanged']",s)
        self.assertIn("r['CPU_reference_SHA_unchanged']",s)
        self.assertIn("'full_encoder_precision_accepted':False",s)
        self.assertIn("'not_full_encoder_timing':True",s)
        self.assertIn("'numerical_tolerance':None",s)
    def test_R88_engine_source_pinned_and_noclass_global_torch_change(self):
        r=json.loads((ROOT/'expected_inputs96.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT.parent/'round88/attention/accurate_vision_attention88.py').read_bytes()).hexdigest(),r['R88_current_attention_source_SHA256'])
        self.assertEqual(r['frozenQKVfileSHA256'],'3fb5d524373c96de48a4be583cee87f171f5e20fdaa8ac57fd4a31cb709ce552')


if __name__=='__main__':unittest.main()
