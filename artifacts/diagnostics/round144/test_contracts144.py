#!/usr/bin/env python3
"""Contracts for R143 staging timing and R144 shape verification (measurement only)."""
import ast,json,unittest
from pathlib import Path
R143=Path(__file__).resolve().parent.parent/'round143'
R144=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in list(R143.glob('*.py'))+list(R144.glob('*.py')):
            ast.parse(p.read_text(),filename=str(p))
    def test_no_sync_in_forward_probe(self):
        s=(R143/'staging_probe143.py').read_text()
        forward=s.split('def collect',1)[0]
        self.assertNotIn('.synchronize(',forward,'no device sync may run during the forward')
        self.assertIn('copy_stream',s)
        # one sync is allowed, and only when the alignment is computed after the step
        self.assertEqual(s.count('.synchronize('),1)
    def test_correction_is_recorded(self):
        b=json.loads((R143/'staging_and_shapes143.json').read_text())
        self.assertTrue(b['r142_correction']['what_was_wrong'])
        self.assertEqual(len(b['r142_correction']['what_was_wrong']),4)
        self.assertIn('uniformly',b['r142_correction']['why_it_matters'])
    def test_staging_verdict_is_not_a_stall(self):
        b=json.loads((R143/'staging_and_shapes143.json').read_text())
        self.assertGreater(b['staging']['slack_to_next_block_ms_median'],15)
        self.assertGreater(b['staging']['slack_to_next_block_ms_min'],10)
        self.assertIn('just-in-time',b['staging']['verdict'])
    def test_true_shapes(self):
        b=json.loads((R144/'shapes144.json').read_text())
        self.assertTrue(b['measurement_only'] and b['no_math_change'])
        self.assertEqual(b['ranks_completed'],8)
        sh=b['shapes']
        self.assertEqual(sh['block0.attn.qkv_proj']['weight_shape'],[10752,5376])
        self.assertEqual(sh['block0.mlp.fc1']['weight_shape'],[14336,5376])
        self.assertEqual(sh['block0.attn.out_proj']['weight_shape'],[5376,3584])
        self.assertEqual(sh['block0.mlp.fc2']['weight_shape'],[5376,7168])
        self.assertEqual(sh['block0.mlp.fc2']['input_dtype'],'torch.float32')
        self.assertEqual(b['fa'][0]['q'],[1,7,41792,128])
        self.assertEqual(b['fa'][0]['n_pad'],23)

if __name__=='__main__':unittest.main()
