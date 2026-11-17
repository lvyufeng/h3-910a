"""Guard independent FP16 vision probe scope before real-device execution."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class VisionFP16ProbeContracts(unittest.TestCase):
    def test_probe_is_separate_from_production_and_BF16_teacher(self):
        text=(ROOT/'vision_probe79.py').read_text();ast.parse(text)
        self.assertIn("'production_changed':False",text);self.assertIn("'full_encoder_precision_accepted':False",text)
        self.assertIn("oracle_path=REFERENCE_ROOT/'FP16_vision.pt'",text)
        self.assertNotIn('hf_vision.pt',text);self.assertNotIn('native_bf16_vector58',text);self.assertNotIn('cast_attention73',text)
        self.assertNotIn('F.scaled_dot_product_attention =',text)
        self.assertIn("value=value.to(torch.float16)",text)
        self.assertIn("expected_source_tensors['visual.'+name]['reference_sha256']",text)
    def test_true_FP16_input_params_and_reference_outputs_required(self):
        text=(ROOT/'vision_probe79.py').read_text()
        self.assertIn("assert pixels.dtype==torch.float16",text)
        self.assertIn("p.dtype==torch.float16",text)
        self.assertIn("a.dtype==b.dtype==torch.float16",text)
        self.assertIn("assert file_sha(source)==manifest['candidate_sha256']",text)
    def test_complete_vision_and_repeat_boundaries_do_not_claim_full_encoder(self):
        text=(ROOT/'vision_probe79.py').read_text()
        self.assertIn("len(deepstack)==len(reference['deepstack'])==3",text)
        self.assertIn("len(r['layers'])==27",text)
        self.assertIn('for index in range(3)',text)
        self.assertIn("r['full_encoder_precision_accepted']=False",text)
        self.assertIn("'whole_goal_complete':False",text)
    def test_reference_stage_hash_matches_download_when_available(self):
        receipt=ROOT/'reference'/'vision_reference_receipt79.json'
        if not receipt.exists():self.skipTest('Stage reference download still in progress')
        proof=json.loads(receipt.read_text());source=ROOT/'reference'/'FP16_vision.pt'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),proof['file_sha256'])
        self.assertEqual(proof['file_sha256'],'c01bbe180f454fe44b182bd9307491ef0f4c342b6c2ff6be08d1e3453d8c312a')
        self.assertFalse(proof['NPU_precision_accepted']);self.assertFalse(proof['whole_goal_complete'])


if __name__=='__main__':unittest.main()
