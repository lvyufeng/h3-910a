"""Protect independent CPU FP16 reference and actual operator support evidence."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class CPUFP16ReferenceContracts(unittest.TestCase):
    def test_actual_CPU_half_operator_coverage_and_finite_repeat_proofs(self):
        proof=json.loads((ROOT/'preflight.json').read_text())
        self.assertTrue(proof['completed']);self.assertTrue(proof['all_required_preflight_operators_supported'])
        self.assertEqual(proof['versions']['torch'],'2.10.0+cpu');self.assertEqual(len(proof['operators']),9)
        self.assertEqual(hashlib.sha256((ROOT/'cpu_fp16_preflight78.py').read_bytes()).hexdigest(),proof['source_sha256'])
        for row in proof['operators'].values():
            self.assertTrue(row['supported']);self.assertTrue(row['repeat_bits_equal']);self.assertTrue(row['output']['finite'])
            self.assertEqual(row['output']['dtype'],'torch.float16')
        self.assertFalse(proof['FP32_linear_teacher_adapter_used']);self.assertFalse(proof['NPU_precision_accepted'])
    def test_pinned_math_backend_support(self):
        proof=json.loads((ROOT/'math_preflight.json').read_text())
        self.assertTrue(proof['supported']);self.assertTrue(proof['completed']);self.assertEqual(proof['backend'],'SDPBackend.MATH')
        self.assertEqual(hashlib.sha256((ROOT/'cpu_math_preflight78.py').read_bytes()).hexdigest(),proof['source_sha256'])
        self.assertEqual({row['head_dim']for row in proof['cases']},{72,128})
        self.assertTrue(all(row['finite']and row['repeat_bits_equal']and row['dtype']=='torch.float16'for row in proof['cases']))
    def test_CPU_reference_no_optimized_teacher_or_old_BF16_outputs(self):
        text=(ROOT/'hf_cpu_fp16_reference78.py').read_text();tree=ast.parse(text)
        imports=[node.module for node in ast.walk(tree)if isinstance(node,ast.ImportFrom)]
        self.assertNotIn('vllm_omni',imports);self.assertNotIn('cast_attention73',imports)
        self.assertNotIn('fresh_encoder_reference.pt',text);self.assertNotIn('hf_selected50.pt',text)
        self.assertIn("'old_BF16_outputs_reused_or_cast_as_gold':False",text)
        self.assertIn("'BF16_output_agreement_required':False",text)
        self.assertIn("'FP32_linear_adapter_used':False",text)
        self.assertIn('converted=original.to(torch.float16)',text)
        self.assertIn('config.text_config.num_hidden_layers=50',text)
        self.assertIn('source_sha.hexdigest()',text)
    def test_raw_selected50_pre_norm_boundary_and_dtype_hooks(self):
        text=(ROOT/'hf_cpu_fp16_reference78.py').read_text()
        self.assertIn('model.language_model.norm.register_forward_pre_hook(norm_pre)',text)
        self.assertIn("'pre_final_norm':PRENORM",text)
        self.assertIn("LINEAR_INPUT_DTYPES=={'torch.float16'}",text);self.assertIn("LINEAR_OUTPUT_DTYPES=={'torch.float16'}",text)
        self.assertIn("'comparison_SDPA_backend'",text);self.assertIn('sdpa_kernel(backends=[SDPBackend.MATH])',text)
        self.assertIn('model.visual.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding',text)
        self.assertIn("hf.Qwen3VLTextRotaryEmbedding(config.text_config,device='cpu')",text)
        self.assertNotIn('model.to(torch.float16)',text)
    def test_multimodal_token_type_ids_and_full_repeat_are_required(self):
        text=(ROOT/'hf_cpu_fp16_reference78.py').read_text()
        self.assertIn('mm[ids==config.image_token_id]=1',text);self.assertIn('mm[ids==config.video_token_id]=2',text)
        self.assertEqual(text.count('mm_token_type_ids=mm'),2)
        self.assertIn("r['math_preflight_file_sha256']=file_sha(ROOT/'math_preflight.json')",text)
        self.assertIn("r['pre_final_norm_matches_layer49_identity']",text)
        self.assertIn("r['repeat_pre_final_norm_bits_equal']",text);self.assertIn("r['repeat_post_final_norm_bits_equal']",text)
    def test_official_dense_checkpoint_manifest_and_audit_are_bounded(self):
        proof=json.loads((ROOT/'official_text_manifest_receipt78.json').read_text())
        self.assertEqual(len(proof['lfs_shards']),14);self.assertFalse(proof['weight_payload_downloaded'])
        self.assertEqual(hashlib.sha256((ROOT/'official_text_manifest78.json').read_bytes()).hexdigest(),proof['raw_sha256'])
        self.assertEqual(proof['revision'],'0919fb684102ef115126612f3fbb84d1027e7317')
        audit=json.loads((ROOT/'npu_fp16_audit78.json').read_text());self.assertFalse(audit['NPU_candidate_executed'])
        source=ROOT.parent/'round76'/'source'/'encoder.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),audit['source_sha256'])
    def test_no_false_inference_from_operator_preflight_to_full_accuracy(self):
        proof=json.loads((ROOT/'preflight.json').read_text())
        self.assertFalse(proof['cpu_reference_generated']);self.assertFalse(proof['whole_goal_complete'])
        self.assertTrue(proof['operator_coverage_does_not_prove_whole_model_accuracy'])


if __name__=='__main__':unittest.main()
