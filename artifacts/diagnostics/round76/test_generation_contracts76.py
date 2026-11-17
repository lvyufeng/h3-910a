"""Actual encoder/handoff and API guard contracts for generated-video verification."""
import ast
import hashlib
import json
from pathlib import Path
import unittest
from summarize_encoder76 import summarize

ROOT=Path(__file__).resolve().parent


class GenerationContracts(unittest.TestCase):
    def test_fresh_optimized_encoder_is_not_live_request_or_native_bf16_claim(self):
        r=summarize();self.assertEqual(r['complete_output_identity_checks'],32)
        self.assertTrue(r['all_output_and_intermediate_identities_equal'])
        self.assertFalse(r['live_encoder_generation_end_to_end_cost_proven']);self.assertFalse(r['nativeBF16compute_used'])
        self.assertEqual(r['conditioning_BF16_sha256'],'67ef196e33bcaa495420da1ebc07d1ff6dcaf89ddbaf75a350940ed8ae210eee')
    def test_invalid_CLI_alias_failure_is_retained(self):
        r=json.loads((ROOT/'failed01'/'generation_result.json').read_text())
        self.assertEqual(r['failure']['type'],'ValueError');self.assertIn("'ring', 'usp'",r['failure']['message']);self.assertEqual(r['cases'],[])
        self.assertEqual(hashlib.sha256((ROOT/'generate_optimized76.py').read_bytes()).hexdigest(),r['source_sha256'])
    def test_fixed_offline_constructor_uses_canonical_parallel_fields(self):
        text=(ROOT/'generate_optimized_fixed76.py').read_text()
        self.assertIn("'ulysses_degree': 8",text);self.assertIn("'ring_degree': 1",text)
        self.assertNotIn("'usp': 8",text);self.assertNotIn("'ring': 1",text)
        self.assertIn("'text_encoder': False",text);self.assertIn("'vae_encoder': True",text)
    def test_payload_validation_happens_before_expensive_engine_creation(self):
        text=(ROOT/'generate_optimized_fixed76.py').read_text()
        self.assertLess(text.index('validated_conditions[label] = MiniMaxH3TextConditioning.from_payload'),text.index('omni = Omni('))
        self.assertIn('tags.dtype == torch.int64',text);self.assertIn('(tags == 0) | (tags == 1)',text)
    def test_offline_script_no_worker_import_runs_main_or_claims_receipt(self):
        tree=ast.parse((ROOT/'generate_optimized_fixed76.py').read_text())
        for node in tree.body:
            if isinstance(node,ast.Assign):
                targets=[target.id for target in node.targets if isinstance(target,ast.Name)]
                self.assertNotIn('writer',targets);self.assertNotIn('omni',targets)
            self.assertNotIsInstance(node,ast.Try)
        self.assertIsInstance(tree.body[-1],ast.If);self.assertIn("__name__ == '__main__'",ast.unparse(tree.body[-1].test))
    def test_temporary_VAE_candidate_does_not_modify_other_production_nodes(self):
        old=ast.parse((ROOT/'source'/'vae.py').read_text());new=ast.parse((ROOT/'candidate_vae76.py').read_text())
        oldcls=next(c for c in old.body if isinstance(c,ast.ClassDef)and c.name=='MiniMaxH3VideoVAE')
        newcls=next(c for c in new.body if isinstance(c,ast.ClassDef)and c.name=='MiniMaxH3VideoVAE')
        init=next(n for n in newcls.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
        matches=[n for n in init.body if isinstance(n,ast.If)and 'VLLM_OMNI_910A_VAE_FINITE_SELECT'in ast.unparse(n.test)]
        self.assertEqual(len(matches),1);init.body.remove(matches[0])
        self.assertEqual(ast.dump(old),ast.dump(new))
    def test_5second_FL2VA49steps_exact_input_validation_and_no_LoRA(self):
        text=(ROOT/'generate_optimized_fixed76.py').read_text()
        self.assertIn('num_inference_steps=49',text);self.assertIn('seed=1234',text);self.assertIn("'duration_seconds': 5",text)
        self.assertIn('prepared.media.num_frames == 124',text)
        self.assertIn("torch.equal(ids, original_inputs['input_ids'])",text)
        self.assertNotIn('lora_path=',text);self.assertNotIn('lora_request=',text)


if __name__=='__main__':unittest.main()
