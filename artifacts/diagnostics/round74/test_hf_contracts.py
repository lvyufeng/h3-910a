"""Source contracts for independent unchanged-BF16 HF multimodal reference."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class HFReferenceContracts(unittest.TestCase):
    def test_true_hf_model_and_real_embedding_vision_inputs_not_shared_features(self):
        text=(ROOT/'hf_multimodal74.py').read_text()
        self.assertIn('model=hf.Qwen3VLModel(config)',text)
        self.assertIn('output=model(input_ids=ids,pixel_values=inputs[\'pixel_values\']',text)
        self.assertIn('image_grid_thw=inputs[\'image_grid_thw\']',text)
        self.assertNotIn('model.embed_tokens=nn.Identity()',text)
        self.assertNotIn('model.language_model.embed_tokens=nn.Identity()',text)
        self.assertNotIn("inputs_embeds=comparison",text)
        self.assertNotIn('fp32_linear',text[text.index('def parameters_sha'):])
    def test_no_hf_linear_or_norm_arithmetic_replacement_and_final_norm_pre_boundary(self):
        tree=ast.parse((ROOT/'hf_multimodal74.py').read_text())
        forbidden=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute)and target.attr=='forward':forbidden.append(ast.unparse(target))
        self.assertEqual(forbidden,[])
        text=(ROOT/'hf_multimodal74.py').read_text()
        self.assertIn('model.language_model.norm.register_forward_pre_hook(norm_pre)',text)
        self.assertIn("metric(comparison['complete_output'],PRENORM[0])",text)
        self.assertIn('output.last_hidden_state',text)
        self.assertNotIn('model.language_model.norm=nn.Identity',text)
    def test_checkpoint_assigns_original_bf16_params_and_rebuilds_only_nonpersistent_rotary(self):
        text=(ROOT/'hf_multimodal74.py').read_text()
        self.assertIn('assert value.dtype==torch.bfloat16',text)
        self.assertIn('model.load_state_dict(state,strict=True,assign=True)',text)
        self.assertIn('assert set(state)==wanted',text)
        self.assertIn('all(not b.is_meta for b in model.buffers())',text)
        self.assertIn('model.visual.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding',text)
        self.assertIn('model.language_model.rotary_emb=hf.Qwen3VLTextRotaryEmbedding',text)
        self.assertNotIn('model.to(torch.bfloat16)',text)
    def test_selected50_layers_and_fullprogress_are_not_partial_reference_pass(self):
        text=(ROOT/'hf_multimodal74.py').read_text()
        self.assertIn('config.text_config.num_hidden_layers=50',text)
        self.assertIn('len(model.language_model.layers)==50',text)
        self.assertIn("len(r['language_layers'])==50",text)
        self.assertIn("r['full_multimodal_reference_completed']=True",text)
        self.assertIn('r[\'completed_language_layers\']=len(r[\'language_layers\'])',text)
    def test_actual_cpu_linears_observed_bf16_and_cpu_reference_scope(self):
        text=(ROOT/'hf_multimodal74.py').read_text()
        self.assertIn('assert args[0].dtype==torch.bfloat16 and module.weight.dtype==torch.bfloat16',text)
        self.assertIn("p.device.type=='cpu'",text)
        self.assertIn('CPU reference wall time is NOT',text)
        self.assertIn("'shared_native_visual_features_used':False",text)
    def test_current_experimental_layers_predeepstack_boundary_is_explicit(self):
        text=(ROOT/'capture_text74.py').read_text()
        self.assertIn('for index in(0,1,2,9,24,49)',text)
        self.assertIn('BEFORE model-loop DeepStack injection',text)
        self.assertIn('output.detach().cpu().contiguous().clone()',text)
        self.assertIn('cast_attention.enabled=True;cast_attention.verify=False',text)


if __name__=='__main__':unittest.main()
