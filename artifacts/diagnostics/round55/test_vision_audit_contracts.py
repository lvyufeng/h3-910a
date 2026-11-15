#!/usr/bin/env python3
"""Structural checks for diagnostic-only vision comparisons and matched inputs."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent

class VisionAuditContracts(unittest.TestCase):
 def test_source_extracts_vision_only(self):
  text=(ROOT/'vision_cpu_reference.py').read_text()
  self.assertIn("n.name.startswith('MiniMaxH3Qwen3VLVision')",text)
  self.assertNotIn('MiniMaxH3Qwen3VLEncoder(',text)
 def test_teacher_uses_same_block_weights_and_input(self):
  text=(ROOT/'vision_cpu_reference.py').read_text()
  self.assertIn('teacher.load_state_dict(module.state_dict(),strict=True)',text)
  self.assertIn("teacher(payload['hidden'],cu_seqlens=payload['cu_seqlens'],position_embeddings=payload['position_embeddings'])",text)
 def test_npu_suboperators_use_teacher_inputs(self):
  text=(ROOT/'vision_npu_matched_reference.py').read_text()
  self.assertIn("native.attn.qkv(norm1.to('npu'))",text)
  self.assertIn("native.mlp.linear_fc1(norm2.to('npu'))",text)
  self.assertIn("native.mlp.act_fn(fc1.to('npu'))",text)
  self.assertIn("native.mlp.linear_fc2(gelu.to('npu'))",text)
 def test_reference_reports_partial_progress(self):
  for name in ('vision_cpu_reference.py','vision_npu_matched_reference.py'):
   text=(ROOT/name).read_text()
   self.assertIn('OUT.write_text(json.dumps(report,indent=2)',text)
   self.assertIn("report['failure']",text)
 def test_cpu_teacher_uses_unchanged_real_capture(self):
  text=(ROOT/'vision_cpu_reference.py').read_text()
  self.assertIn('/tmp/h3_round54_capture01',text)
  self.assertIn("inputs['pixel_values']",text)
  self.assertIn("inputs['image_grid_thw']",text)

 def test_deployed_candidate_changes_only_vision_linears_and_interpolation(self):
  base=ast.parse((ROOT/'source/encoder.py').read_text())
  new=ast.parse((ROOT/'candidate_encoder.py').read_text())
  table=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
  b=table(base);n=table(new)
  self.assertEqual(set(n)-set(b),{'_NpuBF16CompatibleLinear','_vision_fp32_enabled'})
  for name,original in b.items():
   updated=n[name]
   if name=='MiniMaxH3Qwen3VLVisionModel':
    bm={x.name:x for x in original.body if isinstance(x,ast.FunctionDef)}
    nm={x.name:x for x in updated.body if isinstance(x,ast.FunctionDef)}
    for method in bm:
     if method!='fast_pos_embed_interpolate':
      self.assertEqual(ast.dump(bm[method]),ast.dump(nm[method]),method)
   else:
    class RestoreLinear(ast.NodeTransformer):
     def visit_Name(self,node):
      if node.id=='_NpuBF16CompatibleLinear':
       return ast.copy_location(ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Linear',ctx=ast.Load()),node)
      return node
    normalized=RestoreLinear().visit(updated)
    self.assertEqual(ast.dump(original),ast.dump(normalized),name)
 def test_optin_default_off_and_narrow_dtype_guards(self):
  tree=ast.parse((ROOT/'candidate_encoder.py').read_text())
  names={n.name:n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
  self.assertIn("get('VLLM_OMNI_910A_VISION_FP32', '0')",ast.unparse(names['_vision_fp32_enabled']))
  text=ast.unparse(names['_NpuBF16CompatibleLinear'])
  self.assertIn('input.dtype == torch.bfloat16',text)
  self.assertIn('self.weight.dtype == torch.bfloat16',text)
  self.assertIn('return super().forward(input)',text)
  self.assertNotIn('allow_hf32',text)
 def test_ab_reproducer_uses_frozen_unconditional_candidate(self):
  text=(ROOT/'full_tower_ab.py').read_text()
  self.assertIn("ROOT/'candidate_encoder_unconditional_fp32.py'",text)
 def test_quality_request_matches_baseline_parameters(self):
  text=(ROOT/'e2e_vision_precision_probe.py').read_text()
  self.assertIn('"num_inference_steps": "49"',text)
  self.assertIn('"seconds": "5"',text)
  self.assertIn('"VLLM_OMNI_910A_VISION_FP32": "1"',text)
  self.assertIn('env.pop("VLLM_OMNI_910A_AUDIT_DIR", None)',text)

if __name__=='__main__':unittest.main(verbosity=2)
