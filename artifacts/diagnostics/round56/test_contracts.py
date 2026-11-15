#!/usr/bin/env python3
"""Host structural checks for diagnostic captures and independent CPU ablations."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent
class Contracts(unittest.TestCase):
 def test_capture_only_adds_guard_to_encode(self):
  a=ast.parse((ROOT/'source/encoder.py').read_text());b=ast.parse((ROOT/'candidate_capture_encoder.py').read_text())
  def method(tree):
   return next(m for c in tree.body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_encode')
  original=method(a);new=method(b);self.assertIsInstance(new.body[0],ast.Import);self.assertIsInstance(new.body[1],ast.If)
  new.body=new.body[2:];self.assertEqual(ast.dump(a),ast.dump(b))
 def test_all_forward_hooks_return_none(self):
  tree=ast.parse((ROOT/'h3_text_audit56.py').read_text())
  hooks=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('capture','text_pre','last_pre')]
  self.assertEqual(len(hooks),3)
  for hook in hooks:
   returns=[n for n in ast.walk(hook) if isinstance(n,ast.Return)]
   self.assertTrue(returns)
   self.assertTrue(all(isinstance(n.value,ast.Constant) and n.value.value is None for n in returns))
 def test_one_step_is_diagnostic_not_quality_benchmark(self):
  text=(ROOT/'e2e_capture_probe.py').read_text()
  self.assertIn('"num_inference_steps": "1"',text)
  self.assertIn('"diagnostic_only": True',text)
  self.assertIn('"VLLM_OMNI_910A_VISION_FP32": "0"',text)
 def test_cpu_ablation_separates_norm_and_rope(self):
  text=(ROOT/'text_norm_rope_ab.py').read_text()
  self.assertIn("mode in ('norm-native','both-native')",text)
  self.assertIn("mode in ('rope-native','both-native')",text)
  self.assertIn("CAP/'text_inputs.pt'",text)
  self.assertIn("if index==49:",text)
 def test_candidate_limits_changes_to_text_math_and_vision_amp_guard(self):
  base=ast.parse((ROOT/'source/encoder.py').read_text());new=ast.parse((ROOT/'candidate_text_math_encoder.py').read_text())
  table=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
  a=table(base);b=table(new);self.assertEqual(set(b)-set(a),{'_text_hf_math_enabled','_text_hf_embedding_all_reduce'})
  allowed={'MiniMaxH3Qwen3VLRMSNorm','MiniMaxH3Qwen3VLTextRotaryEmbedding','_NpuBF16CompatibleLinear','MiniMaxH3Qwen3VLVocabParallelEmbedding'}
  for name in a:
   if name not in allowed:self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)
  am={n.name:n for n in a['MiniMaxH3Qwen3VLRMSNorm'].body if isinstance(n,ast.FunctionDef)}
  bm={n.name:n for n in b['MiniMaxH3Qwen3VLRMSNorm'].body if isinstance(n,ast.FunctionDef)}
  self.assertEqual(ast.dump(am['__init__']),ast.dump(bm['__init__']))
  self.assertEqual(set(bm)-set(am),{'forward_npu'})
 def test_math_defaults_off_and_protects_fp32(self):
  text=(ROOT/'candidate_text_math_encoder.py').read_text()
  self.assertIn('get("VLLM_OMNI_910A_TEXT_HF_MATH", "0")',text)
  self.assertIn('return super().forward_npu(x, residual)',text)
  self.assertIn('position_ids[:, :, :, None].float() * inv_freq[None, None, None, :].float()',text)
  self.assertIn('with torch.autocast(device_type=input.device.type, enabled=False):',text)

if __name__=='__main__':unittest.main(verbosity=2)
