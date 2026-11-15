#!/usr/bin/env python3
"""Structural regression for the optional text FP32 GEMM candidate."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent
class LinearContracts(unittest.TestCase):
 def test_other_math_and_parameter_loaders_unchanged(self):
  a=ast.parse((ROOT/'candidate_text_math_encoder.py').read_text());b=ast.parse((ROOT/'candidate_text_linear_encoder.py').read_text())
  # The deployed candidate later corrects explanatory docstrings; preserve the
  # exact experimentally executed candidate and compare executable AST instead.
  class StripDocs(ast.NodeTransformer):
   def strip(self,node):
    self.generic_visit(node)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
    return node
   visit_ClassDef=strip;visit_FunctionDef=strip
  a=StripDocs().visit(a);b=StripDocs().visit(b)
  table=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
  x=table(a);y=table(b);self.assertEqual(set(y)-set(x),{'_text_fp32_linear_enabled','_text_fp32_local_linear'})
  changed={'MiniMaxH3Qwen3VLMergedColumnParallelLinear','MiniMaxH3Qwen3VLQKVParallelLinear','MiniMaxH3Qwen3VLRowParallelLinear','MiniMaxH3Qwen3VLTextAttention','MiniMaxH3Qwen3VLTextMLP'}
  for name in x:
   if name not in changed:self.assertEqual(ast.dump(x[name]),ast.dump(y[name]),name)
   else:
    xm={n.name:n for n in x[name].body if isinstance(n,ast.FunctionDef)};ym={n.name:n for n in y[name].body if isinstance(n,ast.FunctionDef)}
    for method in xm:
     if method!='forward':self.assertEqual(ast.dump(xm[method]),ast.dump(ym[method]),name+'.'+method)
 def test_new_forward_paths_are_optin_and_quantized_paths_preserved(self):
  tree=ast.parse((ROOT/'candidate_text_linear_encoder.py').read_text())
  for name in ('MiniMaxH3Qwen3VLMergedColumnParallelLinear','MiniMaxH3Qwen3VLQKVParallelLinear'):
   c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name);fn=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
   self.assertIsInstance(fn.body[0],ast.If)
   self.assertIn('not _is_quantized_linear(self)',ast.unparse(fn.body[0].test))
   self.assertEqual(ast.unparse(fn.body[-1]),'return self.quant_method.apply(self, input_)')
  text=(ROOT/'candidate_text_linear_encoder.py').read_text()
  self.assertIn('get("VLLM_OMNI_910A_TEXT_FP32_LINEAR", "0")',text)
  self.assertIn('input_.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16',text)
 def test_row_partials_remain_fp32_until_after_reduce(self):
  tree=ast.parse((ROOT/'candidate_text_linear_encoder.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MiniMaxH3Qwen3VLRowParallelLinear')
  fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward');branch=fn.body[1]
  text=ast.unparse(branch)
  self.assertIn('F.linear(input_parallel.float(), self.weight.float())',text)
  self.assertLess(text.index('self.group.all_reduce(output_parallel)'),text.index('return output_parallel.to(self.output_dtype)'))
 def test_finite_checks_do_not_invoke_unsupported_bf16_npu_operator(self):
  for name in ('text_only_tp8.py','text_fp32_tp8.py'):
   text=(ROOT/name).read_text()
   self.assertIn('torch.isfinite(cpu_out.float())',text)
   self.assertNotIn('torch.isfinite(out)',text)
if __name__=='__main__':unittest.main(verbosity=2)
