"""Portable actual-code scope and precise FP32/BF16 tail contracts."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


def functions(path):
    return {node.name:node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}


class TailContracts(unittest.TestCase):
    def test_compute_is_two_muls_one_add_one_round_no_gather_fma_or_zero_repair(self):
        tree=ast.parse((ROOT/'vision_rope_tail72.py').read_text())
        build=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='build')
        calls=[node for node in ast.walk(build) if isinstance(node,ast.Call)]
        attrs=[node.func.attr for node in calls if isinstance(node.func,ast.Attribute)]
        self.assertEqual(attrs.count('h_mul'),2);self.assertEqual(attrs.count('h_add'),1)
        self.assertEqual(sum(isinstance(node.func,ast.Name) and node.func.id=='_round_bf16' for node in calls),1)
        self.assertFalse({'vgather','vmadd','vmla','vaxpy','h_rsqrt'}&set(attrs))
        self.assertNotIn('_fix_product_zero', (ROOT/'vision_rope_tail72.py').read_text())
    def test_dma_uses_full_aligned_rows_not_half36(self):
        text=(ROOT/'vision_rope_tail72.py').read_text()
        self.assertIn('width, d, heads, padded = 1152, 72, 16, 1280',text)
        self.assertIn('width // 8',text);self.assertIn('d // 8',text);self.assertIn('width // 16',text)
        self.assertNotIn('d // 16',text)
        self.assertIn('row < tokens',text)
        self.assertIn('t.vector_dup(64, tensor, 0.0, padded // 64',text)
    def test_wrapper_retains_original_cast_cat_and_has_no_forced_input_contiguous(self):
        node=functions(ROOT/'vision_rope_tail72.py')['rope_pair'];text=ast.unparse(node)
        self.assertIn('q32, k32 = (q.float(), k.float())',text)
        self.assertIn('torch.cat((-q32[..., 36:], q32[..., :36]), dim=-1)',text)
        self.assertIn('tail(q32, rq, cos, sin)',text)
        self.assertNotIn('.contiguous(',text)
    def test_four_inputs_one_output_abi_and_current_device_dense_guards(self):
        text=(ROOT/'vision_rope_tail72.py').read_text()
        self.assertIn('inputs=[xgm, rgm, cgm, sgm], outputs=[ygm]',text)
        self.assertIn('x.data_ptr(), rotated.data_ptr(), cos.data_ptr(),',text)
        self.assertIn('sin.data_ptr(), output.data_ptr()',text)
        self.assertIn('not value.is_contiguous()',text);self.assertIn('value.data_ptr() % 32',text)
        self.assertIn('x.device.index != torch.npu.current_device()',text)
    def test_fixed_other_math_and_complete_timer_match_round71(self):
        old=functions(ROOT.parent/'round71'/'vision_encode_tp8.py');new=functions(ROOT/'vision_tail_tp8.py')
        for name in ('local','row_forward','attention_forward','mlp_forward','linear_fp32','native_statistics_norm','prefix_statistics_norm','norm_dispatch70','measure_part'):
            self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
        def timer(path):return next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,ast.FunctionDef) and node.name=='once_complete')
        self.assertEqual(ast.dump(timer(ROOT.parent/'round71'/'vision_encode_tp8.py')),ast.dump(timer(ROOT/'vision_tail_tp8.py')))
    def test_original_entry_is_already_inference_mode_not_a_new_precision_change(self):
        tree=ast.parse((ROOT.parent/'round65'/'source'/'candidate_padded_text_encoder.py').read_text())
        entry=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='encode_ids')
        self.assertIn('torch.inference_mode()', [ast.unparse(node) for node in entry.decorator_list])
    def test_capture_saves_complete_packed_storage_including_v_and_bounded_files(self):
        text=(ROOT/'capture_vision72.py').read_text()
        self.assertIn("'packed_qkv': packed.detach().cpu().contiguous().clone()",text)
        self.assertIn('(8064, 3, 16, 72), (3456, 1152, 72, 1)',text)
        self.assertIn('128 * 1024 * 1024',text);self.assertIn('512 * 1024 * 1024',text)
        self.assertIn('candidate._apply_rotary_pos_emb_vision = original_rope',text)
        self.assertIn('No', (ROOT/'vision_rope_tail72.py').read_text())


if __name__=='__main__':unittest.main()
