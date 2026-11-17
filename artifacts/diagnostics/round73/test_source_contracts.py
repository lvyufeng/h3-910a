"""Actual source contracts for attention casts, profiling and rejected geometry paths."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class SourceContracts(unittest.TestCase):
    def test_stridedcast_no_fp_math_and_full_aligned_token_row_transfer(self):
        text=(ROOT/'vision_cast_guard73.py').read_text()
        build=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)and node.name=='build')
        attrs={node.func.attr for node in ast.walk(build)if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)}
        self.assertFalse({'h_mul','h_add','vmuls','vadds','vgather','vconv','matmul'}&attrs)
        self.assertIn('unpack_bf16(t,words,values,transposed,zeros,padded)',text)
        self.assertIn('width,padded=1152,1280',text);self.assertIn('width//16',text)
        self.assertIn('values[head*72],0,1,9',text)
        self.assertIn('length=(tokens-1)*pitch+width',text)
        self.assertIn('torch.empty_strided(value.shape,value.stride()',text)
    def test_stridedcast_exact_bounds_current_device_and_no_contiguous_input_copy(self):
        text=(ROOT/'vision_cast_guard73.py').read_text()
        cast=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)and node.name=='cast')
        self.assertIn('pitch not in(1152,3456)',text)
        self.assertIn('value.storage_offset()*2+((n-1)*pitch+1152)*2>value.untyped_storage().nbytes()',text)
        self.assertIn('value.device.index!=torch.npu.current_device()',text)
        attrs={node.func.attr for node in ast.walk(cast)if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)}
        self.assertNotIn('contiguous',attrs)
    def test_cast_attention_preserves_matmul_scale_softmax_pv_order(self):
        text=(ROOT/'cast_attention73.py').read_text()
        self.assertIn('scores=torch.matmul(qf,kf.transpose(-2,-1))*(q.shape[-1]**-0.5)',text)
        self.assertIn('probabilities=torch.softmax(scores,dim=-1)',text)
        self.assertIn('output=torch.matmul(probabilities,vf).to(q.dtype)',text)
        self.assertNotIn('pad(',text);self.assertNotIn('clamp',text)
        self.assertIn('reference=self.original(q,k,v)',text)
        self.assertIn("row['candidate_layout']!=row['native_layout']",text)
    def test_original_text_attention_capture_cannot_be_overwritten_by_vision_binding(self):
        tree=ast.parse((ROOT/'cast_attention_fixed_tp8.py').read_text())
        assignments=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id=='original_attention'for target in node.targets):assignments.append(ast.unparse(node.value))
        self.assertEqual(assignments,['candidate.MiniMaxH3Qwen3VLTextAttention.forward'])
        self.assertIn('original_vision_attention73', (ROOT/'cast_attention_fixed_tp8.py').read_text())
    def test_complete_timer_keeps_native_cpuoutput_and_excludes_hash(self):
        def timer(path):return next(node for node in ast.walk(ast.parse(path.read_text()))if isinstance(node,ast.FunctionDef)and node.name=='once_complete')
        old=timer(ROOT.parent/'round72'/'vision_tail_tp8.py');new=timer(ROOT/'cast_attention_fixed_tp8.py')
        self.assertEqual(ast.dump(old),ast.dump(new))
    def test_inner_profile_has_no_extra_unscaled_matrix_lifetime_or_per_op_sync(self):
        text=(ROOT/'attention_stages73.py').read_text()
        self.assertIn('del unscaled',text)
        self.assertNotIn('synchronize(',text)
        self.assertIn('EventProfile(torch,capacity=1024)',text)
        self.assertIn("'same_input_original_bits_equal':original_equal",text)
    def test_capture_only_first_fullinput_source_layout_saved_and_bounded(self):
        text=(ROOT/'attention_profile_tp8.py').read_text()
        self.assertIn('CAPTURE_ACTIVE=True',text);self.assertIn('CAPTURE_ACTIVE=False',text)
        helper=(ROOT/'attention_stages73.py').read_text()
        self.assertIn('64*1024*1024',helper);self.assertIn('256*1024*1024',helper)
        self.assertIn("'storage_offset':value.storage_offset()",helper)


if __name__=='__main__':unittest.main()
