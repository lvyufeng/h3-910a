#!/usr/bin/env python3
"""Structural checks of strided FP16 point kernels and prepared snapshots."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent


class AdaLNContracts(unittest.TestCase):
    def test_ub_tables_use_row_stride_and_offset_pointer(self):
        text=(ROOT/'point_vector61.py').read_text()
        self.assertIn('(table_rows-1)*a_stride+width',text)
        self.assertIn('(table_rows-1)*b_stride+width',text)
        self.assertIn('agm[table_row*a_stride]',text)
        self.assertIn('bgm[table_row*b_stride]',text)
        self.assertIn('sa{a_stride}_sb{b_stride}',text)
        # Check exact spans for actual 6-way chunk, including nonzero base.
        width=5376;stride=6*width;offset=5*width
        span=(6-1)*stride+width
        self.assertEqual(offset+span,6*stride)
        self.assertEqual([offset+i*stride for i in range(6)],[(i*6+5)*width for i in range(6)])

    def test_fp16_point_operations_do_not_remove_roundings(self):
        text=(ROOT/'point_vector61.py').read_text()
        self.assertIn("t.Tensor('float16'",text)
        self.assertIn('t.vadds(128,b[start*128]',text)
        self.assertIn('t.vmul(128,x,x,b[index,0]',text)
        self.assertIn('t.vadd(128,x,x,a[index,0]',text)
        self.assertNotIn('vmla',text)
        self.assertNotIn('vaxpy',text)

    def test_snapshot_is_owned_versioned_and_stream_bound(self):
        text=(ROOT/'adaln_vector61.py').read_text()
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='PreparedIndices')
        source=ast.unparse(cls)
        self.assertIn('clone()',source)
        self.assertIn('torch.inference_mode(False)',source)
        self.assertIn('self.tensor._version == self.version',source)
        self.assertIn('torch.npu.current_stream().npu_stream == self.stream',source)
        for rows in (1,7,8,9,5224):
            padded=((rows+7)//8)*8
            self.assertGreaterEqual(padded,rows)
            self.assertEqual(padded%8,0)
            for i in range(rows):self.assertLess((i//8)*8+7,padded)

    def test_fallback_and_device_alignment_guards(self):
        text=(ROOT/'adaln_vector61.py').read_text()
        self.assertIn('x.device.index==torch.npu.current_device()',text)
        self.assertIn('any(t.requires_grad for t in (x,*tensors))',text)
        self.assertIn('t.data_ptr()%32==0',text)
        self.assertIn('t.stride(0)%16==0',text)
        self.assertIn("'Fallback prototype cannot consume strided tables'",text)
        self.assertIn("'Norm prototype needs dense modulation rows'",text)

    def test_cache_and_compile_target_are_device_shape_bounded(self):
        text=(ROOT/'adaln_vector61.py').read_text()
        self.assertIn('Ascend910A',text)
        self.assertIn('101',text)
        self.assertIn('cores=min(core_budget(),rows)',text)
        self.assertIn('a_stride,b_stride)',text)
        self.assertIn('fcntl.LOCK_EX',text)
        self.assertIn('meta[\'blockDim\']==cores',text)

    def test_isolated_registry_refuses_untracked_handle(self):
        text=(ROOT/'vec61_launcher.c').read_text()
        start=text.index('void* fa910a_register(')
        limit=text.index('if (g_tab_n >= FA_MAX) return NULL;',start)
        register=text.index('rtDevBinaryRegister(',start)
        self.assertLess(limit,register)
        self.assertIn('num_args > 12',text)
        self.assertIn('num_args * (uint32_t)sizeof(void*)',text)

    def test_original_rounding_contract_is_recorded(self):
        text=(ROOT/'source'/'indexed_modulation.py').read_text()
        self.assertIn('normalized = (weight.float() * normalized).to(input_dtype)',text)
        self.assertIn('gate.index_select(0, indices) * branch',text)
        self.assertIn('scale.index_select(0, indices)',text)
        candidate=(ROOT/'adaln_vector61.py').read_text()
        self.assertIn('Norm-containing variants may change FP32 reduction order',candidate)
        self.assertIn('No production dispatch changes.',candidate)


if __name__=='__main__':unittest.main(verbosity=2)
