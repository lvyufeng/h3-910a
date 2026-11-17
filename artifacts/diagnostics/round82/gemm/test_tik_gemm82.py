"""Direct TIK micro-kernel layout and dtype contracts;no device execution."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class DirectTikContracts(unittest.TestCase):
    def test_FP16_operands_FP32_L0C_and_output_are_explicit(self):
        source=(ROOT/'tik_gemm_fp16_fp32out82.py').read_text();ast.parse(source)
        self.assertIn("t.Tensor('float16',(k//16,m,16)",source)
        self.assertIn("t.Tensor('float16',(k//16,n,16)",source)
        self.assertIn("t.Tensor('float32',(n//16,m,16)",source)
        self.assertIn("t.tensor_mov(cu,c0,'m',1,m*n*4//1024,0,0)",source)
        self.assertIn('t.BuildCCE(',source)
        self.assertNotIn('auto_schedule(',source)
    def test_one_core_micro_scope_and_FP32_bias_are_explicit(self):
        source=(ROOT/'tik_gemm_fp16_fp32out82.py').read_text()
        self.assertIn('t.for_range(0,1,block_num=1)',source)
        self.assertIn('init_l1out=True,bias=biasl',source)
        self.assertIn("(m,k,n)!=(128,512,128)",source)
        self.assertIn("'built':True,'executed':False",source)
    def test_input_and_output_layout_roundtrips_preserve_values(self):
        m,k,n=16,32,16
        a=[[i*k+j for j in range(k)]for i in range(m)]
        packed=[[[a[i][q*16+z]for z in range(16)]for i in range(m)]for q in range(k//16)]
        recovered=[[packed[j//16][i][j%16]for j in range(k)]for i in range(m)]
        self.assertEqual(a,recovered)
        out=[[i*n+j for j in range(n)]for i in range(m)]
        cpack=[[[out[i][q*16+z]for z in range(16)]for i in range(m)]for q in range(n//16)]
        self.assertEqual(out,[[cpack[j//16][i][j%16]for j in range(n)]for i in range(m)])
    def test_native_control_uses_original_ND_values_not_packed_operands(self):
        source=(ROOT/'run_tik_gemm_probe82.py').read_text();ast.parse(source)
        self.assertIn('a_np.reshape(m,k//16,16).transpose(1,0,2)',source)
        self.assertIn('b_np.reshape(n,k//16,16).transpose(1,0,2)',source)
        self.assertIn('numpy().transpose(1,0,2).reshape(m,n).copy()',source)
        self.assertIn('torch.nn.functional.linear(ac.to(dev),bc.to(dev)',source)
        self.assertIn('torch.nn.functional.linear(ac,bc,bh)',source)


if __name__=='__main__':unittest.main()
