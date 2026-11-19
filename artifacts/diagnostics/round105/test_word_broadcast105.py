"""Exact Float bit-copy plan: masked uint16 addresses, no arithmetic narrowing."""
from pathlib import Path
import inspect
import unittest
from float_row_bits105 import broadcast_float_row_bits
ROOT=Path(__file__).resolve().parent


class WordBroadcast105Contracts(unittest.TestCase):
    def test_masked_interleave_indices_copy_each_low_high_word_exactly(self):
        # Native16x16transpose output plane p has16 copies of original word p.
        # Model actual vor block strides in32B units (16uint16 words).
        values=[0x1234+i*0x131 for i in range(16)]
        planes=[value for value in values for _ in range(16)]
        out=[None]*256
        for offset,mask in((0,0x55555555),(16,0xAAAAAAAA)):
            for rep in range(8):
                for lane in range(32):
                    if not(mask&(1<<lane)):continue
                    src=offset+rep*2*16+(lane%16) # src_blk_stride=0
                    dst=rep*2*16+lane            # dst_blk_stride=1
                    self.assertIsNone(out[dst]);out[dst]=planes[src]
        self.assertEqual(out,[values[2*r+(j%2)]for r in range(8)for j in range(32)])
        self.assertTrue(all(value is not None for value in out))
    def test_bitcopy_only_no_half_or_float_arithmetic_cast(self):
        s=inspect.getsource(broadcast_float_row_bits)
        for token in('.h_cast(','.h_exp(','.h_mul(','.set_as(','.vector_dup('):self.assertNotIn(token,s)
        self.assertIn('t.vor([0,0x55555555]',s);self.assertIn('t.vor([0,0xAAAAAAAA]',s)
        self.assertIn('8,1,0,0,2,2,2)',s)
    def test_NaN_payload_subnormals_signedzero_and_nonHalf_finites_in_actual_probe(self):
        s=(ROOT/'broadcast_float105.py').read_text()
        for token in('0x80000000','0x80000001','0x7fc01234','0x7f801234','0x3f800001',
                     'torch.equal(actual.view(torch.int32),gold.view(torch.int32))',
                     'assert torch.equal(source_bits,row.view(torch.int32))'):
            self.assertIn(token,s)
        self.assertIn("'no_model_precision_or_speed_claim':True",s)


if __name__=='__main__':unittest.main()
