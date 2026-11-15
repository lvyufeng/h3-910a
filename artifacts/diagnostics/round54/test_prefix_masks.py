#!/usr/bin/env python3
"""Host guards for arbitrary-column prefix masks and cache-version isolation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent

def module():
    s=importlib.util.spec_from_file_location('prefix_test',ROOT/'candidate_prefix_fa910a.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class PrefixTests(unittest.TestCase):
    def setUp(self):self.fa=module()
    def test_every_partial_mask_has_exact_invalid_suffix(self):
        for column in range(1,16):
            hi,lo=self.fa._partial_column_mask(column)
            for lane in range(128):
                word=lo if lane<64 else hi
                self.assertEqual((word>>(lane%64))&1,int(lane%16>=column))
    def test_full_or_invalid_partial_mask_rejected(self):
        for n in (0,16,-1,1.5):
            with self.assertRaises(self.fa.FlashAttn910AError):self.fa._partial_column_mask(n)
    def test_cache_names_include_valid_length_and_new_mask_version(self):
        args=(1,7,43776,128,128,304,30)
        k=self.fa._kernel_cache_key(*args,key_seq_limit=41769)
        self.assertTrue(k.endswith('_c30_k41769_v3'))
        self.assertNotEqual(k,self.fa._kernel_cache_key(*args,key_seq_limit=41792))
        self.assertNotEqual(k,self.fa._kernel_cache_key(*args))
    def cached(self,version):
        work=tempfile.TemporaryDirectory(prefix='h3_prefix_cache_test_');self.addCleanup(work.cleanup)
        self.fa._KERNEL_DIR=str(Path(work.name).resolve());self.fa._runtime_core_budget=lambda:30
        key=self.fa._kernel_cache_key(1,7,43776,128,128,304,30,41769)
        obj=Path(self.fa._KERNEL_DIR)/key/(key+'.o');obj.parent.mkdir();obj.write_bytes(b'test')
        obj.with_suffix('.json').write_text(json.dumps({'blockDim':30,'fa910a_key_seq_limit':41769,'fa910a_kernel_version':version}))
        return self.fa._build(1,7,43776,128,key_seq_limit=41769)
    def test_arbitrary_valid_prefix_cached_build_accepted(self):
        obj,name,cores=self.cached(3);self.assertEqual(cores,30);self.assertIn('_k41769_v3',obj)
    def test_old_mask_metadata_rejected(self):
        with self.assertRaises(self.fa.FlashAttn910AError):self.cached(2)

if __name__=='__main__':unittest.main(verbosity=2)
