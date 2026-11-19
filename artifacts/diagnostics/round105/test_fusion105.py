"""Native general-Float broadcast proof plus BC128 zero-extra-buffer fusion guards."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import inspect
import json
import unittest
import float_row_bits105
ROOT=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('bitcopy105_under_test',ROOT/'flash_vision_bitcopy105.py')
K=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(K)


class Fusion105Contracts(unittest.TestCase):
    def test_identical_embedded_helper_and_no_extra_UB(self):
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(K.broadcast_float_row_bits))),
                         ast.dump(ast.parse(inspect.getsource(float_row_bits105.broadcast_float_row_bits))))
        p=K.plan_kernel();self.assertEqual(p['buffers']['ub_total_bytes'],190848)
        self.assertEqual(p['logical']['mb'],64);self.assertEqual(p['logical']['bc'],128)
    def test_reuses_shift_temps_for_both_trueFloat_states_without_cast_or_cache(self):
        s=(ROOT/'flash_vision_bitcopy105.py').read_text()
        for token in('broadcast_float_row_bits(t, u_lo_flat, row_resc, pv_groups, mb, shift_copy, shift_broad, shift_broad_float)',
                     'broadcast_float_row_bits(t, u_lo_flat, l_f, pv_groups, mb, shift_copy, shift_broad, shift_broad_float)'):
            self.assertIn(token,s)
        self.assertNotIn('_broadcast_row(',s)
        self.assertIn('t.h_mul(o_f, o_f, u_lo)',s);self.assertIn('t.h_div(u_hi, o_f, u_lo)',s)
    def test_operator_probe_exact_all_Float_words_verified_actualSDK(self):
        r=json.loads((ROOT/'results/broadcast01/broadcast_float105.json').read_text());self.assertTrue(r['completed'])
        self.assertEqual(r['helper_SHA256'],hashlib.sha256((ROOT/'float_row_bits105.py').read_bytes()).hexdigest())
        self.assertEqual(len(r['cases']),4)
        for c in r['cases'].values():
            self.assertTrue(c['exact_all_Float_bits']and c['repeated_all_bits_equal']and c['input_bits_unchanged'])
    def test_combined_smoke_and_real_leaf_require_originalBC128_bits(self):
        s=(ROOT/'bc128_bitcopy_smoke105.py').read_text()
        self.assertIn("c['matches_original_R101_BC128_all_output_bits']=True",s)
        s=(ROOT/'leaf_compare105.py').read_text()
        self.assertIn("assert previous.compare(candidate_out,baseline)['raw_bits_equal']",s)
        self.assertIn("expected['R101_BC128_leaf_output_SHAs'][image]",s)
        self.assertIn('candidate_out.shape==old_repeat.shape==new_repeat.shape==(1,16,4032,72)',s)


if __name__=='__main__':unittest.main()
