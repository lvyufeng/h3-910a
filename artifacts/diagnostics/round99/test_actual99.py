"""Recorded streaming-kernel failure, corrected compiler smoke and input-guard fixes."""
from pathlib import Path
import json
import ast
import unittest
ROOT=Path(__file__).resolve().parent


class ActualFlash99Contracts(unittest.TestCase):
    def test_first_step_failure_retained_small_corrected_smoke_complete(self):
        first=json.loads((ROOT/'results/smoke01/flash_smoke99.json').read_text())
        second=json.loads((ROOT/'results/smoke02/flash_smoke99.json').read_text())
        self.assertFalse(first['completed']);self.assertTrue(second['completed'])
        self.assertTrue(second['vs_original_CPU_Half_MATH']['finite'])
        self.assertTrue(second['repeat']['raw_bits_equal']and second['inputs_unchanged'])
        self.assertEqual(second['meta']['block_dim'],4);self.assertEqual(second['meta']['GM_argument_count'],5)
        self.assertEqual(second['vs_original_CPU_Half_MATH']['npu']['shape'],[1,2,128,72])
        self.assertFalse(second['full_encoder_precision_accepted']or second['whole_goal_complete'])
    def test_fold_fix_keeps_pairwise_arithmetic_and_original_failed_source(self):
        old=(ROOT/'flash/flash_vision99.py').read_text()
        fixed=(ROOT/'flash/flash_vision_step_fixed99.py').read_text();ast.parse(fixed)
        self.assertIn('t.h_max(fold_h_a, p_hi_h[0, :, :]',old)
        self.assertNotIn('t.h_max(fold_h_a, p_hi_h[0, :, :]',fixed)
        self.assertIn('t.vec_max(128, fold_h_a',fixed)
        self.assertIn('t.vec_add(f32_mask, fold_f_a, s_f_flat[0]',fixed)
        self.assertIn('t.h_reduce_max(m_half, fold_h_a, 1)',fixed)
        self.assertIn('t.h_reduce_sum(row_sum, fold_f_a, 1)',fixed)
    def test_real_input_guard_failure_preserved_and_explicit_axis_offsetfix(self):
        first=json.loads((ROOT/'results/leaf01/leaf_compare99.json').read_text());self.assertFalse(first['completed'])
        self.assertIn('ReduceMax',first['failure']['message'])
        old=(ROOT/'flash_engine99.py').read_text();fixed=(ROOT/'flash_engine_guard_fixed99.py').read_text()
        self.assertIn('q.abs().amax().item()',old)
        self.assertNotIn('.amax().item()',fixed)
        self.assertIn('reshape(-1).amax(dim=0).item()',fixed)
        self.assertIn('dense.storage_offset()!=0',fixed)
        self.assertIn('clone(memory_format=torch.contiguous_format)',fixed)
        self.assertIn('reshape(-1).all(dim=0)',fixed)
        self.assertIn('copy.deepcopy(self._meta)',fixed)
    def test_exact_native_shift_broadcast_preserves_all_real_leaf_and_vision_bits(self):
        copy=json.loads((ROOT/'results/broadcast01/broadcast_copy99.json').read_text())
        self.assertTrue(copy['completed']and copy['exact_Float_copy']['raw_bits_equal'])
        old=json.loads((ROOT/'results/leaf02/leaf_compare99.json').read_text())
        new=json.loads((ROOT/'results/leaf05/leaf_compare99.json').read_text());self.assertTrue(new['completed'])
        for a,b in zip(old['images'],new['images']):
            self.assertEqual(a['flash_vs_CPU_Half_MATH']['npu']['sha256'],b['flash_vs_CPU_Half_MATH']['npu']['sha256'])
            self.assertTrue(b['all_warm_repeats_and_QKV_unchanged'])
        before=json.loads((ROOT/'results/vision01/vision99.json').read_text())
        after=json.loads((ROOT/'results/vision02/vision99.json').read_text());self.assertTrue(after['completed'])
        for name,c in after['cases'].items():
            self.assertTrue(c['exact_streaming_baseline_all27_DS_merger_SHAs_unchanged'])
            self.assertEqual(c['merged_vs_CPU']['npu']['sha256'],before['cases'][name]['merged_vs_CPU']['npu']['sha256'])
            self.assertEqual([x['output']['sha256']for x in c['raw27_layers']],[x['output']['sha256']for x in before['cases'][name]['raw27_layers']])
        self.assertTrue(after['original_parameters_unchanged']and after['pixels_unchanged']and after['CPU_reference_SHA_unchanged'])

    def test_failed_cast_shape_and_step_sources_and_receipts_preserved(self):
        for root in('leaf03','leaf04'):
            r=json.loads((ROOT/'results'/root/'leaf_compare99.json').read_text());self.assertFalse(r['completed'])
        s=(ROOT/'flash/flash_vision_shift_temp_fixed99.py').read_text()
        self.assertIn("t.h_cast(broad_float, broad_half, 'none')",s)
        self.assertIn('buffers["ub_total_bytes"] += 6656',s)
        self.assertNotIn('t.h_cast(dst_float_flat.reshape',s)

    def test_strict_real_leaf_shapes_old_NZ_SHA_and_metadata_witness(self):
        s=(ROOT/'leaf_compare_guard_fixed99.py').read_text()
        self.assertIn('candidate_out.shape==old_repeat.shape==new_repeat.shape==(1,16,4032,72)',s)
        self.assertIn("expected['R97_NZ_leaf_output_SHAs'][image]",s)
        self.assertIn('copy.deepcopy(old.prepare())',s)


if __name__=='__main__':unittest.main()
