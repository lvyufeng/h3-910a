"""Actual eight-rank complete-vision identity, Float collective counts and resident clocks."""
from pathlib import Path
import json
import statistics
import unittest
ROOT=Path(__file__).resolve().parent


class Actual107WholeVision(unittest.TestCase):
    def test_all27_DSandmerger_bits_identical_across_eight_ranks(self):
        ranks=[json.loads((ROOT/('results/vision_tp01/vision_tp_rank%d.json'%i)).read_text())for i in range(8)]
        self.assertTrue(all(r['completed']for r in ranks))
        for name in ranks[0]['cases']:
            controls=[{'merged':r['cases'][name]['merged_vs_CPU']['npu']['sha256'],
                       'raw27':[x['output']['sha256']for x in r['cases'][name]['raw27_layers']],
                       'DS':[x['npu']['sha256']for x in r['cases'][name]['DeepStack_vs_CPU']]}for r in ranks]
            self.assertTrue(all(row==controls[0]for row in controls));self.assertEqual(len(controls[0]['raw27']),27)
        for r in ranks:
            self.assertTrue(r['original_parameters_unchanged']and r['original_pixels_unchanged']and r['original_CPU_gold_unchanged'])
            self.assertTrue(r['native_recovery']['merged']['raw_bits_equal'])
            self.assertTrue(all(v['raw_bits_equal']for v in r['native_recovery']['deepstack']))
    def test_actual_one_vision_counts_genuine_H2_and_Float_HCCL(self):
        r=json.loads((ROOT/'results/vision_tp01/vision_tp_rank0.json').read_text())
        c=r['cases']['corrected_TP8_BC128'];counts=c['actual_one_vision_TP_counts']
        self.assertEqual(counts,{'attention_calls':27,'mlp_calls':27,'attention_segments':54,'all_reduce_calls':54,
                                 'column_runs':54,'row_runs':54,'primed_weight_modules':108})
        self.assertEqual(c['head2']['local_heads'],2)
        self.assertEqual(c['TP_after_warm']['geometry']['local_mlp_channels'],538)
        self.assertTrue(c['all_repeats_merged_and_DS_bits_equal'])
    def test_summary_recomputes_paired8rankmax_times_without_mixing_fullencoder(self):
        ranks=[json.loads((ROOT/('results/vision_tp01/vision_tp_rank%d.json'%i)).read_text())for i in range(8)]
        summary=json.loads((ROOT/'summary107.json').read_text())
        for name in summary['cases']:
            times=[max(r['cases'][name]['warm_s'][i]for r in ranks)for i in range(5)]
            self.assertEqual(times,summary['cases'][name]['paired8rankmax_clock_s'])
            self.assertEqual(statistics.median(times),summary['cases'][name]['median_s'])
        self.assertFalse(summary['full_encoder_precision_accepted']or summary['whole_goal_complete'])
        self.assertIsNone(summary['numerical_tolerance'])


if __name__=='__main__':unittest.main()
