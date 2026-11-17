"""Runtime-schema guard for the evidence summarizer;synthetic numbers are test-only."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class SummarySchemaContracts(unittest.TestCase):
    def test_summarizer_uses_driver_actual_timing_and_final_check_paths(self):
        text=(ROOT/'summarize81.py').read_text();ast.parse(text)
        self.assertIn("r['cases']['final_checks']['repeat_verified']",text)
        self.assertIn("r['timing']['samples_synchronized_wall_s_including_d2h']",text)
        self.assertIn("pad['tail_nonzero_count']",text)
        self.assertNotIn("['warm_forward']",text)
        self.assertNotIn("['tail_nonzero']",text)
    def test_rank_max_is_by_same_call_before_median(self):
        text=(ROOT/'summarize81.py').read_text()
        self.assertIn("call_max=[max(item['timing'][call]for item in rows)for call in range(3)]",text)
        self.assertIn('statistics.median(call_max)',text)
    def test_cpu_accuracy_is_not_inferred_from_repeat_or_timing(self):
        text=(ROOT/'summarize81.py').read_text()
        self.assertIn("'full_encoder_precision_accepted':False",text)
        self.assertIn("'whole_goal_complete':False",text)
        self.assertIn("['selected50_token_subsets']",text)
        self.assertIn("['embedding_full_numeric']",text)


if __name__=='__main__':unittest.main()
