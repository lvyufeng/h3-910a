"""Static guards for exact-CPU conditioning interventions on the unchanged text graph."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class IsolationContracts(unittest.TestCase):
    def test_previous_driver_is_immutable_and_current_root_is_distinct(self):
        p=ROOT.parent/'round81'/'encode_fp16_padded_kv_fixed81.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'519ffad3d6ca409a9eca6d228fed5a03f294ad26702f77a39151f31979575116')
        source=(ROOT/'condition_isolation82.py').read_text();ast.parse(source)
        self.assertIn('/tmp/h3_round82_condition_isolation01',source)
    def test_three_factor_cases_keep_same_real_text_model(self):
        source=(ROOT/'condition_isolation82.py').read_text()
        for case in ('CPU_initial_only','CPU_DeepStack_only','CPU_initial_and_DeepStack'):
            self.assertIn(case,source)
        self.assertIn('encoder.text_model(',source)
        self.assertIn('if len(case_layers) != 50:',source)
        self.assertIn('kv_adapter.prime(encoder.text_model)',source)
    def test_exact_transfer_positions_masks_and_native_recovery_are_required(self):
        source=(ROOT/'condition_isolation82.py').read_text()
        self.assertIn('CPU FP16 conditioning changed during NPU transfer',source)
        self.assertIn('position IDs differ;cannot isolate only vision conditioning',source)
        self.assertIn('visual mask differs;cannot isolate only vision conditioning',source)
        self.assertIn('native output did not recover after isolation',source)
        self.assertIn('e34340b57f0a800cfd3c479a91dc7ea55ee86be934d07a87d91d296eb9678a5e',source)
    def test_isolation_is_not_full_encoder_timing_or_new_teacher(self):
        source=(ROOT/'condition_isolation82.py').read_text()
        self.assertIn("'not_full_encoder_performance': True",source)
        self.assertIn("'source': str(CPU_ROOT/'FP16_multimodal_inputs.pt')",source)
        self.assertIn("'full_encoder_precision_accepted': False",source)
        self.assertIn('hidden = stock_forward() if use_stock else run_equivalent',source)
        self.assertIn('del cpu_on_npu, restored_native',source)


if __name__=='__main__':unittest.main()
