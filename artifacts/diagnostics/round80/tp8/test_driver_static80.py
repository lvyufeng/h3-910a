#!/usr/bin/env python3
"""Static contract tests for the Round80 TP8 FP16 diagnostic driver.

These tests are standard-library only: they parse the driver source with ``ast``
and inspect text.  They deliberately do NOT import the driver, so they run on a
machine without torch / torch_npu.  They also cannot prove numerics: nothing here
executes the model, and the driver itself has never been executed by its author.
"""
import ast
import hashlib
import json
import py_compile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'encode_fp16_tp8_80.py'
README = HERE / 'README.md'

CANDIDATE_SHA256 = '2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78'
CPU_SOURCE_SHA256 = '50165e77adb393b4573fbe0fdb29c62a6ddb14b8f39082d0ea67e904848010ea'
CPU_PARAMETER_SHA256 = 'a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e'


def source() -> str:
    return DRIVER.read_text(encoding='utf-8')


class DriverStaticTests(unittest.TestCase):
    def test_driver_compiles(self):
        py_compile.compile(str(DRIVER), doraise=True)

    def test_candidate_and_cpu_reference_are_pinned(self):
        text = source()
        self.assertIn(CANDIDATE_SHA256, text)
        self.assertIn(CPU_SOURCE_SHA256, text)
        self.assertIn(CPU_PARAMETER_SHA256, text)
        self.assertIn('candidate_encoder79.py', text)
        # The candidate must be hash-checked before it is executed.
        self.assertIn('if actual != CANDIDATE_SHA256:', text)

    def test_bare_text_tensor_preserves_batch_and_stock_entry_is_probed(self):
        text = source()
        self.assertIn('produce (1,seq,hidden)', text)
        self.assertIn('removes the singleton batch correctly', text)
        self.assertNotIn('documented production defect', text)
        self.assertIn('stock_encode_ids_probe', text)
        self.assertIn('use_stock = not probe[\'raised\']', text)
        self.assertIn('hidden = stock_forward() if use_stock else run_equivalent', text)

    def test_attention_override_is_scoped_and_clamp_free(self):
        text = source()
        self.assertIn('candidate._scaled_dot_product_attention = fp16_f32_math_causal_sdpa', text)
        self.assertIn("'global_functional_monkeypatch': False", text)
        self.assertIn("'torch_nn_functional_replaced': False", text)
        self.assertIn("'clamp_applied': False", text)
        # No global functional monkeypatch and no numeric clamp expression.
        self.assertNotIn('F.scaled_dot_product_attention =', text)
        self.assertNotIn('torch.nn.functional.scaled_dot_product_attention =', text)
        self.assertNotIn('clamp(', text)
        # The FP32-math causal path must be explicit.
        self.assertIn('masked_fill(causal_mask[begin:end], float(\'-inf\'))', text)
        self.assertIn('torch.softmax(scores, dim=-1)', text)
        # FP16 in, one FP16 out.
        self.assertIn('return out.to(query.dtype)', text)

    def test_cpu_reference_gate_is_fail_closed(self):
        text = source()
        for key in ('full_multimodal_reference_completed',
                    'repeat_pre_final_norm_bits_equal',
                    'repeat_post_final_norm_bits_equal',
                    'pre_final_norm_matches_layer49_identity'):
            self.assertIn(key, text)
        self.assertIn('if not all(value is True for value in gate.values()):', text)
        self.assertIn('refusing to ', text)
        self.assertIn('build an NPU precision comparison against an unfinished gold', text)
        self.assertIn("reference['original_selected_parameter_sha256'] != CPU_PARAMETER_SHA256", text)
        self.assertIn("reference['source_sha256'] != CPU_SOURCE_SHA256", text)

    def test_tp8_and_warm_call_contract(self):
        text = source()
        self.assertIn('NUM_RANKS = 8', text)
        self.assertIn('WARM_CALLS = 3', text)
        self.assertIn('text_encoder_tp_size', text)
        self.assertIn('GroupCoordinator([list(range(size))], rank, \'hccl\')', text)
        self.assertIn('initialize_model_parallel(sequence_parallel_size=size, ulysses_degree=size,', text)
        self.assertIn('tensor_parallel_size=1', text)

    def test_timing_includes_device_to_host_copy(self):
        text = source()
        self.assertIn('return hidden.cpu()', text)
        self.assertIn("'synchronized_wall_s_including_d2h'", text)
        self.assertIn("'not_stock_encode_ids': not use_stock", text)
        self.assertIn("'not_official_benchmark': True", text)
        # Timing must be synchronized on both sides of the measured region.
        self.assertIn('torch.npu.synchronize()', text)
        self.assertGreaterEqual(text.count('torch.npu.synchronize()'), 4)

    def test_no_precision_pass_is_claimed(self):
        text = source()
        self.assertIn("'full_encoder_precision_accepted': False", text)
        self.assertIn("'numerical_threshold_chosen': None", text)
        self.assertIn("'numerical_pass_field_absent': True", text)
        self.assertNotIn("'numerical_pass': True", text)
        self.assertIn('the numeric call belongs to the', text)
        self.assertIn('full_encoder_precision_accepted`` is always False', text)

    def test_partial_row_rounding_is_declared_not_waived(self):
        text = source()
        self.assertIn('row_parallel_partial_rounding', text)
        self.assertIn("'status': 'known deviation, not fixed, not waived'", text)
        self.assertIn('context_from_root_leaf_controls', text)
        self.assertIn('NOT measured by this driver', text)

    def test_no_removed_r65_or_r73_machinery_is_imported(self):
        tree = ast.parse(source())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        for forbidden in ('shadow_cache65', 'norm_tail_guard67', 'norm_prefix68',
                          'activation_cast69', 'cast_attention73', 'vision_cast73',
                          'native_bf16_vector58', 'rope_vector60', 'padded_fp32_projection59'):
            self.assertNotIn(forbidden, imported)
        self.assertIn('VLLM_OMNI_910A_TEXT_FP32_LINEAR', source())
        self.assertIn('would reintroduce the removed R65 shadow/padded', source())

    def test_evidence_is_exclusive_and_plan_only_is_separated(self):
        text = source()
        self.assertIn('from evidence_io80 import EvidenceWriter', text)
        self.assertIn('tp8_rank{rank}.json', text)
        self.assertIn('plan_rank{rank}.json', text)
        # Plan-only must return before any distributed/model setup.
        self.assertLess(text.index('if plan_only:'), text.index('init_distributed_environment('))

    def test_boundary_matches_h3_pre_final_norm(self):
        text = source()
        self.assertIn("any(name == 'norm' for name, _ in encoder.text_model.named_children())", text)
        self.assertIn('matches_cpu_reference_boundary', text)
        self.assertIn("'cpu_key_compared': 'pre_final_norm'", text)

    def test_cleanup_cannot_mask_the_original_exception(self):
        text = source()
        # Cleanup helpers are guarded and only run if distributed init happened.
        self.assertIn('if dist_ready:', text)
        self.assertIn("cleanup['destroy_model_parallel_error'] = repr(error)", text)
        self.assertIn('r[\'failure\'] = {', text)
        # The re-raise happens inside the except block, before the finally writes.
        self.assertLess(text.index('r[\'failure\'] = {'), text.index("r['cleanup'] = cleanup"))

    def test_readme_states_not_executed(self):
        self.assertTrue(README.exists(), 'README.md must describe the prepared-only status')
        readme = README.read_text(encoding='utf-8')
        self.assertIn('NOT EXECUTED', readme)
        self.assertIn(CANDIDATE_SHA256, readme)


if __name__ == '__main__':
    unittest.main()