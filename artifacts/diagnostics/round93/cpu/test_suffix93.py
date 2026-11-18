#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round93 CPU suffix driver.

No torch, no torch_npu, no CPU model, no NPU, no SSH: these tests prove source
structure, gate *ordering* (the Round78 25-boundary gate runs before any NPU
input is read), reuse of the pinned Round92 helper, restore/write guards, entity
pins and the non-claims.  They are NOT an accuracy or speed result and they do not
execute the suffix; the root runs it after review.
"""
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'suffix93.py'
REPO = HERE.parents[1]  # artifacts/diagnostics
HELPER = REPO / 'round92' / 'cpu' / 'original_cpu_block25_92.py'

_spec = importlib.util.spec_from_file_location('suffix93_under_test', SOURCE)
suffix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suffix)
TEXT = SOURCE.read_text()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PinContracts(unittest.TestCase):
    def test_helper_pin_matches_the_committed_round92_helper(self):
        self.assertEqual(sha256(HELPER), suffix.HELPER_SHA256)
        self.assertEqual(suffix.HELPER_MODULE_NAME, 'original_cpu_block25_92')

    def test_frozen_round78_pins_are_the_recorded_reference(self):
        self.assertEqual(suffix.R78_RECEIPT_SHA256,
                         '44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9')
        self.assertEqual(suffix.R78_MM_INPUTS_FILE_SHA256,
                         '6ab37f7202731843aebce9f3f8453232c72d45b2239caba679bcb2611e626e70')
        self.assertEqual(suffix.R78_SELECTED50_FILE_SHA256,
                         '0137a434148366c3dcbf825970274ad0faa036666561d7aecc0170cf136b3fe4')
        reference = json.loads((REPO / 'round78' / 'reference.json').read_text())
        self.assertEqual(suffix.R78_RAW49_TENSOR_SHA256,
                         reference['language_layers'][49]['output']['sha256'])
        self.assertEqual(suffix.R78_RAW49_TENSOR_SHA256,
                         reference['stages']['FP16_selected50']['tensors']['pre_final_norm']['sha256'])
        self.assertNotEqual(suffix.R78_RAW49_TENSOR_SHA256,
                            reference['stages']['FP16_layer49']['file_sha256'])
        self.assertEqual(suffix.R78_RAW24_TENSOR_SHA256,
                         '7a2178195b3ec65e09ba4c7587acb052e04996a4023f6b1a5f1b2088e2eef084')

    def test_corrected_case_is_the_actual_round91_raw24_state(self):
        self.assertEqual(suffix.CORRECTED_CASE, 'corrected_vision_native_text')
        self.assertEqual(suffix.CORRECTED_RAW24_SHA256,
                         'e232f61b4260b81b08bb0cf3eb96a7df6d0a56ed3c0c82ab9d0ec0f15004a528')

    def test_roots_and_output_names_are_disjoint_round93_paths(self):
        self.assertEqual(str(suffix.ROOT_DEFAULT), '/tmp/h3_round93_suffix01')
        self.assertEqual(str(suffix.INPUTS_DEFAULT),
                         '/tmp/h3_round92_case02/NPU_raw24_inputs92.pt')
        self.assertNotIn('/', suffix.REFERENCE_NAME + suffix.PROGRESS_NAME
                         + suffix.OUTPUTS_NAME + suffix.FAILURE_NAME)
        self.assertEqual(suffix.REFERENCE_NAME, 'CPU_suffix_reference93.json')
        self.assertEqual(suffix.PROGRESS_NAME, 'CPU_suffix_progress93.json')
        self.assertEqual(suffix.OUTPUTS_NAME, 'CPU_suffix_outputs93.pt')
        self.assertEqual(suffix.FAILURE_NAME, 'CPU_suffix_failure93.json')


class LayerRangeContracts(unittest.TestCase):
    def test_suffix_is_exactly_raw_layers_25_to_49(self):
        self.assertEqual(suffix.LAYER_INDICES, tuple(range(25, 50)))
        self.assertEqual(len(suffix.LAYER_INDICES), 25)
        self.assertEqual((suffix.FIRST_LAYER, suffix.LAST_LAYER), (25, 49))
        self.assertNotIn(24, suffix.LAYER_INDICES)
        self.assertNotIn(50, suffix.LAYER_INDICES)
        self.assertEqual(suffix.SELECTED_LANGUAGE_LAYERS, 50)
        self.assertEqual(len(suffix.PARAMETER_SUFFIXES), 11)

    def test_boundary_geometry_is_the_original_h3_text_geometry(self):
        self.assertEqual((suffix.EXPECTED_BATCH, suffix.EXPECTED_SEQ,
                          suffix.EXPECTED_HIDDEN), (1, 2043, 5120))
        self.assertIn("layers[index].get('output', {}).get('sha256')", TEXT)
        self.assertIn("receipt['r78_boundary_truths']", TEXT)
        self.assertIn("if truths[LAST_LAYER] != R78_RAW49_TENSOR_SHA256", TEXT)
        self.assertIn("layers[FIRST_LAYER - 1]['output']['sha256'] != R78_RAW24_TENSOR_SHA256",
                      TEXT)
        # both the actual corrected input and the frozen selected-50 truth are shape-pinned
        self.assertEqual(TEXT.count('(EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN)'), 2)


class ReuseContracts(unittest.TestCase):
    def test_every_heavy_operation_is_reused_from_the_pinned_helper(self):
        for call in ('original.load_r78_receipt(', 'original.load_cpu_raw24(',
                     'original.load_original_text_inputs(', 'original.build_layer(',
                     'original.split_original_positions(', 'original.build_rotary(',
                     'original.build_causal_mask(', 'original.run_block25(',
                     'original.compare_pair(', 'original.tensor_info(',
                     'original.load_inputs_bundle(', 'original.install_helpers()',
                     'original.bytes_sha(', 'original.raw_tensor_sha(',
                     'original.materialise('):
            self.assertIn(call, TEXT, call)
        for reimplemented in ('def build_layer(', 'def run_block25(',
                              'def load_inputs_bundle(', 'def build_rotary(',
                              'def build_causal_mask(', 'def create_causal_mask('):
            self.assertNotIn(reimplemented, TEXT, reimplemented)
        # helpers are called with no positional shadows, so install_helpers() keeps the
        # reviewed full_row_fp32_fixed84 metrics instead of caller-supplied callables
        self.assertIn('receipt[\'helper_metrics\'] = original.install_helpers()', TEXT)
        self.assertNotIn('install_helpers(compare', TEXT)
        self.assertNotIn('original.run_reference(', TEXT)
        self.assertEqual(TEXT.count('original.run_block25('), 1)

    def test_helper_is_imported_by_verified_path_not_copied_or_patched(self):
        self.assertIn('if actual != HELPER_SHA256:', TEXT)
        self.assertIn('if file_sha(module.__file__) != HELPER_SHA256:', TEXT)
        self.assertIn('importlib.util.spec_from_file_location(HELPER_MODULE_NAME, chosen)',
                      TEXT)
        self.assertIn("if HELPER_MODULE_NAME in sys.modules:", TEXT)
        self.assertNotIn('original_cpu_block25_92.HELPER_SHA256 =', TEXT)

    def test_stale_layer25_helper_metadata_is_removed_not_published(self):
        self.assertIn("stale = info.pop('only_layer_25_weights_loaded', None)", TEXT)
        self.assertIn("'only_suffix_layers_25_to_49_loaded': True", TEXT)
        self.assertIn("'stale_helper_layer25_metadata_ignored_not_published': True", TEXT)


class GateOrderContracts(unittest.TestCase):
    def test_all_25_r78_boundaries_are_required_before_any_npu_input_is_read(self):
        gate = TEXT.index('refusing to read NPU inputs')
        load = TEXT.index('original.load_inputs_bundle(')
        self.assertLess(gate, load)
        self.assertIn("'matches_R78_all_25_boundaries': not mismatched", TEXT)
        self.assertIn('if mismatched:', TEXT)
        self.assertIn("'validated_before_any_NPU_input_was_read': True", TEXT)

    def test_suffix_reconstruction_gate_precedes_the_corrected_cases(self):
        baseline = TEXT.index("STAGE[0] = 'baseline_pass'")
        build = TEXT.index("STAGE[0] = 'build_suffix'")
        corrected = TEXT.index("STAGE[0] = 'corrected_pass:%d'")
        self.assertLess(build, baseline)
        self.assertLess(baseline, corrected)
        # a missing Round78 conversion entry must fail in seconds, not after the 25-layer build
        self.assertLess(TEXT.index('verify_conversion_inventory(reference, receipt)'),
                        TEXT.index("STAGE[0] = 'build_suffix'"))
        self.assertIn("'all_present': not missing", TEXT)
        self.assertLess(TEXT.index('parameters_before = layers_parameters_sha'),
                        TEXT.index("STAGE[0] = 'baseline_pass'"))

    def test_original_text_inputs_and_rotary_precede_both_passes(self):
        inputs = TEXT.index('original.load_original_text_inputs(')
        rotary = TEXT.index('original.build_rotary(')
        mask = TEXT.index('original.build_causal_mask(')
        baseline = TEXT.index("STAGE[0] = 'baseline_pass'")
        self.assertLess(inputs, rotary)
        self.assertLess(rotary, baseline)
        self.assertLess(mask, baseline)


class MathContracts(unittest.TestCase):
    def test_mitigation_is_half_weights_only_with_no_teacher_or_gold_cast(self):
        for forbidden in ('bfloat16', 'torch.float32', 'torch.npu', 'torch.cuda',
                          'atol', 'rtol', 'allclose', 'text_model.norm', 'final_norm('):
            self.assertNotIn(forbidden, TEXT, forbidden)
        self.assertIn("info.get('parameter_dtypes') != [str(torch.float16)]", TEXT)
        self.assertIn("'no_Float_linear_weights_or_teacher': True", TEXT)
        self.assertIn("'no_BF16_gold_output_cast': True", TEXT)
        self.assertIn("'no_final_norm_after_layer49': True", TEXT)
        self.assertIn("'no_DeepStack_in_suffix': True", TEXT)

    def test_changed_case_is_propagated_twice_and_must_be_bit_repeatable(self):
        self.assertIn('for pass_index in (1, 2):', TEXT)
        self.assertIn('if pass_shas[1] != pass_shas[2]:', TEXT)
        self.assertIn('corrected suffix propagation is not bit-repeatable', TEXT)
        self.assertIn("receipt['corrected_repeat_bits_equal_all_25_boundaries'] = True", TEXT)
        self.assertIn("'all_25_boundaries_recorded': len(shas) == len(LAYER_INDICES)", TEXT)

    def test_selected50_truth_is_the_frozen_round78_payload(self):
        self.assertIn("selected_payload['pre_final_norm']", TEXT)
        self.assertIn('R78_SELECTED50_FILE_SHA256', TEXT)
        self.assertIn("if not baseline_vs_truth['raw_bits_equal']:", TEXT)
        self.assertIn("'corrected_vs_R78_truth': corrected_vs_truth", TEXT)

    def test_error_growth_is_descriptive_not_a_tolerance(self):
        self.assertIn("'descriptive_only_no_tolerance': True", TEXT)
        self.assertIn("'selected50_over_input_amplification'", TEXT)
        self.assertIn("'numerical_tolerance': None", TEXT)
        self.assertIn("'full_encoder_precision_accepted': False", TEXT)
        self.assertIn("'whole_goal_complete': False", TEXT)
        self.assertIn("'contains_no_NPU_arithmetic_factor': True", TEXT)
        self.assertIn("'not_a_speed_measurement': True", TEXT)

    def test_isolation_semantics_state_the_limit_of_the_control(self):
        self.assertIn('It alone cannot show a unique hardware ', TEXT)
        self.assertIn('input-only propagation control', TEXT)
        self.assertNotIn('is a hardware bug', TEXT)
        self.assertNotIn('precision accepted', TEXT)


class SafetyContracts(unittest.TestCase):
    def test_no_npu_ssh_subprocess_or_cann_access(self):
        tree = ast.parse(TEXT)
        imported = set()
        for node in tree.body:  # module scope only; torch is imported lazily in functions
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or '').split('.')[0])
        self.assertEqual(imported, {'__future__', 'argparse', 'hashlib', 'importlib',
                                    'json', 'os', 'sys', 'traceback', 'pathlib'})
        for forbidden in ('subprocess', 'socket', 'urllib', 'requests', 'paramiko',
                          'torch_npu', 'set_env.sh', 'torch.distributed', 'ssh '):
            self.assertNotIn(forbidden, TEXT, forbidden)
        self.assertIn('    import torch\n', TEXT)

    def test_module_scope_stays_stdlib_only(self):
        tree = ast.parse(TEXT)
        for node in tree.body:
            self.assertIsInstance(node, (ast.Import, ast.ImportFrom, ast.Assign,
                                         ast.AnnAssign, ast.FunctionDef, ast.ClassDef,
                                         ast.Expr, ast.If), type(node).__name__)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 9)

    def test_run_root_is_never_reused_and_final_writes_are_exclusive(self):
        self.assertIn('for path in (reference_path, progress_path, outputs_path, failure_path):',
                      TEXT)
        self.assertIn('refusing to reuse a root that already holds', TEXT)
        self.assertIn('if reference_path.exists() or outputs_path.exists():', TEXT)
        self.assertIn("temporary.name != destination.name + '.tmp'", TEXT)
        self.assertIn("failure_path.open('x')", TEXT)

    def test_invariants_rehash_every_frozen_operand(self):
        for key in ('parameters_unchanged', 'cpu_raw24_unchanged', 'r78_receipt_unchanged',
                    'r78_multimodal_inputs_unchanged', 'r78_selected50_unchanged',
                    'inputs_file_unchanged', 'helper_unchanged', 'source_unchanged'):
            self.assertIn(key, TEXT, key)


class HelperImportTests(unittest.TestCase):
    def test_pinned_helper_imports_and_the_second_import_is_refused(self):
        if suffix.HELPER_MODULE_NAME in sys.modules:
            del sys.modules[suffix.HELPER_MODULE_NAME]
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / suffix.HELPER_NAME
            staged.write_bytes(HELPER.read_bytes())
            module, chosen, digest = suffix.load_original_helper(staged)
            self.assertEqual(digest, suffix.HELPER_SHA256)
            self.assertEqual(Path(chosen).name, suffix.HELPER_NAME)
            self.assertTrue(getattr(module, '__doc__', '').startswith('Round92'))
            self.assertEqual(sha256(module.__file__), suffix.HELPER_SHA256)
            for attribute in ('load_r78_receipt', 'load_cpu_raw24', 'build_layer',
                              'run_block25', 'install_helpers', 'compare_pair',
                              'load_inputs_bundle', 'R78_RECEIPT_NAME',
                              'R78_MM_INPUTS_NAME', 'CPU_THREADS'):
                self.assertTrue(hasattr(module, attribute), attribute)
            with self.assertRaises(RuntimeError) as raised:
                suffix.load_original_helper(staged)
            self.assertIn('already imported', str(raised.exception))
        del sys.modules[suffix.HELPER_MODULE_NAME]

    def test_helper_pristine_copy_is_byte_identical_to_the_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / suffix.HELPER_NAME
            staged.write_bytes(HELPER.read_bytes())
            self.assertEqual(sha256(staged), suffix.HELPER_SHA256)

    def test_wrong_helper_bytes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / suffix.HELPER_NAME
            staged.write_bytes(HELPER.read_bytes() + b'\n')
            with self.assertRaises(RuntimeError) as raised:
                suffix.load_original_helper(staged)
            self.assertIn('Round92 helper SHA mismatch', str(raised.exception))


class WriteGuardTests(unittest.TestCase):
    def test_atomic_replace_accepts_only_root_local_tmp_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'CPU_suffix_outputs93.pt'
            temporary = root / (destination.name + '.tmp')
            temporary.write_text('payload')
            suffix.atomic_replace(root, temporary, destination)
            self.assertEqual(destination.read_text(), 'payload')
            self.assertFalse(temporary.exists())
            good = root / 'x.tmp'
            good.write_text('x')
            with self.assertRaises(RuntimeError):
                suffix.atomic_replace(root, good, root / 'y')
            outside = Path(directory).parent / 'outside93.tmp'
            with self.assertRaises(RuntimeError):
                suffix.atomic_replace(root, outside, root / 'outside93')

    def test_file_sha_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'blob.bin'
            path.write_bytes(b'round93' * 1000)
            self.assertEqual(suffix.file_sha(path),
                             hashlib.sha256(b'round93' * 1000).hexdigest())


class ManifestTests(unittest.TestCase):
    def test_manifest_matches_the_delivered_files_and_stays_non_claiming(self):
        manifest = json.loads((HERE / 'manifest93.json').read_text())
        for name, entry in manifest['files'].items():
            self.assertEqual(sha256(HERE / name), entry['sha256'], name)
            self.assertEqual((HERE / name).stat().st_size, entry['bytes'], name)
        self.assertFalse(manifest['executed_on_cpu'])
        self.assertFalse(manifest['runs_real_model'])
        self.assertFalse(manifest['proves_precision'])
        self.assertFalse(manifest['proves_speed'])
        self.assertTrue(manifest['no_ssh_or_remote_command'])
        self.assertTrue(manifest['no_install_or_build'])
        self.assertTrue(manifest['no_production_files_modified'])
        self.assertEqual(manifest['helper']['sha256'], suffix.HELPER_SHA256)
        self.assertEqual(manifest['api']['entry'], 'run_reference(root, inputs_path, '
                                                   'helper_path, baseline_repeats, case)')


if __name__ == '__main__':
    unittest.main()