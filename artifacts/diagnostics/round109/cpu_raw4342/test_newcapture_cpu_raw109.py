#!/usr/bin/env python3
"""Stdlib-only contracts for the R109 original-CPU raw42/raw43 capture.

No torch, transformers, CPU model, NPU, CANN, SSH or git is used here: the checks cover
source discipline, pinned helper/reference identities, pure gate logic, CLI surface and the
manifest.  They are structure only, not a CPU execution result or any numerical claim.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUFFIX_HELPER = REPO / 'round93/cpu/suffix93.py'
ORIGINAL_HELPER = REPO / 'round92/cpu/original_cpu_block25_92.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = load_module('capture109_under_test', HERE / 'newcapture_cpu_raw109.py')
SOURCE = (HERE / 'newcapture_cpu_raw109.py').read_text()


class SourceContracts(unittest.TestCase):
    def test_module_scope_imports_are_stdlib_only(self):
        tree = ast.parse(SOURCE)
        roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split('.')[0])
        self.assertEqual(roots, {'__future__', 'argparse', 'hashlib', 'importlib', 'json',
                                 'os', 'sys', 'pathlib'})
        self.assertNotIn('torch', roots)          # torch is imported lazily inside functions

    def test_capture_never_opens_an_npu_bundle_or_runs_the_r93_full_driver(self):
        for forbidden in ('load_inputs_bundle', 'INPUTS_DEFAULT', 'NPU_raw24_inputs92',
                          'R91_RAW24_SHA256', 'suffix.run_reference', 'run_reference('):
            self.assertNotIn(forbidden, SOURCE)

    def test_saved_payload_is_a_plain_tensor_and_gated_before_success(self):
        for token in ('torch.save(tensor, path)', 'if not torch.is_tensor(reloaded)',
                      "'plain_torch_Tensor': True", 'raw42_is_input_to_raw43',
                      'no_NPU_input_bundle_read', "'numerical_tolerance': None"):
            self.assertIn(token, SOURCE)
        self.assertLess(SOURCE.index('boundary_gate(shas, truths'), SOURCE.index('atomic_save('))

    def test_module_is_never_imported_with_torch_available(self):
        self.assertEqual(CAPTURE.CAPTURE_LAYERS, (42, 43))
        self.assertEqual(CAPTURE.REQUIRED_PASSES, 2)
        self.assertFalse(hasattr(CAPTURE, 'torch'))


class PinContracts(unittest.TestCase):
    def test_pinned_helpers_match_the_repository_files(self):
        self.assertEqual(CAPTURE.file_sha(SUFFIX_HELPER), CAPTURE.SUFFIX_HELPER_SHA256)
        self.assertEqual(CAPTURE.file_sha(ORIGINAL_HELPER), CAPTURE.ORIGINAL_HELPER_SHA256)
        suffix = load_module('suffix93_pincheck', SUFFIX_HELPER)
        original = load_module('original92_pincheck', ORIGINAL_HELPER)
        receipt = CAPTURE.base_receipt(CAPTURE.ROOT_DEFAULT)
        CAPTURE.verify_pins(suffix, CAPTURE.SUFFIX_PINS, 'helper_constants', receipt)
        CAPTURE.verify_pins(original, CAPTURE.ORIGINAL_PINS,
                            'original_helper_constants', receipt)
        self.assertTrue(receipt['helper_constants']['all_match'])
        self.assertTrue(receipt['original_helper_constants']['all_match'])
        self.assertEqual(tuple(suffix.LAYER_INDICES), tuple(range(25, 50)))
        self.assertEqual(len(suffix.PARAMETER_SUFFIXES) * 25, 275)

    def test_verify_pins_reports_mismatch_instead_of_passing(self):
        class Fake:
            HELPER_SHA256 = 'x' * 64
        receipt = {}
        with self.assertRaises(RuntimeError):
            CAPTURE.verify_pins(Fake, {'HELPER_SHA256': 'a' * 64}, 'pins', receipt)
        self.assertFalse(receipt['pins']['all_match'])
        self.assertEqual(set(receipt['pins']['mismatched']), {'HELPER_SHA256'})

    def test_suffix_helper_loader_rejects_a_tampered_copy_then_loads_the_real_one(self):
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / 'suffix93.py'
            tampered.write_bytes(SUFFIX_HELPER.read_bytes() + b'\n# tampered\n')
            with self.assertRaises(RuntimeError) as raised:
                CAPTURE.load_suffix_helper(tampered)
            self.assertIn('SHA mismatch', str(raised.exception))
        module, path, digest = CAPTURE.load_suffix_helper(SUFFIX_HELPER)
        self.assertEqual(digest, CAPTURE.SUFFIX_HELPER_SHA256)
        self.assertEqual(Path(path), SUFFIX_HELPER)
        self.assertEqual(module.HELPER_SHA256, CAPTURE.ORIGINAL_HELPER_SHA256)


class GateContracts(unittest.TestCase):
    def test_boundary_gate_accepts_exact_25_layers_and_names_the_mismatches(self):
        truths = {index: 'sha%d' % index for index in range(25, 50)}
        self.assertEqual(CAPTURE.boundary_gate(dict(truths), truths, truths), [])
        broken = dict(truths)
        del broken[42]
        broken[43] = 'other'
        mismatched = CAPTURE.boundary_gate(broken, truths, truths)
        self.assertEqual([row['layer'] for row in mismatched], [42, 43])

    def test_validate_capture_detects_shape_dtype_finite_and_boundary_problems(self):
        truths = {42: 'a' * 64}
        good = {'shape': [1, 2043, 5120], 'dtype': 'torch.float16', 'finite': True,
                'sha256': 'a' * 64}
        self.assertEqual(CAPTURE.validate_capture(good, 42, truths), [])
        wrong = {'shape': [1, 4032, 5120], 'dtype': 'torch.float32', 'finite': False,
                 'sha256': 'b' * 64}
        self.assertEqual(CAPTURE.validate_capture(wrong, 42, truths),
                         ['shape', 'dtype', 'finite', 'R78_boundary_SHA256'])

    def test_r78_file_pins_cover_exactly_the_four_frozen_files(self):
        self.assertEqual(sorted(CAPTURE.R78_FILE_PINS),
                         sorted(['reference.json', 'FP16_multimodal_inputs.pt',
                                 'FP16_layer24.pt', 'FP16_selected50.pt']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in CAPTURE.R78_FILE_PINS:
                (root / name).write_bytes(b'payload-' + name.encode())
            (root / 'unrelated.pt').write_bytes(b'ignored')
            shas, mismatched = CAPTURE.check_r78_file_pins(root)
            self.assertEqual(sorted(shas), sorted(CAPTURE.R78_FILE_PINS))
            self.assertEqual(sorted(mismatched), sorted(CAPTURE.R78_FILE_PINS))


class RootContracts(unittest.TestCase):
    def test_refuse_existing_is_fail_closed_on_missing_or_reused_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(CAPTURE.refuse_existing(root, [CAPTURE.REFERENCE_NAME]), None)
            (root / CAPTURE.REFERENCE_NAME).write_text('{}')
            with self.assertRaises(RuntimeError):
                CAPTURE.refuse_existing(root, [CAPTURE.REFERENCE_NAME])
        with self.assertRaises(RuntimeError):
            CAPTURE.refuse_existing(Path(tempfile.gettempdir()) / 'no-such-r109-root', [])

    def test_atomic_save_stays_in_root_uses_only_declared_names_and_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / CAPTURE.TENSOR_NAMES[42]
            CAPTURE.atomic_save(root, object(),
                                destination, saver=lambda obj, path: Path(path).write_bytes(b'x'))
            self.assertTrue(destination.is_file())
            self.assertFalse(destination.with_name(destination.name + '.tmp').exists())
            outside = Path(directory).parent / CAPTURE.TENSOR_NAMES[43]
            with self.assertRaises(RuntimeError):
                CAPTURE.atomic_save(root, object(), outside,
                                    saver=lambda obj, path: Path(path).write_bytes(b'x'))
            with self.assertRaises(RuntimeError):
                CAPTURE.atomic_save(root, object(), root / 'other.pt',
                                    saver=lambda obj, path: Path(path).write_bytes(b'x'))

    def test_base_receipt_states_boundaries_and_makes_no_claim(self):
        receipt = CAPTURE.base_receipt(CAPTURE.ROOT_DEFAULT)
        for key in ('reads_only_original_CPU_baseline', 'no_NPU_input_bundle_read',
                    'no_corrected_raw24_case', 'no_Float_linear_weights_or_teacher',
                    'no_BF16_gold_output_cast', 'R78_files_not_modified',
                    'no_DeepStack_in_suffix', 'no_final_norm_after_layer49'):
            self.assertIn(key, receipt)
            self.assertTrue(receipt.get(key))
        self.assertFalse(receipt['completed'])
        self.assertFalse(receipt['full_encoder_precision_accepted'])
        self.assertFalse(receipt['whole_goal_complete'])
        self.assertIsNone(receipt['numerical_tolerance'])
        self.assertTrue(receipt['not_a_speed_measurement'])
        self.assertEqual(receipt['expected_tensor']['dtype'], 'torch.float16')
        self.assertEqual(receipt['expected_tensor']['shape'], [1, 2043, 5120])
        self.assertEqual(receipt['required_full_baseline_passes'], 2)

    def test_cli_surface_and_run_signature(self):
        parser = CAPTURE.build_parser()
        args = parser.parse_args(['--root', '/tmp/x', '--suffix-helper', '/a/suffix93.py',
                                  '--original-helper', '/a/original_cpu_block25_92.py',
                                  '--also-raw49'])
        self.assertEqual(str(args.root), '/tmp/x')
        self.assertTrue(args.also_raw49)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.root, CAPTURE.ROOT_DEFAULT)
        self.assertIsNone(defaults.suffix_helper)
        parameters = inspect.signature(CAPTURE.run).parameters
        for name in ('root', 'suffix_helper_path', 'original_helper_path', 'also_raw49'):
            self.assertIn(name, parameters)

    def test_file_sha_and_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'payload.bin'
            path.write_bytes(b'abc')
            self.assertEqual(CAPTURE.file_sha(path), hashlib.sha256(b'abc').hexdigest())
            receipt = Path(directory) / CAPTURE.REFERENCE_NAME
            CAPTURE._write_json(receipt, {'completed': True})
            self.assertEqual(json.loads(receipt.read_text()), {'completed': True})
            self.assertFalse(receipt.with_name(receipt.name + '.tmp').exists())


class ManifestContracts(unittest.TestCase):
    def test_manifest_covers_staged_files_externals_and_output_inventory(self):
        manifest = json.loads((HERE / 'manifest109.json').read_text())
        self.assertEqual(manifest['round'], 109)
        self.assertEqual(sorted(manifest['files']),
                         sorted(['newcapture_cpu_raw109.py', 'test_newcapture_cpu_raw109.py',
                                 'make_manifest109.py', 'README.md']))
        for name, digest in manifest['files'].items():
            self.assertEqual(CAPTURE.file_sha(HERE / name), digest)
        self.assertEqual(manifest['external_pins']['suffix93.py']['sha256'],
                         CAPTURE.SUFFIX_HELPER_SHA256)
        self.assertEqual(manifest['external_pins']['original_cpu_block25_92.py']['sha256'],
                         CAPTURE.ORIGINAL_HELPER_SHA256)
        self.assertEqual(manifest['r78_file_pins'], CAPTURE.R78_FILE_PINS)
        self.assertEqual(manifest['capture_layers'], [42, 43])
        self.assertEqual(manifest['required_full_baseline_passes'], 2)
        self.assertEqual(manifest['outputs_in_root']['always_written'],
                         [CAPTURE.TENSOR_NAMES[42], CAPTURE.TENSOR_NAMES[43],
                          CAPTURE.REFERENCE_NAME])
        self.assertTrue(manifest['outputs_in_root']['never_overwrites_existing'])
        self.assertIsNone(manifest['no_new_oracle_or_teacher']['numerical_tolerance'])
        self.assertTrue(manifest['unverified_cpu_assumptions'])


if __name__ == '__main__':
    unittest.main()