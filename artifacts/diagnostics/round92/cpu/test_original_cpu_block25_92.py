#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round92 original CPU layer-25 oracle.

No torch, no transformers, no model, no NPU, no SSH: this file imports the
module under test as a stdlib-only module (which must succeed without torch
installed), parses its source with ``ast``, and cross-checks every pinned value
against the committed Round78 CPU reference receipt, the committed Round91
actual eight-rank receipt and the committed pinned original HF source.  Passing
these tests is NOT a CPU/NPU accuracy result and NOT a speed result.
"""

import ast
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / 'original_cpu_block25_92.py'
DIAGNOSTICS = ROOT.parents[1]
R78 = DIAGNOSTICS / 'round78' / 'reference.json'
R91 = DIAGNOSTICS / 'round91' / 'results' / 'full01' / 'ranks' / 'tp8_rank0.json'
HF_SOURCE = DIAGNOSTICS / 'round86' / 'modeling_qwen3_vl_original86.py'

_spec = importlib.util.spec_from_file_location('original_cpu_block25_92_under_test',
                                               MODULE_PATH)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)
SOURCE = MODULE_PATH.read_text()
TREE = ast.parse(SOURCE)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module_scope_imports():
    names = []
    for node in TREE.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append((node.module or '').split('.')[0])
    return names


def calls_in(node):
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def called_attr_names():
    return {call.func.attr for call in calls_in(TREE)
            if isinstance(call.func, ast.Attribute)}


def call_keyword_names():
    names = set()
    for call in calls_in(TREE):
        for keyword in call.keywords:
            if keyword.arg is not None:
                names.add(keyword.arg)
    return names


class SourceContract(unittest.TestCase):
    def test_module_scope_is_stdlib_only_and_torch_is_lazy(self):
        imports = module_scope_imports()
        self.assertTrue(imports)
        forbidden = {'torch', 'torch_npu', 'transformers', 'numpy', 'safetensors', 'tiktools'}
        self.assertFalse(forbidden & set(imports), imports)
        for name in imports:
            self.assertIn(name, {'__future__', 'argparse', 'hashlib', 'json', 'os',
                                 'traceback', 'pathlib'})
        # torch is imported only inside function bodies (deferred, execution-time).
        function_imports = [node for node in ast.walk(TREE) if isinstance(node, ast.Import)
                            and any(alias.name == 'torch' for alias in node.names)]
        self.assertGreaterEqual(len(function_imports), 3)
        for node in function_imports:
            self.assertNotIn(node, TREE.body)

    def test_docstring_declares_source_only_and_no_result(self):
        doc = ast.get_docstring(TREE) or ''
        self.assertIn('SOURCE ONLY', doc)
        for phrase in ('never been executed', 'not an accuracy, speed or',
                       'chooses no numerical tolerance'):
            self.assertIn(phrase, doc)

    def test_public_api_matches_the_root_contract(self):
        import inspect

        self.assertTrue(callable(module.run_reference))
        self.assertTrue(callable(module.main))
        signature = inspect.signature(module.run_reference)
        self.assertEqual(list(signature.parameters),
                         ['root', 'inputs_path', 'compare', 'tensor_info'])
        self.assertEqual(signature.parameters['root'].default,
                         Path('/tmp/h3_round92_case01'))
        for name in ('inputs_path', 'compare', 'tensor_info'):
            self.assertIsNone(signature.parameters[name].default)
        self.assertEqual(module.INPUTS_NAME, 'NPU_raw24_inputs92.pt')
        self.assertEqual(module.REFERENCE_NAME, 'CPU_block25_reference92.json')
        self.assertEqual(module.OUTPUTS_NAME, 'CPU_block25_outputs92.pt')
        self.assertEqual(module.LAYER_INDEX, 25)
        self.assertEqual(module.SELECTED_LANGUAGE_LAYERS, 50)
        self.assertIn("if __name__ == '__main__':", SOURCE)

    def test_outputs_payload_keys_match_the_root_contract(self):
        self.assertIn(
            "torch.save({'outputs': outputs, 'input_SHAs': input_shas,\n"
            "                    'position_ids': original_positions}, temporary_outputs)", SOURCE)
        self.assertIn("outputs = {'CPU_raw24': baseline}", SOURCE)
        self.assertIn("input_shas = {'CPU_raw24': R78_RAW24_TENSOR_SHA256}", SOURCE)
        self.assertIn('outputs_file_SHA256', SOURCE)
        self.assertIn('inputs_source_path', SOURCE)
        self.assertIn('inputs_file_SHA256', SOURCE)


class PinsMatchCommittedReceipts(unittest.TestCase):
    def test_pins_match_the_committed_round78_cpu_receipt(self):
        receipt = json.loads(R78.read_text())
        self.assertEqual(sha256(R78), module.R78_RECEIPT_SHA256)
        self.assertEqual(receipt['versions'], module.EXPECTED_VERSIONS)
        self.assertEqual(receipt['HF_modeling_source_sha256'], module.HF_SOURCE_SHA256)
        self.assertEqual(receipt['CPU_num_threads'], module.CPU_THREADS)
        self.assertIs(receipt['CPU_mkldnn_enabled'], True)
        self.assertEqual(receipt['original_language_layer_count'],
                         module.EXPECTED_ORIGINAL_NUM_HIDDEN_LAYERS)
        self.assertEqual(receipt['original_selected_parameter_sha256'],
                         module.R78_ORIGINAL_SELECTED_PARAMETER_SHA256)
        self.assertEqual(len(receipt['language_layers']), module.SELECTED_LANGUAGE_LAYERS)
        self.assertEqual(receipt['language_layers'][24]['output']['sha256'],
                         module.R78_RAW24_TENSOR_SHA256)
        self.assertEqual(receipt['language_layers'][25]['output']['sha256'],
                         module.R78_RAW25_TENSOR_SHA256)
        self.assertEqual(receipt['stages']['FP16_layer24']['file_sha256'],
                         module.R78_RAW24_FILE_SHA256)
        self.assertEqual(receipt['stages']['FP16_layer24']['tensors']['sha256'],
                         module.R78_RAW24_TENSOR_SHA256)
        self.assertEqual(receipt['stages']['FP16_layer24']['tensors']['shape'],
                         [module.EXPECTED_BATCH, module.EXPECTED_SEQ, module.EXPECTED_HIDDEN])
        self.assertEqual(receipt['stages']['FP16_layer24']['tensors']['dtype'], 'torch.float16')
        inputs = receipt['stages']['FP16_multimodal_inputs']
        self.assertEqual(inputs['file_sha256'], module.R78_MM_INPUTS_FILE_SHA256)
        self.assertEqual(inputs['tensors']['inputs_embeds']['sha256'],
                         module.R78_MM_INPUTS_EMBEDS_SHA256)
        self.assertEqual(inputs['tensors']['position_ids']['sha256'],
                         module.R78_POSITION_IDS_SHA256)
        self.assertEqual(inputs['tensors']['position_ids']['shape'],
                         [3, module.EXPECTED_BATCH, module.EXPECTED_SEQ])
        self.assertEqual(inputs['tensors']['position_ids']['dtype'], 'torch.int64')

    def test_layer_parameter_suffixes_are_exactly_the_eleven_recorded(self):
        receipt = json.loads(R78.read_text())
        conversion = receipt['parameter_conversion']
        prefix = 'language_model.layers.25.'
        recorded = sorted(name[len(prefix):] for name in conversion
                          if name.startswith(prefix))
        self.assertEqual(recorded, sorted(module.LAYER_PARAMETER_SUFFIXES))
        self.assertEqual(len(recorded), 11)
        # Shapes of the real layer-25 tensors are the pinned Qwen3-VL geometry.
        expected_shapes = {
            'self_attn.q_proj.weight': [8192, 5120], 'self_attn.k_proj.weight': [1024, 5120],
            'self_attn.v_proj.weight': [1024, 5120], 'self_attn.o_proj.weight': [5120, 8192],
            'self_attn.q_norm.weight': [128], 'self_attn.k_norm.weight': [128],
            'mlp.gate_proj.weight': [25600, 5120], 'mlp.up_proj.weight': [25600, 5120],
            'mlp.down_proj.weight': [5120, 25600], 'input_layernorm.weight': [5120],
            'post_attention_layernorm.weight': [5120],
        }
        for suffix, shape in expected_shapes.items():
            entry = conversion[prefix + suffix]
            self.assertEqual(entry['shape'], shape, suffix)
            self.assertEqual(entry['reference_dtype'], 'torch.float16', suffix)
            self.assertEqual(entry['original_dtype'], 'torch.bfloat16', suffix)

    def test_round91_raw24_pins_match_the_committed_actual_receipt(self):
        receipt = json.loads(R91.read_text())
        self.assertTrue(receipt['completed'])
        actual = {}
        for case, payload in receipt['cases'].items():
            row = [item for item in payload['raw50_layers'] if item['layer'] == 24]
            self.assertEqual(len(row), 1, case)
            actual[case] = row[0]['output']['sha256']
        self.assertEqual(sorted(actual), sorted(module.ALLOWED_CASES))
        for case, expected in actual.items():
            self.assertEqual(module.R91_RAW24_SHA256[case], expected, case)
            self.assertIn(case, module.ALLOWED_CASES)
        self.assertEqual(len({value for value in actual.values()}), 3)


class ReconstructedLayerContract(unittest.TestCase):
    def test_only_layer25_rebuilt_no_full_model_no_final_norm_no_deepstack(self):
        constructed = {call.func.attr for call in calls_in(TREE)
                       if isinstance(call.func, ast.Attribute)
                       and call.func.attr.startswith('Qwen3VL')}
        self.assertEqual(constructed, {'Qwen3VLTextDecoderLayer',
                                       'Qwen3VLTextRotaryEmbedding'})
        keywords = call_keyword_names()
        for forbidden in ('deepstack_visual_embeds', 'visual_pos_masks', 'pixel_values',
                          'image_grid_thw', 'mm_token_type_ids', 'input_ids'):
            self.assertNotIn(forbidden, keywords)
        # No attribute access ending in `.norm` (the final norm) anywhere.
        self.assertNotIn('.norm', {attribute.attr for attribute in ast.walk(TREE)
                                   if isinstance(attribute, ast.Attribute)})
        self.assertIn("text_config.num_hidden_layers = SELECTED_LANGUAGE_LAYERS", SOURCE)
        self.assertIn("only_layer_25_weights_loaded", SOURCE)
        self.assertIn("'model.' + name", SOURCE)

    def test_attention_backend_is_original_sdpa_math_without_autocast(self):
        self.assertIn('_attn_implementation = \'sdpa\'', SOURCE)
        self.assertIn('SDPBackend.MATH', SOURCE)
        self.assertIn('sdpa_kernel(backends=[SDPBackend.MATH])', SOURCE)
        for forbidden in ('autocast', 'torch.float32', 'allclose', 'atol', 'rtol',
                          'threshold'):
            self.assertNotIn(forbidden, SOURCE)
        # BF16 appears only as the *storage* dtype assertion of the original
        # checkpoint tensors; it is never a compute or output cast target.
        self.assertEqual(SOURCE.count('torch.bfloat16'), 1)
        self.assertIn('is not the original BF16 storage', SOURCE)
        for forbidden in ('.to(torch.bfloat16)', 'bfloat16()', 'dtype=torch.bfloat16'):
            self.assertNotIn(forbidden, SOURCE)
        self.assertIn('to(torch.float16)', SOURCE)
        self.assertIn('numerical_tolerance\': None', SOURCE)
        self.assertIn("'no_Float_linear_weights_or_teacher': True", SOURCE)
        self.assertIn("'no_BF16_gold_output_cast': True", SOURCE)
        self.assertIn("'no_final_norm_after_layer25': True", SOURCE)
        self.assertIn("'no_DeepStack_after_raw24': True", SOURCE)

    def test_single_layer_parameter_names_are_unprefixed_not_checkpoint_keys(self):
        self.assertIn('if set(wanted) != set(expected.values()):', SOURCE)
        self.assertNotIn('if set(wanted) != set(expected):', SOURCE)

    def test_original_sdpa_can_elide_mask_without_changing_causal_semantics(self):
        self.assertIn('if mask is not None and not torch.is_tensor(mask):', SOURCE)
        self.assertIn('assert layer.self_attn.is_causal is True', SOURCE)
        self.assertIn("'mask': None if mask is None else tensor_info(mask)", SOURCE)
        self.assertIn('matches_R78_raw25_SHA256', SOURCE)

    def test_position_split_and_mask_call_mirror_the_pinned_hf_source(self):
        self.assertEqual(sha256(HF_SOURCE), module.HF_SOURCE_SHA256)
        hf_tree = ast.parse(HF_SOURCE.read_text())
        hf_calls = [call for call in calls_in(hf_tree)
                    if isinstance(call.func, ast.Name)
                    and call.func.id == 'create_causal_mask']
        self.assertEqual(len(hf_calls), 1)
        original_kwargs = [keyword.arg for keyword in hf_calls[0].keywords]
        self.assertEqual(original_kwargs, ['config', 'inputs_embeds', 'attention_mask',
                                           'past_key_values', 'position_ids'])
        my_calls = [call for call in calls_in(TREE)
                    if isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'create_causal_mask']
        self.assertEqual(len(my_calls), 1)
        self.assertEqual([keyword.arg for keyword in my_calls[0].keywords],
                         original_kwargs)
        self.assertIsInstance(my_calls[0].keywords[3].value, ast.Constant)
        self.assertIsNone(my_calls[0].keywords[3].value.value)
        # The three original M-RoPE branches are implemented verbatim.
        self.assertIn('expand(4, position_ids.shape[0], -1)', SOURCE)
        self.assertIn('position_ids.shape[0] == 4', SOURCE)
        self.assertIn('text_positions, rotary_positions = None, position_ids', SOURCE)
        hf_source = HF_SOURCE.read_text()
        self.assertIn('text_position_ids = position_ids[0]', hf_source)
        self.assertIn('position_ids = position_ids[1:]', hf_source)
        self.assertIn('text_position_ids = None', hf_source)


class FailClosedContract(unittest.TestCase):
    def test_reference_gate_runs_before_npu_inputs_and_cases(self):
        gate = SOURCE.index("STAGE = 'reference_gate'")
        gate_check = SOURCE.index('refusing to evaluate NPU inputs')
        inputs = SOURCE.index("STAGE = 'npu_inputs'")
        cases = SOURCE.index("STAGE = 'case:'")
        self.assertLess(gate, gate_check)
        self.assertLess(gate_check, inputs)
        self.assertLess(inputs, cases)
        self.assertLess(SOURCE.index("STAGE = 'r78_receipt'"), gate)
        self.assertLess(SOURCE.index("STAGE = 'cpu_raw24'"), gate)
        self.assertIn('matches_R78_raw25_SHA256', SOURCE)
        self.assertIn('repeat_bits_equal', SOURCE)

    def test_no_overwrite_and_no_tolerance_waiver(self):
        self.assertGreaterEqual(SOURCE.count('refusing to overwrite'), 2)
        self.assertIn("with failure_path.open('x') as handle", SOURCE)
        self.assertNotIn('mode=\'w\'', SOURCE)
        self.assertIn('os.replace(temporary_outputs, outputs_path)', SOURCE)
        self.assertIn('os.replace(temporary_reference, reference_path)', SOURCE)
        self.assertIn("'full_encoder_precision_accepted': False", SOURCE)
        self.assertIn("'whole_goal_complete': False", SOURCE)

    def test_unknown_cases_and_declared_sha_mismatches_are_refused(self):
        self.assertIn('cases outside the Round91 receipt', SOURCE)
        self.assertIn('differs from the declared capture value', SOURCE)
        self.assertIn('differs from the Round91 actual receipt', SOURCE)
        self.assertIn('positions are not the original Round78 M-RoPE rows', SOURCE)
        self.assertIn('parameter %s is not a real CPU FP16 tensor', SOURCE)
        self.assertIn('FP16 conversion SHA differs from Round78', SOURCE)

    def test_root_expected_pins_cross_check_is_optional_and_fail_closed(self):
        self.assertEqual(module.EXPECTED_PINS_NAME, 'expected_raw24_92.json')
        self.assertIn('def cross_check_root_expected_pins(root, declared, receipt):', SOURCE)
        self.assertIn('redundant_gate_skipped', SOURCE)
        self.assertIn('disagrees with the Round91 actual', SOURCE)
        npu_stage = SOURCE.index("STAGE = 'npu_inputs'")
        cross_check_call = SOURCE.index('cross_check_root_expected_pins(root, declared, '
                                        'receipt)', npu_stage)
        self.assertLess(SOURCE.index("STAGE = 'reference_gate'"), cross_check_call)
        self.assertLess(cross_check_call, SOURCE.index("STAGE = 'case:'"))
        # The root's committed expected-pins file (if present) is exactly our pins.
        pins_path = ROOT.parent / 'expected_raw24_92.json'
        if pins_path.is_file():
            pins = json.loads(pins_path.read_text())['cases']
            for case, entry in pins.items():
                self.assertEqual(entry['sha256'], module.R91_RAW24_SHA256[case], case)
                self.assertIn(case, module.ALLOWED_CASES)

    def test_metrics_helper_is_optional_and_inference_safe(self):
        self.assertIn('def install_helpers(compare=None, tensor_info=None):', SOURCE)
        self.assertIn('full_row_fp32_fixed84', SOURCE)
        self.assertIn('with torch.inference_mode(False):', SOURCE)
        self.assertIn('def materialise(value):', SOURCE)


class ManifestContract(unittest.TestCase):
    def test_manifest_matches_delivered_files_and_stays_honest(self):
        manifest_path = ROOT / 'manifest92.json'
        manifest = json.loads(manifest_path.read_text())
        for name, entry in manifest['files'].items():
            path = ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(entry['bytes'], path.stat().st_size, name)
            self.assertEqual(entry['sha256'], sha256(path), name)
        for name, expected in (('original_cpu_block25_92.py', True),
                               ('test_original_cpu_block25_92.py', True),
                               ('README.md', True)):
            self.assertIn(name, manifest['files'])
            self.assertTrue(expected)
        for flag in ('source_only', 'no_ssh_or_remote_command', 'no_npu_or_gpu_execution',
                     'no_torch_or_transformers_imported_by_author', 'no_model_execution_by_author',
                     'no_production_files_modified', 'no_git_commit'):
            self.assertIs(manifest[flag], True, flag)
        for flag in ('proves_precision', 'proves_speed', 'selects_tolerance',
                     'claims_lossless_pass'):
            self.assertIs(manifest[flag], False, flag)
        self.assertIsNone(manifest['numerical_tolerance'])
        self.assertEqual(manifest['task'], 'task-40')


if __name__ == '__main__':
    unittest.main()