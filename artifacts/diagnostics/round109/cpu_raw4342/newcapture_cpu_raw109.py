#!/usr/bin/env python3
"""R109: persist the ORIGINAL-HF CPU FP16 raw layers 42/43 for an exact-input NPU block-43 probe.

STATUS: SOURCE ONLY.  The author never ran this against torch / transformers / a CPU model /
an NPU / CANN, never used SSH, and edited no production file.  The root stages the immutable R93
``suffix93.py`` adapter and the pinned R92 ``original_cpu_block25_92.py`` helper into a NEW
private root and runs this script *after* its NPU timers finish.

It reuses the R93 original-HF CPU suffix adapter verbatim: rebuild original text decoder layers
25..49 from the pinned checkpoint, rebuild the original M-RoPE rows, causal mask and
``SDPBackend.MATH`` attention context from the frozen R78 inputs, then propagate the frozen R78
CPU raw layer-24 state through the suffix.  Before anything is written, **all 25** R78 raw
boundaries (25..49) must match their frozen SHA256 values and a second full pass must reproduce
all 25 SHAs bit-identically; raw49 must equal the frozen selected-50 tensor.

Only then are the two ORIGINAL CPU tensors R93 never persisted saved here, each a plain
``torch.Tensor`` ``Half[1, 2043, 5120]``, CPU, contiguous:

``CPU_raw42_109.pt``  state after original raw layer 42 == exact INPUT of raw layer 43
``CPU_raw43_109.pt``  state after original raw layer 43

Raw layer 42 is the real input of block 43 (no DeepStack after raw24, no final norm), so the
pair is the original CPU FP16 reference for an exact-input NPU layer-43 comparison.

It is not a new oracle, a Float-weight teacher, a BF16 gold cast, a corrected-input case, a
hidden conditioning cache, a numerical tolerance or a speed measurement; the R78 files are
read only and must hash unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT_DEFAULT = Path('/tmp/h3_round109_cpu01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
R78_ROOT = Path('/tmp/h3_round78_hf_fp1601')

# --- immutable staged adapters: imported by verified byte SHA, never edited -------------
SUFFIX_HELPER_NAME = 'suffix93.py'
SUFFIX_HELPER_SHA256 = '6d062fdb04e484a35a103df51024fb13ba622b6fdb60de148ba0a93205f4fdd6'
SUFFIX_MODULE_NAME = 'suffix93_r109'
ORIGINAL_HELPER_SHA256 = '838027c883bed6d1e5a0b839e0c2accde2457bef9cc81d58129fdcad38d65f4b'

# --- frozen R78 CPU FP16 gold that this capture must reproduce, never modify -------------
R78_RECEIPT_SHA256 = '44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
R78_MM_INPUTS_FILE_SHA256 = '6ab37f7202731843aebce9f3f8453232c72d45b2239caba679bcb2611e626e70'
R78_RAW24_FILE_SHA256 = '6d081a27aa99eeaf0bd73084a72d09c969f180788041d246cf0b1983e182b0e9'
R78_RAW24_TENSOR_SHA256 = '7a2178195b3ec65e09ba4c7587acb052e04996a4023f6b1a5f1b2088e2eef084'
R78_SELECTED50_NAME = 'FP16_selected50.pt'
R78_SELECTED50_FILE_SHA256 = '0137a434148366c3dcbf825970274ad0faa036666561d7aecc0170cf136b3fe4'
R78_RAW49_TENSOR_SHA256 = '82944620090359e7fc7eccb86471117e010b7838da83de5a43ceb46bf3453bd5'
HF_SOURCE_SHA256 = 'fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
R78_FILE_PINS = {
    'reference.json': R78_RECEIPT_SHA256,
    'FP16_multimodal_inputs.pt': R78_MM_INPUTS_FILE_SHA256,
    'FP16_layer24.pt': R78_RAW24_FILE_SHA256,
    R78_SELECTED50_NAME: R78_SELECTED50_FILE_SHA256,
}

FIRST_LAYER, LAST_LAYER = 25, 49
CAPTURE_LAYERS = (42, 43)
EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN = 1, 2043, 5120
EXPECTED_TENSOR_SHAPE = (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN)
EXPECTED_TENSOR_DTYPE = 'torch.float16'
EXPECTED_CPU_THREADS = 16          # read from the pinned R92 helper's CPU_THREADS constant
EXPECTED_VERSIONS = {'torch': '2.10.0+cpu', 'transformers': '5.14.1'}
REQUIRED_PASSES = 2                # two full baseline passes must agree on all 25 SHAs

REFERENCE_NAME = 'CPU_raw_reference109.json'
FAILURE_NAME = 'CPU_raw_failure109.json'
TENSOR_NAMES = {42: 'CPU_raw42_109.pt', 43: 'CPU_raw43_109.pt', 49: 'CPU_raw49_109.pt'}
STAGE = ['start']

# Helper constants this round was written against; checked before any model build.
SUFFIX_PINS = {
    'HELPER_SHA256': ORIGINAL_HELPER_SHA256,
    'R78_RECEIPT_SHA256': R78_RECEIPT_SHA256,
    'R78_MM_INPUTS_FILE_SHA256': R78_MM_INPUTS_FILE_SHA256,
    'R78_RAW24_TENSOR_SHA256': R78_RAW24_TENSOR_SHA256,
    'R78_SELECTED50_FILE_SHA256': R78_SELECTED50_FILE_SHA256,
    'R78_RAW49_TENSOR_SHA256': R78_RAW49_TENSOR_SHA256,
}
ORIGINAL_PINS = {
    'R78_RECEIPT_SHA256': R78_RECEIPT_SHA256,
    'R78_MM_INPUTS_FILE_SHA256': R78_MM_INPUTS_FILE_SHA256,
    'R78_RAW24_FILE_SHA256': R78_RAW24_FILE_SHA256,
    'R78_RAW24_TENSOR_SHA256': R78_RAW24_TENSOR_SHA256,
    'HF_SOURCE_SHA256': HF_SOURCE_SHA256,
    'CPU_THREADS': EXPECTED_CPU_THREADS,
    'MODEL': MODEL,
    'EXPECTED_VERSIONS': dict(EXPECTED_VERSIONS),
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_suffix_helper(path=None):
    """Import the immutable R93 adapter under a private name after a byte-SHA gate."""
    candidates = [Path(path)] if path is not None else [
        Path(__file__).resolve().parent / SUFFIX_HELPER_NAME, ROOT_DEFAULT / SUFFIX_HELPER_NAME]
    chosen = next((item for item in candidates if item.is_file()), None)
    if chosen is None:
        raise RuntimeError('staged R93 suffix helper not found in %r'
                           % [str(item) for item in candidates])
    actual = file_sha(chosen)
    if actual != SUFFIX_HELPER_SHA256:
        raise RuntimeError('R93 suffix helper SHA mismatch: %s != %s'
                           % (actual, SUFFIX_HELPER_SHA256))
    if SUFFIX_MODULE_NAME in sys.modules:
        raise RuntimeError('%s already imported; refusing an ambiguous helper'
                           % SUFFIX_MODULE_NAME)
    spec = importlib.util.spec_from_file_location(SUFFIX_MODULE_NAME, chosen)
    module = importlib.util.module_from_spec(spec)
    sys.modules[SUFFIX_MODULE_NAME] = module
    spec.loader.exec_module(module)
    if file_sha(module.__file__) != SUFFIX_HELPER_SHA256:
        raise RuntimeError('imported R93 suffix helper changed underneath the SHA gate')
    return module, chosen, actual


def verify_pins(module, expected, label, receipt):
    """Fail closed unless every pinned constant of a staged helper matches this round."""
    mismatched = {name: {'helper': getattr(module, name, None), 'this_round': wanted}
                  for name, wanted in expected.items()
                  if getattr(module, name, None) != wanted}
    receipt[label] = {'pins_checked': sorted(expected), 'mismatched': mismatched,
                      'all_match': not mismatched}
    if mismatched:
        raise RuntimeError('staged helper differs from the pins of this round: %r' % mismatched)


def boundary_gate(shas, truths, indices):
    """Return every layer whose boundary SHA is not the frozen R78 value (empty == pass)."""
    return [{'layer': index, 'actual': shas.get(index), 'expected': truths.get(index)}
            for index in indices if shas.get(index) != truths.get(index)]


def validate_capture(info, layer, truths, expected_shape=EXPECTED_TENSOR_SHAPE,
                     expected_dtype=EXPECTED_TENSOR_DTYPE):
    """Pure checks on one captured-tensor receipt entry; returns the failed check names."""
    problems = []
    if list(info.get('shape') or []) != list(expected_shape):
        problems.append('shape')
    if info.get('dtype') != expected_dtype:
        problems.append('dtype')
    if info.get('finite') is not True:
        problems.append('finite')
    if info.get('sha256') != truths.get(layer):
        problems.append('R78_boundary_SHA256')
    return problems


def check_r78_file_pins(root=R78_ROOT):
    """Current bytes of every frozen R78 input read here versus their declared pins."""
    shas = {name: file_sha(Path(root) / name) for name in R78_FILE_PINS}
    mismatched = {name: {'actual': shas[name], 'pinned': pin}
                  for name, pin in R78_FILE_PINS.items() if shas[name] != pin}
    return shas, mismatched


def refuse_existing(root, names):
    """Fail closed: a private root must hold none of these artifact names yet."""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError('missing root directory: %s' % root)
    present = sorted(name for name in names if (root / name).exists())
    if present:
        raise RuntimeError('refusing to reuse a root that already holds %r; stage a new '
                           'private root instead of replaying' % present)


def atomic_save(root, tensor, destination, saver=None):
    """Save inside the root through ``<name>.tmp`` then ``os.replace``; never elsewhere."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.parent != root:
        raise RuntimeError('capture destination escaped the run root: %s' % destination)
    if destination.name not in TENSOR_NAMES.values():
        raise RuntimeError('unexpected capture artifact name %r' % destination.name)
    temporary = destination.with_name(destination.name + '.tmp')
    (saver or _torch_save)(tensor, temporary)
    if not temporary.is_file():
        raise RuntimeError('saver produced no file for %s' % destination.name)
    os.replace(temporary, destination)


def _torch_save(tensor, path):
    import torch

    torch.save(tensor, path)


def base_receipt(root):
    """Receipt skeleton with the non-negotiable boundaries of this capture."""
    return {
        'scope': __doc__, 'round': 109, 'task': 'task-60',
        'kind': 'original-HF CPU FP16 raw layers 42/43 capture (exact input to raw layer 43)',
        'root': str(root), 'source_sha256': file_sha(Path(__file__)),
        'capture_layers': list(CAPTURE_LAYERS), 'raw42_is_input_to_raw43': True,
        'boundary': 'original HF raw decoder layers 25..49, PRE DeepStack injection, PRE final '
                    'norm; raw42 is the exact input of raw layer 43',
        'no_DeepStack_in_suffix': True, 'no_final_norm_after_layer49': True,
        'reads_only_original_CPU_baseline': True, 'no_NPU_input_bundle_read': True,
        'no_corrected_raw24_case': True, 'no_Float_linear_weights_or_teacher': True,
        'no_BF16_gold_output_cast': True, 'R78_files_not_modified': True,
        'expected_tensor': {'dtype': EXPECTED_TENSOR_DTYPE,
                            'shape': list(EXPECTED_TENSOR_SHAPE)},
        'required_full_baseline_passes': REQUIRED_PASSES,
        'numerical_tolerance': None, 'full_encoder_precision_accepted': False,
        'whole_goal_complete': False, 'not_a_speed_measurement': True, 'completed': False,
    }


def run(root=ROOT_DEFAULT, suffix_helper_path=None, original_helper_path=None,
        repeats=REQUIRED_PASSES, also_raw49=False):
    """Capture CPU_raw42_109.pt / CPU_raw43_109.pt; write the receipt only on full success."""
    import torch

    root = Path(root)
    receipt = base_receipt(root)
    receipt['also_raw49_requested'] = bool(also_raw49)
    names = [REFERENCE_NAME, FAILURE_NAME] + [TENSOR_NAMES[i] for i in CAPTURE_LAYERS]
    if also_raw49:
        names.append(TENSOR_NAMES[LAST_LAYER])
    try:
        refuse_existing(root, names)
        root = root.resolve()
        if R78_ROOT.resolve() == root or R78_ROOT.resolve() in root.parents:
            raise RuntimeError('refusing a capture root inside the frozen R78 gold tree: %s'
                               % root)
        if repeats != REQUIRED_PASSES:
            raise RuntimeError('this capture requires %d full baseline passes, not %r'
                               % (REQUIRED_PASSES, repeats))
        if not MODEL.is_dir():
            raise RuntimeError('pinned checkpoint directory is missing: %s' % MODEL)
        STAGE[0] = 'suffix_helper'
        suffix, suffix_file, suffix_sha = load_suffix_helper(suffix_helper_path)
        receipt['suffix_helper'] = {
            'module': SUFFIX_MODULE_NAME, 'path': str(suffix_file), 'sha256': suffix_sha,
            'matches_pin': suffix_sha == SUFFIX_HELPER_SHA256,
            'imported_by_verified_path_not_edited': True}
        verify_pins(suffix, SUFFIX_PINS, 'helper_constants', receipt)
        layers = tuple(suffix.LAYER_INDICES)
        if layers != tuple(range(FIRST_LAYER, LAST_LAYER + 1)):
            raise RuntimeError('staged R93 helper suffix layers changed: %r' % (list(layers),))

        STAGE[0] = 'original_helper'
        original, helper_file, helper_sha = suffix.load_original_helper(original_helper_path)
        receipt['original_helper'] = {
            'module': suffix.HELPER_MODULE_NAME, 'path': str(helper_file), 'sha256': helper_sha,
            'matches_pin': helper_sha == ORIGINAL_HELPER_SHA256,
            'reused_verbatim_not_edited': True}
        verify_pins(original, ORIGINAL_PINS, 'original_helper_constants', receipt)
        STAGE[0] = 'cpu_helpers'
        receipt['helper_metrics'] = original.install_helpers()
        torch.set_num_threads(original.CPU_THREADS)
        receipt['cpu_execution'] = {
            'torch': torch.__version__, 'threads': torch.get_num_threads(),
            'threads_source': 'pinned R92 helper CPU_THREADS',
            'mkldnn_enabled': bool(torch.backends.mkldnn.enabled), 'works_on_cpu_only': True}
        if (torch.get_num_threads() != original.CPU_THREADS
                or not torch.backends.mkldnn.enabled):
            raise RuntimeError('could not reproduce the Round78 CPU execution settings')

        STAGE[0] = 'r78_receipt'
        reference = original.load_r78_receipt(receipt)
        STAGE[0] = 'r78_file_pins'
        shas_before, pin_mismatched = check_r78_file_pins()
        receipt['r78_files_before'] = {'sha256': shas_before, 'mismatched': pin_mismatched,
                                       'all_match_pins': not pin_mismatched}
        if pin_mismatched:
            raise RuntimeError('frozen R78 files changed: %r' % pin_mismatched)
        hf_path = Path(reference['HF_modeling_source_path'])
        hf_sha = file_sha(hf_path)
        receipt['hf_source'] = {'path': str(hf_path), 'sha256': hf_sha,
                                'matches_pin': hf_sha == HF_SOURCE_SHA256}
        if not hf_path.is_file() or hf_sha != HF_SOURCE_SHA256:
            raise RuntimeError('pinned original HF modeling source changed: %s' % hf_path)

        STAGE[0] = 'conversion_inventory'
        suffix.verify_conversion_inventory(reference, receipt)
        STAGE[0] = 'cpu_raw24'
        cpu_raw24 = original.load_cpu_raw24(receipt)
        STAGE[0] = 'original_text_inputs'
        embeds, positions = original.load_original_text_inputs(receipt)
        STAGE[0] = 'boundary_truths'
        truths = suffix.boundary_truths(original, reference, receipt)
        STAGE[0] = 'build_suffix'
        layers, hf, text_config = suffix.build_suffix_layers(original, reference, receipt)
        if len(layers) != len(suffix.LAYER_INDICES):
            raise RuntimeError('original suffix layer construction is incomplete')
        import transformers

        receipt['runtime_modules'] = {
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'matches_pinned_versions': (torch.__version__ == EXPECTED_VERSIONS['torch'] and
                                        transformers.__version__ ==
                                        EXPECTED_VERSIONS['transformers']),
            'bf16_cast_used_in_capture': False}
        if not receipt['runtime_modules']['matches_pinned_versions']:
            raise RuntimeError('runtime library versions changed: %r'
                               % receipt['runtime_modules'])

        STAGE[0] = 'positions_rotary_mask'
        text_positions, rotary_positions = original.split_original_positions(
            positions, receipt)
        position_embeddings = original.build_rotary(hf, text_config, rotary_positions,
                                                    embeds, receipt)
        mask = original.build_causal_mask(hf, text_config, embeds, text_positions, receipt)
        parameters_before = suffix.layers_parameters_sha(original, layers)
        layer_by_index = dict(zip(suffix.LAYER_INDICES, layers))

        baseline, first_shas = None, None
        for pass_index in range(1, REQUIRED_PASSES + 1):
            STAGE[0] = 'baseline_pass:%d' % pass_index
            sink = {}
            outputs, shas = suffix.propagate(
                original, layers, cpu_raw24, mask, text_positions, position_embeddings,
                sink, 'baseline_pass%d' % pass_index)
            mismatched = boundary_gate(shas, truths, suffix.LAYER_INDICES)
            if mismatched:
                raise RuntimeError('original CPU suffix failed the R78 boundary gate on pass '
                                   '%d: %r' % (pass_index, mismatched))
            if pass_index == 1:
                first_shas = dict(shas)
            elif shas != first_shas:
                raise RuntimeError('baseline repeat is not bit-identical: %r'
                                   % boundary_gate(shas, first_shas, suffix.LAYER_INDICES))
            baseline = dict(outputs)   # keep the last fully gated pass's tensors
        receipt['baseline_passes'] = {
            'passes': REQUIRED_PASSES, 'all_25_R78_boundaries_matched': True,
            'mismatched': [], 'repeat_bit_identical_all_25': True,
            'boundaries': {str(i): sink[i] for i in suffix.LAYER_INDICES},
            'R78_boundary_SHA256': {str(i): truths[i] for i in suffix.LAYER_INDICES},
            'captured_from_pass': REQUIRED_PASSES,
            'validated_before_any_capture_or_NPU_input': True}

        STAGE[0] = 'capture'
        captured = {}
        capture_layers = list(CAPTURE_LAYERS) + ([LAST_LAYER] if also_raw49 else [])
        for layer in capture_layers:
            info = original.tensor_info(baseline[layer])
            problems = validate_capture(info, layer, truths)
            if problems:
                raise RuntimeError('captured raw layer %d failed %r' % (layer, problems))
            destination = root / TENSOR_NAMES[layer]
            atomic_save(root, original.materialise(baseline[layer]), destination)
            reloaded = torch.load(destination, map_location='cpu', weights_only=True)
            if not torch.is_tensor(reloaded):
                raise RuntimeError('capture %s is not a plain torch.Tensor'
                                   % destination.name)
            reloaded_info = original.tensor_info(reloaded)
            if reloaded_info['sha256'] != info['sha256']:
                raise RuntimeError('captured %s changed on reload' % destination.name)
            captured[layer] = {
                'layer': layer, 'name': destination.name, 'path': str(destination),
                'file_sha256': file_sha(destination), 'file_bytes': destination.stat().st_size,
                'tensor': info, 'plain_torch_Tensor': True, 'reload_tensor_sha256_matches': True,
                'matches_R78_boundary_SHA256': True,
                'role': ('exact INPUT of original raw layer 43' if layer == CAPTURE_LAYERS[0]
                         else 'original OUTPUT of raw layer %d' % layer)}
        receipt['captured_tensors'] = {str(k): v for k, v in captured.items()}

        STAGE[0] = 'captured_pair_replay'
        raw42 = torch.load(root / TENSOR_NAMES[42], map_location='cpu', weights_only=True)
        raw43 = torch.load(root / TENSOR_NAMES[43], map_location='cpu', weights_only=True)
        replay = original.run_block25(layer_by_index[43], raw42, mask, text_positions,
                                      position_embeddings)
        pair = original.compare_pair(replay, raw43)
        receipt['captured_pair_replay'] = {
            'input': TENSOR_NAMES[42], 'layer': 43, 'output': TENSOR_NAMES[43],
            'comparison': pair,
            'checks': 'disk raw42 -> original run_block25(raw layer 43) == disk raw43 bits'}
        if pair['raw_bits_equal'] is not True:
            raise RuntimeError('captured pair replay is not bit-identical to CPU_raw43_109.pt')

        STAGE[0] = 'final_checks'
        parameters_after = suffix.layers_parameters_sha(original, layers)
        receipt['layers_parameters'] = {
            'before': parameters_before, 'after': parameters_after,
            'unchanged': parameters_before == parameters_after,
            'retained_parameter_tensors': (len(suffix.LAYER_INDICES)
                                           * len(suffix.PARAMETER_SUFFIXES))}
        shas_after, _ = check_r78_file_pins()
        receipt['r78_files_after'] = {'sha256': shas_after,
                                      'unchanged': shas_after == shas_before}
        receipt['original_parameters_unchanged'] = parameters_before == parameters_after
        if parameters_before != parameters_after:
            raise RuntimeError('original CPU parameters changed during the capture')
        if shas_after != shas_before:
            raise RuntimeError('frozen R78 files changed during the capture')
        receipt['completed'] = True
        _write_json(Path(root) / REFERENCE_NAME, receipt)
        return receipt
    except BaseException as error:
        receipt['failure'] = {'stage': STAGE[0], 'type': type(error).__name__,
                             'message': str(error)}
        if isinstance(root, Path) and root.is_dir() and not (root / REFERENCE_NAME).exists():
            _write_json(root / FAILURE_NAME, receipt)
        raise


def _write_json(path, payload):
    """Private-root atomic write through <name>.tmp."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT,
                        help='new private CPU root holding the staged helpers')
    parser.add_argument('--suffix-helper', type=Path, default=None,
                        help='staged immutable R93 suffix93.py (default: next to this script)')
    parser.add_argument('--original-helper', type=Path, default=None,
                        help='staged immutable R92 original_cpu_block25_92.py')
    parser.add_argument('--also-raw49', action='store_true',
                        help='additionally save CPU_raw49_109.pt for the selected-50 proof')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    receipt = run(root=args.root, suffix_helper_path=args.suffix_helper,
                  original_helper_path=args.original_helper, also_raw49=args.also_raw49)
    print(json.dumps({'completed': receipt['completed'], 'root': receipt['root'],
                      'captured': {name: entry['file_sha256']
                                   for name, entry in receipt['captured_tensors'].items()}},
                     indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())