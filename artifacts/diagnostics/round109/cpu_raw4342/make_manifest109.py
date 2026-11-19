#!/usr/bin/env python3
"""Write manifest109.json for the R109 raw42/raw43 CPU capture (stdlib only, no torch).

Run this from this directory after editing any file here:
``python3 make_manifest109.py``.  It records the byte SHA256 of every staged file, the
external helper/R78/HF pins this capture depends on, the output inventory, the public API,
the gates and the CPU assumptions the root still has to verify.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'manifest109.json'
FILES = ('newcapture_cpu_raw109.py', 'test_newcapture_cpu_raw109.py',
         'make_manifest109.py', 'README.md')
EXTERNALS = {
    'suffix93.py': HERE.parents[1] / 'round93/cpu/suffix93.py',
    'original_cpu_block25_92.py': HERE.parents[1] / 'round92/cpu/original_cpu_block25_92.py',
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_capture():
    spec = importlib.util.spec_from_file_location('newcapture_cpu_raw109_manifest',
                                                  HERE / 'newcapture_cpu_raw109.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    capture = load_capture()
    files, lines = {}, {}
    for name in FILES:
        path = HERE / name
        files[name] = file_sha(path)
        lines[name] = len(path.read_text().splitlines())
    externals = {}
    for name, path in EXTERNALS.items():
        digest = file_sha(path)
        externals[name] = {'path': str(path), 'sha256': digest}
    if externals['suffix93.py']['sha256'] != capture.SUFFIX_HELPER_SHA256:
        raise SystemExit('staged suffix93.py no longer matches the pinned SHA')
    if externals['original_cpu_block25_92.py']['sha256'] != capture.ORIGINAL_HELPER_SHA256:
        raise SystemExit('staged original_cpu_block25_92.py no longer matches the pinned SHA')
    manifest = {
        'round': 109, 'task': 'task-60',
        'kind': 'ORIGINAL-HF CPU FP16 raw layers 42/43 capture for an exact-input NPU '
                'block-43 probe',
        'status': 'SOURCE ONLY; never executed on a CPU model / NPU by its author',
        'files': files, 'file_lines': lines,
        'external_pins': externals,
        'r78_file_pins': capture.R78_FILE_PINS,
        'r78_tensor_pins': {
            'raw24': capture.R78_RAW24_TENSOR_SHA256,
            'raw49_selected50': capture.R78_RAW49_TENSOR_SHA256},
        'hf_modeling_source_sha256': capture.HF_SOURCE_SHA256,
        'suffix_layers': [capture.FIRST_LAYER, capture.LAST_LAYER],
        'capture_layers': list(capture.CAPTURE_LAYERS),
        'raw42_is_input_to_raw43': True,
        'expected_tensor': {'dtype': capture.EXPECTED_TENSOR_DTYPE,
                            'shape': list(capture.EXPECTED_TENSOR_SHAPE)},
        'required_full_baseline_passes': capture.REQUIRED_PASSES,
        'outputs_in_root': {
            'root_default': str(capture.ROOT_DEFAULT),
            'files': [capture.TENSOR_NAMES[42], capture.TENSOR_NAMES[43],
                      capture.TENSOR_NAMES[49], capture.REFERENCE_NAME,
                      capture.FAILURE_NAME],
            'always_written': [capture.TENSOR_NAMES[42], capture.TENSOR_NAMES[43],
                               capture.REFERENCE_NAME],
            'optional_raw49_flag': '--also-raw49',
            'never_overwrites_existing': True,
            'outside_R78_root': True},
        'api': {
            'run': 'run%s -> receipt dict (repeats must stay %d: two full gated passes)'
                   % (inspect.signature(capture.run), capture.REQUIRED_PASSES),
            'main': 'main(argv=None) -> int',
            'cli': ['--root', '--suffix-helper', '--original-helper', '--also-raw49'],
            'module_scope_imports': 'stdlib only; torch/transformers imported lazily'},
        'gates': [
            'staged suffix93.py byte SHA256 == pinned 6d062fdb...',
            'staged original_cpu_block25_92.py byte SHA256 == pinned 838027c8...',
            'staged helper constants equal this round\'s R78/HF pins',
            'R78 reference.json + inputs + raw24 + selected50 file SHAs equal their pins',
            'R78 raw24 tensor SHA == 7a2178... and raw49 tensor SHA == 829446...',
            'ALL 25 original CPU suffix boundaries 25..49 reproduce R78 bit-exactly, twice',
            'raw49 boundary == frozen selected-50 tensor; both full passes agree on all 25 SHAs',
            'saved tensors reload as plain Half[1,2043,5120]; raw42 -> layer43 == raw43 bits',
            'original parameters and every frozen R78 file hash unchanged afterwards'],
        'no_new_oracle_or_teacher': {
            'no_Float_linear_weights': True, 'no_BF16_gold_cast': True,
            'no_corrected_NPU_input_read': True, 'no_NPU_input_bundle_opened': True,
            'no_final_norm_injected': True, 'no_hidden_conditioning_cache': True,
            'numerical_tolerance': None, 'not_a_speed_measurement': True,
            'R78_gold_not_modified': True},
        'unverified_cpu_assumptions': [
            'root must stage suffix93.py and original_cpu_block25_92.py into the private root',
            'the private root must contain no previous capture/failure artifact',
            'torch 2.10.0+cpu / transformers 5.14.1 and MKLDNN must match the R78 gold',
            'two full suffix passes fit the root run limit (about 20 minutes of CPU)',
            'the root runs this only after its NPU timers finish (no concurrent benchmark)'],
        'root_follow_up_not_this_source': 'root NPU block-43 probe and any numerical '
                                          'comparison are separate work (R110)',
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(MANIFEST), 'files': files}, indent=2))


if __name__ == '__main__':
    main()