#!/usr/bin/env python3
"""Round92 original-HF CPU FP16 decoder-layer-25 same-input reference.

STATUS: SOURCE ONLY.  This file has never been executed by its author against
torch, transformers, a CPU model, an NPU or CANN: no SSH, no install, no
production edit, no commit.  Every number this program prints belongs to the
root agent that stages it, plus the frozen Round78 CPU gold and the Round91
actual eight-rank FP16 raw-layer-24 capture.  It is not an accuracy, speed or
official-lossless result by itself.

What it computes
----------------
The root captures, from the actual eight-rank FP16 H3 text encoder, the *raw
layer-24 output* -- exactly the Round78 ``language_layers[24]`` boundary
("rawdecoderPREDeepStackinjection", PRE final norm) -- for several full-encoder
cases.  This module rebuilds exactly one original HF decoder layer, layer 25,
out of the pinned official checkpoint, and evaluates it twice per input with

* the original HF ``Qwen3VLTextRotaryEmbedding`` (Float32 CPU inverse
  frequencies, M-RoPE interleaved layout, cos/sin cast to FP16 as in the
  original), and
* the original HF ``create_causal_mask`` call of ``Qwen3VLTextModel.forward``
  (same five keyword arguments, ``past_key_values=None``, the original
  all-ones attention mask), under ``SDPBackend.MATH``;

on (a) the frozen Round78 CPU FP16 raw layer-24 tensor (``CPU_raw24``) and
(b) every actual NPU raw layer-24 tensor of the capture bundle.  Weights,
positions, rotary tensors and the causal mask are shared by all cases: only the
layer-25 input tensor changes.

Reference gate (fail closed, before any NPU input is read)
---------------------------------------------------------
``CPU_raw24`` -> CPU layer25 must reproduce the frozen Round78 raw layer-25
output SHA256 ``a83f92f8...`` bit-exactly, and do so twice.  That single gate
validates the whole reconstruction: pinned original HF source, original FP16
parameter conversion, M-RoPE rows, causal mask, SDPA MATH attention, and the
eleven layer-25 weights.  If it does not reproduce, this module raises instead
of producing numbers.

What it is and is not
---------------------
Evaluating the original CPU FP16 block 25 on an actual NPU raw layer-24 state is
an *input-only* sensitivity control: it bounds how much of the NPU block-25 gap
is explained by different inputs versus per-block NPU arithmetic.  It is not a
Float-weight teacher, it casts no BF16 gold output, it adds no final norm after
layer 25 and no DeepStack after raw24, it chooses no numerical tolerance, and it
makes no speed claim.  The frozen Round78 CPU reference is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

ROOT_DEFAULT = Path('/tmp/h3_round92_case01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
R78_ROOT = Path('/tmp/h3_round78_hf_fp1601')

# --- frozen Round78 CPU FP16 reference (independently recomputed HF gold) ----
R78_RECEIPT_NAME = 'reference.json'
R78_RECEIPT_SHA256 = '44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
R78_MM_INPUTS_NAME = 'FP16_multimodal_inputs.pt'
R78_MM_INPUTS_FILE_SHA256 = '6ab37f7202731843aebce9f3f8453232c72d45b2239caba679bcb2611e626e70'
R78_MM_INPUTS_EMBEDS_SHA256 = 'aed76ae16f046bcaefe87296c680b0d274bbd530598e5d7fe9d7fa530d368de9'
R78_POSITION_IDS_SHA256 = '05a92359cd6a171559fadc0185ea2c7619a1969f7d771bb89c6a7d4df31a814c'
R78_RAW24_NAME = 'FP16_layer24.pt'
R78_RAW24_FILE_SHA256 = '6d081a27aa99eeaf0bd73084a72d09c969f180788041d246cf0b1983e182b0e9'
R78_RAW24_TENSOR_SHA256 = '7a2178195b3ec65e09ba4c7587acb052e04996a4023f6b1a5f1b2088e2eef084'
R78_RAW25_TENSOR_SHA256 = 'a83f92f8f9108ff05c25d7967eb575ee9a2b93e7c7463a44b9536595b519c01d'
R78_ORIGINAL_SELECTED_PARAMETER_SHA256 = (
    'a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e')
HF_SOURCE_SHA256 = 'fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
CONFIG_SHA256 = 'd2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427'
CHECKPOINT_INDEX_SHA256 = '06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a'
EXPECTED_VERSIONS = {'torch': '2.10.0+cpu', 'transformers': '5.14.1'}
CPU_THREADS = 16

# --- pinned layer geometry of the official text encoder ---------------------
LAYER_INDEX = 25
SELECTED_LANGUAGE_LAYERS = 50
EXPECTED_ORIGINAL_NUM_HIDDEN_LAYERS = 64
EXPECTED_BATCH = 1
EXPECTED_SEQ = 2043
EXPECTED_HIDDEN = 5120
LAYER_PARAMETER_SUFFIXES = (
    'input_layernorm.weight',
    'mlp.down_proj.weight',
    'mlp.gate_proj.weight',
    'mlp.up_proj.weight',
    'post_attention_layernorm.weight',
    'self_attn.k_norm.weight',
    'self_attn.k_proj.weight',
    'self_attn.o_proj.weight',
    'self_attn.q_norm.weight',
    'self_attn.q_proj.weight',
    'self_attn.v_proj.weight',
)

# --- Round91 actual NPU raw layer-24 output SHAs (rank0 receipt, all 8 ranks) -
R91_RAW24_SHA256 = {
    'native': '48a183903d547d3b5440ad8f251451dfa888ba8c1a7f88b6f79e0a666b9027f4',
    'native_geometry_cache': '48a183903d547d3b5440ad8f251451dfa888ba8c1a7f88b6f79e0a666b9027f4',
    'corrected_vision_native_text': 'e232f61b4260b81b08bb0cf3eb96a7df6d0a56ed3c0c82ab9d0ec0f15004a528',
    'corrected_vision_native_text_geometry_cache': (
        'e232f61b4260b81b08bb0cf3eb96a7df6d0a56ed3c0c82ab9d0ec0f15004a528'),
    'corrected_vision_precise_text_geometry_cache': (
        '6dd3016db8e7a6abbc425449358a4af5cf5f472d6512f239b0b63bda83a87c4b'),
}
ALLOWED_CASES = tuple(sorted(R91_RAW24_SHA256))

INPUTS_NAME = 'NPU_raw24_inputs92.pt'
REFERENCE_NAME = 'CPU_block25_reference92.json'
OUTPUTS_NAME = 'CPU_block25_outputs92.pt'
FAILURE_NAME = 'CPU_block25_failure92.json'
EXPECTED_PINS_NAME = 'expected_raw24_92.json'

STAGE = 'start'
HELPERS = {
    'tensor_info': None, 'compare': None, 'compare_source': 'unknown',
    'tensor_info_source': 'unknown', 'module': None, 'path': None, 'sha256': None,
}


# ===========================================================================
# hashing / metric helpers (torch is imported lazily, never at module scope)
# ===========================================================================
def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def materialise(value):
    """Normal (non-inference) CPU contiguous clone, safe for hashing/saving."""
    import torch

    with torch.inference_mode(False):
        return value.detach().to('cpu').contiguous().clone()


def bytes_sha(cpu_contiguous):
    import torch

    return hashlib.sha256(memoryview(cpu_contiguous.view(torch.uint8).numpy())).hexdigest()


def raw_tensor_sha(value):
    return bytes_sha(materialise(value))


def _internal_tensor_info(value):
    import torch

    cpu = value.detach().cpu().contiguous()
    entry = {
        'shape': list(cpu.shape),
        'dtype': str(cpu.dtype),
        'bytes': cpu.numel() * cpu.element_size(),
        'sha256': hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest(),
        'finite': bool(torch.isfinite(cpu.float()).all()) if cpu.is_floating_point() else True,
    }
    if cpu.is_floating_point() and cpu.numel():
        entry['max_abs'] = float(cpu.float().abs().max())
    return entry


def _internal_compare(value, reference):
    """Byte-for-byte the staged full_row_fp32_fixed84.compare semantics."""
    import torch

    a = value.detach().cpu().contiguous()
    b = reference.detach().cpu().contiguous()
    if a.shape != b.shape and a.numel() == b.numel():
        b = b.reshape(a.shape)
    result = {'npu': _internal_tensor_info(a), 'cpu': _internal_tensor_info(b)}
    if a.shape != b.shape:
        result.update({'shape_mismatch': True, 'comparable': False, 'raw_bits_equal': False})
        return result
    finite = bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all())
    result.update({'shape_mismatch': False, 'comparable': True, 'finite': finite,
                   'dtype_matches': a.dtype == b.dtype})
    if not finite or a.dtype != b.dtype:
        result['raw_bits_equal'] = False
        return result
    delta = a.double() - b.double()
    denominator = b.double().square().mean().sqrt().clamp_min(1e-30)
    result.update({'raw_bits_equal': bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8))),
                   'elements': a.numel(),
                   'max_abs_error': float(delta.abs().max()),
                   'RMSE': float(delta.square().mean().sqrt()),
                   'relative_rms': float(delta.square().mean().sqrt() / denominator)})
    return result


def tensor_info(value):
    return HELPERS['tensor_info'](value)


def compare_pair(value, reference):
    """Compare candidate vs baseline; neutral key names, no tolerance."""
    pair = HELPERS['compare'](value, reference)
    return {
        'candidate': pair.get('npu'), 'baseline': pair.get('cpu'),
        'shape_mismatch': pair.get('shape_mismatch'), 'comparable': pair.get('comparable'),
        'dtype_matches': pair.get('dtype_matches'), 'finite': pair.get('finite'),
        'raw_bits_equal': pair.get('raw_bits_equal'), 'elements': pair.get('elements'),
        'max_abs_error': pair.get('max_abs_error'), 'RMSE': pair.get('RMSE'),
        'relative_rms': pair.get('relative_rms'), 'implementation': HELPERS['compare_source'],
    }


def install_helpers(compare=None, tensor_info=None):
    """Use the staged full_row_fp32_fixed84 helper when importable, else fallback.

    A caller-supplied callable always wins for that argument; the two arguments
    are resolved independently so a partially supplied pair can never be
    silently replaced.
    """
    record = {'module': None, 'path': None, 'sha256': None,
              'compare_source': 'internal_identical_formulas',
              'tensor_info_source': 'internal_identical_formulas'}
    chosen_info = tensor_info if tensor_info is not None else _internal_tensor_info
    chosen_compare = compare if compare is not None else _internal_compare
    if tensor_info is not None:
        record['tensor_info_source'] = 'caller_supplied'
    if compare is not None:
        record['compare_source'] = 'caller_supplied'
    if compare is None or tensor_info is None:
        try:
            import full_row_fp32_fixed84 as helper
        except ImportError:
            helper = None
        if helper is not None:
            record.update(module='full_row_fp32_fixed84',
                          path=str(Path(helper.__file__).resolve()),
                          sha256=file_sha(Path(helper.__file__)))
            if tensor_info is None:
                chosen_info = helper.tensor_info
                record['tensor_info_source'] = 'full_row_fp32_fixed84.tensor_info'
            if compare is None:
                chosen_compare = helper.compare
                record['compare_source'] = 'full_row_fp32_fixed84.compare'
    HELPERS.update({'tensor_info': chosen_info, 'compare': chosen_compare})
    HELPERS.update(record)
    return dict(record)


def parameters_sha(module):
    import torch

    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(memoryview(parameter.detach().cpu().contiguous()
                                 .view(torch.uint8).numpy()))
    return digest.hexdigest()


# ===========================================================================
# frozen CPU reference, raw24 and original text-model inputs
# ===========================================================================
def load_r78_receipt(receipt):
    path = R78_ROOT / R78_RECEIPT_NAME
    actual = file_sha(path)
    if actual != R78_RECEIPT_SHA256:
        raise RuntimeError('Round78 CPU reference receipt changed: %s' % actual)
    reference = json.loads(path.read_text())
    gates = {
        'completed': reference.get('completed'),
        'full_multimodal_reference_completed': reference.get('full_multimodal_reference_completed'),
        'repeat_pre_final_norm_bits_equal': reference.get('repeat_pre_final_norm_bits_equal'),
        'repeat_post_final_norm_bits_equal': reference.get('repeat_post_final_norm_bits_equal'),
        'pre_final_norm_matches_layer49_identity': reference.get(
            'pre_final_norm_matches_layer49_identity'),
    }
    if not all(value is True for value in gates.values()):
        raise RuntimeError('Round78 CPU FP16 gold is not complete/repeat-verified: '
                           + json.dumps(gates))
    if reference['versions'] != EXPECTED_VERSIONS:
        raise RuntimeError('Round78 gold versions %r are not the pinned %r'
                           % (reference['versions'], EXPECTED_VERSIONS))
    if reference['HF_modeling_source_sha256'] != HF_SOURCE_SHA256:
        raise RuntimeError('Round78 gold was not built from the pinned HF source')
    if reference['CPU_num_threads'] != CPU_THREADS or reference['CPU_mkldnn_enabled'] is not True:
        raise RuntimeError('Round78 gold CPU execution settings changed')
    if reference['original_language_layer_count'] != EXPECTED_ORIGINAL_NUM_HIDDEN_LAYERS:
        raise RuntimeError('original checkpoint layer count changed')
    if reference['original_selected_parameter_sha256'] != R78_ORIGINAL_SELECTED_PARAMETER_SHA256:
        raise RuntimeError('Round78 gold official parameter SHA changed')
    layers = reference['language_layers']
    if len(layers) != SELECTED_LANGUAGE_LAYERS:
        raise RuntimeError('Round78 gold language-layer inventory is not 50')
    for index, expected in ((24, R78_RAW24_TENSOR_SHA256), (25, R78_RAW25_TENSOR_SHA256)):
        if layers[index]['output']['sha256'] != expected:
            raise RuntimeError('Round78 gold raw layer %d SHA changed' % index)
    conversion = reference['parameter_conversion']
    prefix = 'language_model.layers.%d.' % LAYER_INDEX
    present = sorted(name for name in conversion if name.startswith(prefix))
    if present != [prefix + suffix for suffix in LAYER_PARAMETER_SUFFIXES]:
        raise RuntimeError('Round78 layer-25 conversion inventory changed: %r' % present)
    hf_path = Path(reference['HF_modeling_source_path'])
    if not hf_path.is_file() or file_sha(hf_path) != HF_SOURCE_SHA256:
        raise RuntimeError('pinned original HF source is missing or changed: %s' % hf_path)
    receipt['cpu_reference_receipt'] = {
        'path': str(path), 'sha256': actual, 'gates': gates,
        'versions': reference['versions'], 'CPU_num_threads': reference['CPU_num_threads'],
        'CPU_mkldnn_enabled': reference['CPU_mkldnn_enabled'],
        'boundary': reference['comparison_boundary'],
        'raw_layer24_SHA256': layers[24]['output']['sha256'],
        'raw_layer25_SHA256': layers[25]['output']['sha256'],
        'original_selected_parameter_sha256': reference['original_selected_parameter_sha256'],
        'layer25_conversion_entries': len(present),
        'not_modified_by_this_round': True,
    }
    return reference


def load_cpu_raw24(receipt):
    import torch

    path = R78_ROOT / R78_RAW24_NAME
    actual = file_sha(path)
    if actual != R78_RAW24_FILE_SHA256:
        raise RuntimeError('Round78 CPU raw layer-24 payload changed: %s' % actual)
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not torch.is_tensor(value):
        raise RuntimeError('Round78 raw layer-24 payload is not a tensor')
    cpu = materialise(value)
    if tuple(cpu.shape) != (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN):
        raise RuntimeError('unexpected CPU raw24 shape %r' % (tuple(cpu.shape),))
    if cpu.dtype != torch.float16:
        raise RuntimeError('unexpected CPU raw24 dtype %s' % cpu.dtype)
    if not bool(torch.isfinite(cpu.float()).all()):
        raise RuntimeError('CPU raw24 contains non-finite values')
    info = tensor_info(cpu)
    if info['sha256'] != R78_RAW24_TENSOR_SHA256:
        raise RuntimeError('CPU raw24 tensor SHA changed: %s' % info['sha256'])
    receipt['cpu_raw24'] = {'path': str(path), 'file_sha256': actual,
                            'file_bytes': path.stat().st_size, 'tensor': info,
                            'boundary': 'Round78 language_layers[24] rawdecoderPREDeepStackinjection'}
    return cpu


def load_original_text_inputs(receipt):
    """Frozen Round78 original text-model inputs: exact M-RoPE rows + embeddings.

    ``FP16_multimodal_inputs.pt`` holds exactly what the original
    ``Qwen3VLTextModel.forward`` received, so the causal mask and the rotary
    tensors are rebuilt from the same tensors the gold itself used.
    """
    import torch

    path = R78_ROOT / R78_MM_INPUTS_NAME
    actual = file_sha(path)
    if actual != R78_MM_INPUTS_FILE_SHA256:
        raise RuntimeError('Round78 multimodal input payload changed: %s' % actual)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    for name in ('inputs_embeds', 'position_ids', 'visual_pos_masks',
                 'deepstack_visual_embeds'):
        if name not in payload:
            raise RuntimeError('Round78 multimodal payload lost %r' % name)
    embeds = materialise(payload['inputs_embeds'])
    positions = materialise(payload['position_ids'])
    if tuple(embeds.shape) != (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN):
        raise RuntimeError('unexpected original text inputs_embeds shape %r' % (tuple(embeds.shape),))
    if embeds.dtype != torch.float16:
        raise RuntimeError('unexpected original text inputs_embeds dtype %s' % embeds.dtype)
    embeds_info, positions_info = tensor_info(embeds), tensor_info(positions)
    if embeds_info['sha256'] != R78_MM_INPUTS_EMBEDS_SHA256:
        raise RuntimeError('original text inputs_embeds SHA changed')
    if positions_info['sha256'] != R78_POSITION_IDS_SHA256:
        raise RuntimeError('original M-RoPE position_ids SHA changed')
    receipt['original_text_inputs'] = {
        'path': str(path), 'file_sha256': actual, 'file_bytes': path.stat().st_size,
        'inputs_embeds': embeds_info, 'position_ids': positions_info,
        'note': 'exact tensors the original HF text model used; mask/rotary rebuilt from these',
    }
    return embeds, positions


# ===========================================================================
# original HF layer 25 and its exact input geometry
# ===========================================================================
def build_layer(model_dir, reference, layer_index):
    """Rebuild ONLY the original HF layer-25 module from original BF16 storage."""
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf

    for library, version in (('torch', torch.__version__),
                             ('transformers', transformers.__version__)):
        if version != EXPECTED_VERSIONS[library]:
            raise RuntimeError('%s %s is not the pinned %s'
                               % (library, version, EXPECTED_VERSIONS[library]))
    hf_source = Path(hf.__file__)
    if file_sha(hf_source) != HF_SOURCE_SHA256:
        raise RuntimeError('imported HF modeling source is not the pinned original')
    if file_sha(model_dir / 'config.json') != CONFIG_SHA256:
        raise RuntimeError('checkpoint config changed')
    if file_sha(model_dir / 'model.safetensors.index.json') != CHECKPOINT_INDEX_SHA256:
        raise RuntimeError('checkpoint index changed')

    config = Qwen3VLConfig.from_pretrained(str(model_dir), trust_remote_code=False,
                                           local_files_only=True)
    text_config = config.text_config
    original_layers = int(text_config.num_hidden_layers)
    if original_layers != EXPECTED_ORIGINAL_NUM_HIDDEN_LAYERS:
        raise RuntimeError('unexpected original layer count %d' % original_layers)
    text_config.num_hidden_layers = SELECTED_LANGUAGE_LAYERS
    text_config.use_cache = False
    config.dtype = torch.float16
    text_config.dtype = torch.float16
    config.vision_config.dtype = torch.float16
    config._attn_implementation = 'sdpa'
    config.vision_config._attn_implementation = 'sdpa'
    text_config._attn_implementation = 'sdpa'

    with torch.device('meta'):
        layer = hf.Qwen3VLTextDecoderLayer(text_config, layer_index)
    wanted = dict(layer.named_parameters())
    prefix = 'language_model.layers.%d.' % layer_index
    expected = {prefix + suffix: suffix for suffix in LAYER_PARAMETER_SUFFIXES}
    if set(wanted) != set(expected.values()):
        raise RuntimeError('layer parameter inventory changed: %r' % sorted(wanted))

    weight_map = json.loads((model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    conversion = reference['parameter_conversion']
    shards = {}
    for name in sorted(expected):
        key = 'model.' + name
        if key not in weight_map:
            raise RuntimeError('checkpoint lost %r' % key)
        shards.setdefault(weight_map[key], []).append((name, key))
    state, verified = {}, {}
    for shard, entries in sorted(shards.items()):
        with safe_open(str(model_dir / shard), framework='pt', device='cpu') as handle:
            for name, key in entries:
                original = handle.get_tensor(key)
                if original.dtype != torch.bfloat16:
                    raise RuntimeError('%s is not the original BF16 storage' % key)
                converted = original.to(torch.float16)
                if not bool(torch.isfinite(converted).all()):
                    raise RuntimeError('%s is not finite as FP16' % key)
                entry = conversion[name]
                if list(converted.shape) != entry['shape']:
                    raise RuntimeError('%s shape differs from the Round78 conversion' % name)
                sha = raw_tensor_sha(converted)
                if sha != entry['reference_sha256']:
                    raise RuntimeError('%s FP16 conversion SHA differs from Round78' % name)
                state[expected[name]] = converted
                verified[name] = {'shape': entry['shape'], 'reference_sha256': sha,
                                  'original_dtype': str(original.dtype)}
    if len(state) != len(LAYER_PARAMETER_SUFFIXES):
        raise RuntimeError('layer-25 weight loading is incomplete')
    layer.load_state_dict(state, strict=True, assign=True)
    layer.eval()
    parameters = dict(layer.named_parameters())
    if len(parameters) != len(LAYER_PARAMETER_SUFFIXES):
        raise RuntimeError('unexpected loaded parameter count %d' % len(parameters))
    for name, parameter in parameters.items():
        if parameter.dtype != torch.float16 or parameter.is_meta or parameter.device.type != 'cpu':
            raise RuntimeError('parameter %s is not a real CPU FP16 tensor' % name)
    if any(buffer.is_meta or buffer.device.type != 'cpu' for buffer in layer.buffers()):
        raise RuntimeError('layer buffers are not real CPU tensors')

    info = {
        'original_HF_source': {'path': str(hf_source), 'sha256': HF_SOURCE_SHA256,
                               'matches_pin': True},
        'versions': {'torch': torch.__version__,
                     'transformers': transformers.__version__},
        'config_sha256': CONFIG_SHA256, 'checkpoint_index_sha256': CHECKPOINT_INDEX_SHA256,
        'original_num_hidden_layers': original_layers,
        'selected_language_layers': SELECTED_LANGUAGE_LAYERS, 'layer_index': layer_index,
        'attention_implementation': text_config._attn_implementation,
        'sdpa_backend': 'SDPBackend.MATH',
        'parameter_names': sorted(parameters),
        'parameter_count': len(parameters),
        'parameter_dtypes': sorted({str(p.dtype) for p in parameters.values()}),
        'loaded_checkpoint_keys': sorted(expected.values()),
        'only_layer_25_weights_loaded': True,
        'conversion_reference_sha256_verified': verified,
        'load_state_dict': 'strict=True assign=True from the original dense BF16 storage',
    }
    return layer, hf, text_config, info


def split_original_positions(position_ids, receipt):
    """Mirror the original Qwen3VLTextModel.forward M-RoPE row split exactly."""
    import torch

    if not torch.is_tensor(position_ids) or position_ids.dtype != torch.int64:
        raise RuntimeError('original position_ids must be an int64 tensor')
    if position_ids.ndim == 2:
        expanded = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        text_positions, rotary_positions = expanded[0], expanded[1:]
        branch = 'ndim2_expand4_text_row0_rotary_rows1to3'
    elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_positions, rotary_positions = position_ids[0], position_ids[1:]
        branch = 'ndim3_rows4_text_row0_rotary_rows1to3'
    elif position_ids.ndim == 3:
        text_positions, rotary_positions = None, position_ids
        branch = 'ndim3_rows%d_text_none_rotary_all_rows' % position_ids.shape[0]
    else:
        raise RuntimeError('unsupported original position_ids ndim %d' % position_ids.ndim)
    if rotary_positions.shape[0] != 3:
        raise RuntimeError('expected 3 M-RoPE rotary rows, got %d' % rotary_positions.shape[0])
    if tuple(rotary_positions.shape[1:]) != (EXPECTED_BATCH, EXPECTED_SEQ):
        raise RuntimeError('unexpected rotary position shape %r'
                           % (tuple(rotary_positions.shape),))
    receipt['positions'] = {
        'raw': tensor_info(position_ids), 'branch': branch,
        'text_position_ids_present': text_positions is not None, 'rotary_rows': 3,
        'mirrors_original_Qwen3VLTextModel_forward': True,
    }
    return text_positions, rotary_positions


def build_rotary(hf, text_config, rotary_positions, hidden_states, receipt):
    """Original HF Qwen3VLTextRotaryEmbedding on CPU Float32 inverse frequencies."""
    import torch

    rotary = hf.Qwen3VLTextRotaryEmbedding(text_config, device='cpu')
    cos, sin = rotary(hidden_states, rotary_positions)
    cos, sin = materialise(cos), materialise(sin)
    if cos.dtype != torch.float16 or sin.dtype != torch.float16:
        raise RuntimeError('original rotary did not return FP16 cos/sin')
    receipt['rotary'] = {
        'module': 'Qwen3VLTextRotaryEmbedding(text_config, device=cpu)',
        'cos': tensor_info(cos), 'sin': tensor_info(sin),
        'inv_freq': tensor_info(rotary.inv_freq),
        'mrope_section': [int(value) for value in rotary.mrope_section],
        'attention_scaling': float(rotary.attention_scaling),
        'learned_parameters': sum(p.numel() for p in rotary.parameters()),
        'same_rotary_for_every_case': True,
    }
    return cos, sin


def build_causal_mask(hf, text_config, mask_input, text_positions, receipt):
    """Original create_causal_mask call, same five kwargs as the original forward."""
    import torch

    attention_mask = torch.ones((EXPECTED_BATCH, EXPECTED_SEQ), dtype=torch.int64)
    mask = hf.create_causal_mask(config=text_config, inputs_embeds=mask_input,
                                 attention_mask=attention_mask, past_key_values=None,
                                 position_ids=text_positions)
    if mask is not None and not torch.is_tensor(mask):
        raise RuntimeError('original create_causal_mask returned an unsupported mask type')
    # Original SDPA may elide an all-ones causal mask and rely on is_causal=True.
    # Keep that original representation; the exact R78 output SHA gate still applies.
    receipt['causal_mask'] = {
        'construction': 'original Qwen3VLTextModel.forward call (pinned HF source)',
        'kwargs': ['config', 'inputs_embeds', 'attention_mask', 'past_key_values=None',
                   'position_ids'],
        'attention_mask': tensor_info(attention_mask),
        'inputs_embeds_source': 'frozen Round78 original text inputs_embeds',
        'inputs_embeds': tensor_info(mask_input),
        'mask': None if mask is None else tensor_info(mask),
        'mask_representation': 'original_sdpa_implicit_causal_None' if mask is None else 'original_tensor',
        'same_mask_for_every_case': True,
        'no_plus_minus_50000_clamp': True,
    }
    return mask


def run_block25(layer, hidden_states, mask, text_positions, position_embeddings):
    """One original HF decoder-layer-25 forward under SDPBackend.MATH."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    assert layer.self_attn.is_causal is True
    with torch.inference_mode(), sdpa_kernel(backends=[SDPBackend.MATH]):
        output = layer(hidden_states, attention_mask=mask, position_ids=text_positions,
                       past_key_values=None, position_embeddings=position_embeddings)
    if not torch.is_tensor(output):
        raise RuntimeError('original HF decoder layer did not return a tensor')
    output = materialise(output)
    if output.dtype != torch.float16:
        raise RuntimeError('original HF decoder layer output is not FP16')
    if not bool(torch.isfinite(output.float()).all()):
        raise RuntimeError('original HF decoder layer output is not finite')
    return output


def evaluate_twice(layer, hidden_states, mask, text_positions, position_embeddings):
    first = run_block25(layer, hidden_states, mask, text_positions, position_embeddings)
    second = run_block25(layer, hidden_states, mask, text_positions, position_embeddings)
    if bytes_sha(first) != bytes_sha(second):
        raise RuntimeError('original HF CPU block-25 evaluation is not bit-repeatable')
    return first, True


def load_inputs_bundle(path, receipt):
    """Validate the root's actual-NPU raw-layer-24 capture bundle, fail closed."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise RuntimeError('missing NPU raw24 capture bundle: %s' % path)
    actual = file_sha(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError('capture bundle is not a mapping')
    for name in ('inputs', 'position_ids', 'source_raw24_SHAs', 'captured_source_SHA256'):
        if name not in payload:
            raise RuntimeError('capture bundle lost %r' % name)
    if not isinstance(payload['inputs'], dict) or not payload['inputs']:
        raise RuntimeError('capture bundle carries no case inputs')
    unknown = sorted(set(payload['inputs']) - set(ALLOWED_CASES))
    if unknown:
        raise RuntimeError('capture bundle has cases outside the Round91 receipt: %r' % unknown)
    declared = payload['source_raw24_SHAs']
    if not isinstance(declared, dict) or set(declared) != set(payload['inputs']):
        raise RuntimeError('declared source raw24 SHAs do not match the case inputs')
    if not isinstance(payload['captured_source_SHA256'], str):
        raise RuntimeError('captured_source_SHA256 must be a string')

    inputs, cases = {}, {}
    for case in sorted(payload['inputs']):
        hidden = materialise(payload['inputs'][case])
        if tuple(hidden.shape) != (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN):
            raise RuntimeError('%s raw24 shape is %r' % (case, tuple(hidden.shape)))
        if hidden.dtype != torch.float16:
            raise RuntimeError('%s raw24 dtype is %s' % (case, hidden.dtype))
        info = tensor_info(hidden)
        if info['finite'] is not True:
            raise RuntimeError('%s raw24 is not finite' % case)
        if info['sha256'] != declared[case]:
            raise RuntimeError('%s raw24 SHA differs from the declared capture value' % case)
        if case in R91_RAW24_SHA256 and info['sha256'] != R91_RAW24_SHA256[case]:
            raise RuntimeError('%s raw24 SHA differs from the Round91 actual receipt' % case)
        inputs[case] = hidden
        cases[case] = {
            'input': info, 'declared_source_raw24_SHA256': declared[case],
            'matches_R91_actual_receipt': (case in R91_RAW24_SHA256
                                           and info['sha256'] == R91_RAW24_SHA256[case]),
            'case_allowed_by_R91_receipt': case in R91_RAW24_SHA256,
        }
    positions = materialise(payload['position_ids'])
    if tensor_info(positions)['sha256'] != R78_POSITION_IDS_SHA256:
        raise RuntimeError('capture bundle positions are not the original Round78 M-RoPE rows')
    receipt['inputs_source'] = {
        'path': str(path), 'file_sha256': actual, 'file_bytes': path.stat().st_size,
        'captured_source_SHA256': payload['captured_source_SHA256'],
        'cases': cases, 'position_ids': tensor_info(positions),
        'positions_match_original_R78_rows': True,
        'input_tensors_are_actual_NPU_raw24_outputs_not_CPU_conditioning': True,
        'input_keys_are_raw_layer24_outputs_not_precomputed_conditioning': True,
    }
    return inputs, positions, declared


def cross_check_root_expected_pins(root, declared, receipt):
    """Optional: the root's own expected_raw24_92.json must agree with us.

    The captured bundle declares per-case raw layer-24 SHA-256 values.  When the
    root also leaves its own expected-pins file next to the bundle, this module
    requires the two to agree with each other *and* with the pinned Round91
    actual receipt; otherwise it refuses.  The file is optional only because it
    is a redundant gate, never because disagreement is tolerated.
    """
    path = Path(root) / EXPECTED_PINS_NAME
    if not path.is_file():
        receipt['expected_pins_cross_check'] = {'present': False, 'path': str(path),
                                                'effect': 'redundant_gate_skipped'}
        return
    payload = json.loads(path.read_text())
    cases = payload.get('cases')
    if not isinstance(cases, dict) or not cases:
        raise RuntimeError('root expected-pins file has no cases: %s' % path)
    unknown = sorted(set(cases) - set(ALLOWED_CASES))
    if unknown:
        raise RuntimeError('root expected-pins file has cases outside the Round91 '
                           'receipt: %r' % unknown)
    for case, entry in cases.items():
        sha = entry.get('sha256') if isinstance(entry, dict) else None
        if sha != R91_RAW24_SHA256[case]:
            raise RuntimeError('root expected pin for %s disagrees with the Round91 actual '
                               'receipt: %r' % (case, sha))
    missing = sorted(set(declared) - set(cases))
    if missing:
        raise RuntimeError('capture bundle declares cases the root expected-pins file does '
                           'not cover: %r' % missing)
    for case in sorted(declared):
        if declared[case] != cases[case]['sha256']:
            raise RuntimeError('capture bundle declares a different %s raw24 SHA than the '
                               'root expected-pins file' % case)
    receipt['expected_pins_cross_check'] = {
        'present': True, 'path': str(path), 'file_sha256': file_sha(path),
        'cases': sorted(cases), 'R91source_SHA256': payload.get('R91source_SHA256'),
        'agrees_with_embedded_R91_pins': True,
        'bundle_declared_SHA256_matches_root_expected': True,
        'case_sets_equal': set(cases) == set(declared),
    }


# ===========================================================================
# entry point
# ===========================================================================
def run_reference(root=ROOT_DEFAULT, inputs_path=None, compare=None, tensor_info=None):
    """Write CPU_block25_reference92.json and CPU_block25_outputs92.pt; no overwrite."""
    import torch

    global STAGE
    STAGE = 'start'
    root = Path(root)
    inputs_path = Path(inputs_path) if inputs_path is not None else root / INPUTS_NAME
    reference_path = root / REFERENCE_NAME
    outputs_path = root / OUTPUTS_NAME
    failure_path = root / FAILURE_NAME
    receipt = {
        'scope': __doc__,
        'kind': 'original-HF CPU FP16 decoder-layer-25 same-input reference',
        'source_sha256': file_sha(Path(__file__)),
        'root': str(root),
        'layer_index': LAYER_INDEX,
        'selected_language_layers': SELECTED_LANGUAGE_LAYERS,
        'boundary': 'original HF raw decoder layer 25 output, PRE DeepStack injection, '
                    'PRE final norm',
        'no_final_norm_after_layer25': True,
        'no_DeepStack_after_raw24': True,
        'no_Float_linear_weights_or_teacher': True,
        'no_BF16_gold_output_cast': True,
        'numerical_tolerance': None,
        'full_encoder_precision_accepted': False,
        'whole_goal_complete': False,
        'not_a_speed_measurement': True,
        'does_not_modify_R78_gold': True,
        'only_layer25_parameters_loaded': True,
        'cpu_baseline_is_frozen_R78_raw24_not_recomputed': True,
        'gate_order': 'R78 receipt gates -> CPU raw24 -> original text inputs -> layer '
                      'weights -> reproduce R78 raw25 -> then validate NPU inputs',
        'completed': False,
    }
    try:
        if not root.is_dir():
            raise RuntimeError('missing root directory: %s' % root)
        if reference_path.exists() or outputs_path.exists():
            raise RuntimeError('refusing to overwrite existing R92 outputs in %s' % root)
        STAGE = 'metric_helpers'
        receipt['helper_metrics'] = install_helpers(compare, tensor_info)
        torch.set_num_threads(CPU_THREADS)
        receipt['cpu_execution'] = {
            'torch': torch.__version__, 'threads': torch.get_num_threads(),
            'mkldnn_enabled': bool(torch.backends.mkldnn.enabled),
            'works_on_cpu_only': True,
            'accelerator_present_but_unused': bool(
                torch.cuda.is_available() or hasattr(torch, 'npu')),
        }
        if torch.get_num_threads() != CPU_THREADS or not torch.backends.mkldnn.enabled:
            raise RuntimeError('could not reproduce the Round78 CPU execution settings')

        STAGE = 'r78_receipt'
        reference = load_r78_receipt(receipt)
        STAGE = 'cpu_raw24'
        cpu_raw24 = load_cpu_raw24(receipt)
        STAGE = 'original_text_inputs'
        original_embeds, original_positions = load_original_text_inputs(receipt)
        STAGE = 'original_layer'
        layer, hf, text_config, layer_info = build_layer(MODEL, reference, LAYER_INDEX)
        receipt['original_layer'] = layer_info
        STAGE = 'positions_rotary_mask'
        text_positions, rotary_positions = split_original_positions(original_positions, receipt)
        position_embeddings = build_rotary(hf, text_config, rotary_positions, original_embeds,
                                           receipt)
        mask = build_causal_mask(hf, text_config, original_embeds, text_positions, receipt)
        parameters_before = parameters_sha(layer)

        STAGE = 'reference_gate'
        baseline, baseline_repeat = evaluate_twice(layer, cpu_raw24, mask, text_positions,
                                                   position_embeddings)
        baseline_sha = bytes_sha(baseline)
        receipt['reference_gate'] = {
            'input': 'CPU_raw24: frozen Round78 raw layer-24 tensor',
            'repeat_bits_equal': baseline_repeat,
            'output': tensor_info(baseline),
            'R78_raw_layer25_SHA256': R78_RAW25_TENSOR_SHA256,
            'matches_R78_raw25_SHA256': baseline_sha == R78_RAW25_TENSOR_SHA256,
            'validated_before_any_NPU_input_was_read': True,
            'no_tolerance_inputs': True,
        }
        if baseline_sha != R78_RAW25_TENSOR_SHA256:
            raise RuntimeError('CPU layer 25 failed to reproduce the frozen Round78 raw '
                               'layer-25 output (%s != %s); refusing to evaluate NPU inputs'
                               % (baseline_sha, R78_RAW25_TENSOR_SHA256))

        STAGE = 'npu_inputs'
        inputs, bundle_positions, declared = load_inputs_bundle(inputs_path, receipt)
        cross_check_root_expected_pins(root, declared, receipt)
        if bytes_sha(bundle_positions) != bytes_sha(original_positions):
            raise RuntimeError('capture bundle positions differ from the frozen original rows')
        receipt['inputs_source']['positions_bit_equal_to_frozen_original'] = True

        outputs = {'CPU_raw24': baseline}
        input_shas = {'CPU_raw24': R78_RAW24_TENSOR_SHA256}
        cases = {'CPU_raw24': {'is_baseline': True,
                               'input': receipt['cpu_raw24']['tensor'],
                               'repeat_bits_equal': baseline_repeat,
                               'output': tensor_info(baseline),
                               'matches_R78_raw25_SHA256': True}}
        amplification = {}
        for case in sorted(inputs):
            STAGE = 'case:' + case
            output, repeat = evaluate_twice(layer, inputs[case], mask, text_positions,
                                            position_embeddings)
            outputs[case] = output
            input_shas[case] = bytes_sha(inputs[case])
            versus = compare_pair(output, baseline)
            input_versus = compare_pair(inputs[case], cpu_raw24)
            cases[case] = {
                'is_baseline': False,
                'input': receipt['inputs_source']['cases'][case]['input'],
                'input_matches_R91_actual_receipt':
                    receipt['inputs_source']['cases'][case]['matches_R91_actual_receipt'],
                'repeat_bits_equal': repeat,
                'output': tensor_info(output),
                'vs_CPU_raw24_block25': versus,
                'input_vs_CPU_raw24_input': input_versus,
            }
            input_rms = input_versus['relative_rms']
            output_rms = versus['relative_rms']
            amplification[case] = {
                'input_relative_rms_vs_CPU_raw24': input_rms,
                'block25_output_relative_rms_vs_CPU_raw24': output_rms,
                'output_over_input_amplification': (None if not input_rms
                                                    else output_rms / input_rms),
            }
            receipt['cases'] = cases
            receipt['input_error_amplification'] = amplification
        receipt['cases'] = cases
        receipt['input_error_amplification'] = amplification
        receipt['isolation_semantics'] = (
            'Original CPU FP16 block 25 evaluated on an actual NPU raw layer-24 state is an '
            'input-only sensitivity control: it separates input drift from per-block NPU '
            'arithmetic. It is not a Float-weight teacher, casts no BF16 gold output, adds no '
            'final norm and no DeepStack, and selects no tolerance. It alone cannot show a '
            'unique hardware operator bug nor a lossless pass.')

        STAGE = 'invariants'
        receipt['frozen_after'] = {
            'parameters_sha256': parameters_sha(layer),
            'cpu_raw24_tensor_sha256': raw_tensor_sha(cpu_raw24),
            'r78_receipt_sha256': file_sha(R78_ROOT / R78_RECEIPT_NAME),
            'r78_multimodal_inputs_sha256': file_sha(R78_ROOT / R78_MM_INPUTS_NAME),
            'inputs_file_sha256': file_sha(inputs_path),
            'source_sha256': file_sha(Path(__file__)),
        }
        receipt['frozen_after'].update({
            'parameters_unchanged':
                receipt['frozen_after']['parameters_sha256'] == parameters_before,
            'cpu_raw24_unchanged':
                receipt['frozen_after']['cpu_raw24_tensor_sha256'] == R78_RAW24_TENSOR_SHA256,
            'r78_receipt_unchanged':
                receipt['frozen_after']['r78_receipt_sha256'] == R78_RECEIPT_SHA256,
            'r78_multimodal_inputs_unchanged':
                receipt['frozen_after']['r78_multimodal_inputs_sha256']
                == R78_MM_INPUTS_FILE_SHA256,
            'inputs_file_unchanged':
                receipt['frozen_after']['inputs_file_sha256']
                == receipt['inputs_source']['file_sha256'],
            'source_unchanged':
                receipt['frozen_after']['source_sha256'] == receipt['source_sha256'],
        })
        if not all(value for key, value in receipt['frozen_after'].items()
                   if key.endswith('_unchanged')):
            raise RuntimeError('a frozen input, weight or reference changed during the run')

        STAGE = 'write_outputs'
        temporary_outputs = outputs_path.with_name(outputs_path.name + '.tmp')
        temporary_reference = reference_path.with_name(reference_path.name + '.tmp')
        torch.save({'outputs': outputs, 'input_SHAs': input_shas,
                    'position_ids': original_positions}, temporary_outputs)
        outputs_sha = file_sha(temporary_outputs)
        receipt['outputs_file'] = {
            'path': str(outputs_path), 'bytes': temporary_outputs.stat().st_size,
            'sha256': outputs_sha, 'tensor_keys': sorted(outputs),
            'input_SHAs': dict(input_shas), 'position_ids': tensor_info(original_positions),
        }
        receipt['outputs_file_SHA256'] = outputs_sha
        receipt['inputs_source_path'] = str(inputs_path)
        receipt['inputs_file_SHA256'] = receipt['inputs_source']['file_sha256']
        receipt['completed'] = True
        temporary_reference.write_text(json.dumps(receipt, indent=2) + '\n')
        if reference_path.exists() or outputs_path.exists():
            raise RuntimeError('refusing to overwrite outputs created concurrently')
        os.replace(temporary_outputs, outputs_path)
        os.replace(temporary_reference, reference_path)
        return receipt
    except BaseException as error:
        failure = {
            'scope': 'Round92 original CPU layer-25 reference failure record',
            'stage': STAGE, 'completed': False,
            'error': {'type': type(error).__name__, 'message': str(error)},
            'traceback': traceback.format_exc(),
            'reference_written': reference_path.exists(),
            'outputs_written': outputs_path.exists(),
            'source_sha256': file_sha(Path(__file__)),
        }
        try:
            with failure_path.open('x') as handle:
                handle.write(json.dumps(failure, indent=2) + '\n')
        except FileExistsError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Original HF CPU FP16 decoder-layer-25 same-input reference (Round92)')
    parser.add_argument('--root', default=str(ROOT_DEFAULT))
    parser.add_argument('--inputs', default=None)
    arguments = parser.parse_args(argv)
    root = Path(arguments.root)
    receipt = run_reference(root=root,
                            inputs_path=Path(arguments.inputs) if arguments.inputs else None)
    print(json.dumps({
        'reference': str(root / REFERENCE_NAME), 'outputs': str(root / OUTPUTS_NAME),
        'completed': receipt['completed'],
        'reference_gate_matches_R78_raw25':
            receipt['reference_gate']['matches_R78_raw25_SHA256'],
        'cases': sorted(receipt['cases']),
        'outputs_file_SHA256': receipt['outputs_file_SHA256'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())