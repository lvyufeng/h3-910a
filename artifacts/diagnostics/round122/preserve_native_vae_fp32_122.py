#!/usr/bin/env python3
"""Preserve upstream's original FP32 VAE weight contract under a Half engine default.

The remote audio-VAE factory constructs nn.Modules and uses load_state_dict, which
copies into the newly constructed parameter dtype. Its construction must occur
under FP32, otherwise final .float() can hide a Half roundtrip. This patch changes
loading only, not VAE inference, weights, input conditioning or output math.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/mseco/h3_gpu12201')
SOURCE = ROOT / 'vllm-omni-fl2va-release/vllm_omni/diffusion/models/minimax_h3/vae.py'

OLD = '''    component_cls = get_class_from_dynamic_module(
        class_reference,
        component_path,
    )
    return component_cls.from_pretrained(component_path)
'''
NEW = '''    # H3's native video/audio VAE weights are intentionally FP32. The
    # engine-wide Half default must not round them during nn.Module factory
    # construction before the wrapper restores the final FP32 dtype.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        component_cls = get_class_from_dynamic_module(
            class_reference,
            component_path,
        )
        return component_cls.from_pretrained(component_path)
    finally:
        torch.set_default_dtype(previous_dtype)
'''


def main():
    receipt = ROOT / 'native_VAE_FP32_load_guard122.json'
    assert not receipt.exists()
    original = SOURCE.read_text()
    assert original.count(OLD) == 1
    backup = ROOT / 'original_cuda_sources122b/vae.py'
    assert not backup.exists()
    backup.write_text(original)
    changed = original.replace(OLD, NEW)
    compile(changed, str(SOURCE), 'exec')
    SOURCE.write_text(changed)
    diff = ''.join(difflib.unified_diff(original.splitlines(keepends=True), changed.splitlines(keepends=True),
                                      fromfile='upstream/vae.py', tofile='FP32_load_guard/vae.py'))
    (ROOT / 'native_VAE_FP32_load_guard122.patch').write_text(diff)
    row = {'scope': __doc__, 'applied': True, 'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
           'guarded_sha256': hashlib.sha256(changed.encode()).hexdigest(),
           'native_weight_contract': 'video/audio VAE parameters remain original FP32 values',
           'inference_math_changed': False, 'checkpoint_bytes_changed': False}
    receipt.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
