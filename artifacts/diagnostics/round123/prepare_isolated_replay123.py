#!/usr/bin/env python3
"""Prepare an isolated, source-pinned H3 transformer for matched GPU/NPU replay.

Copy the exact successful GPU transformer source plus the pure sampler/branch
helpers. On NPU, permit the verified native FP16-input FP32-output torch.mm and use
explicit peak-reduction axes with the same mathematical maximum. Shared installed
vLLM-Omni and model files are untouched. No BF16 execution or weight relay.
"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--backend', choices=['cuda', 'npu'], required=True)
    args = p.parse_args()
    root = Path(args.root)
    package = root / 'h3_replay123'
    assert not package.exists()
    assert not (root / 'isolated_replay_source123.json').exists()
    package.mkdir()
    (package / '__init__.py').write_text('')
    source = root / 'transformer_successful_GPU_source123.py'
    original = source.read_bytes()
    expected = '7cf0c53e6d9199b7944d9026c2bdc1120b1e32084cffe266c434d2d6449b8029'
    assert hashlib.sha256(original).hexdigest() == expected
    text = original.decode()
    changes = []
    if args.backend == 'npu':
        old = 'if x.device.type != "cuda" or layer.weight.dtype != torch.float16:'
        new = 'if x.device.type not in ("cuda", "npu") or layer.weight.dtype != torch.float16:'
        assert text.count(old) == 1
        text = text.replace(old, new)
        changes.append('Permit the verified NPU FP16-input/weight FP32-output torch.mm; no arithmetic edit.')
        old = 'peak = float(product.abs().amax())'
        new = 'peak = float(product.abs().amax(dim=tuple(range(product.ndim))))'
        assert text.count(old) == 1
        text = text.replace(old, new)
        changes.append('Use explicitly named peak-reduction axes to avoid the verified CANN allocator failure; preserve the same exact mathematical maximum.')
        assert hashlib.sha256(text.encode()).hexdigest() == '5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b'
    (package / 'transformer.py').write_text(text)
    helper_names = ['denoise_loop', 'scheduling_minimax_h3_euler_ancestral']
    helpers = {}
    for name in helper_names:
        path = root / f'{name}_GPU_source123.py'
        data = path.read_bytes()
        (package / f'{name}.py').write_bytes(data)
        helpers[name] = hashlib.sha256(data).hexdigest()
    compile(text, str(package / 'transformer.py'), 'exec')
    report = {'scope': __doc__, 'completed': True, 'backend': args.backend,
              'original_successful_GPU_source_sha256': expected,
              'replay_transformer_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'changes': changes, 'helpers_sha256': helpers,
              'shared_vllm_omni_source_modified': False, 'model_weights_modified': False}
    (root / 'isolated_replay_source123.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
