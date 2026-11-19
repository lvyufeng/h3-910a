#!/usr/bin/env python3
"""Make explicit FP16 dtype-only adaptation inside the isolated CUDA checkout.

The stock H3 encoder, DiT and transport route hard-code BF16. These three literal
replacements are required for a real FP16 trial; --dtype half alone does not do it.
Do not change attention math, weight values, the schedule, VAE, image conditioning,
normalization, or any accelerator-wide settings. Raw official BF16 checkpoint bytes
remain untouched and are converted only by the actual parameter loaders.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/mseco/h3_gpu12201')
SOURCE = ROOT / 'vllm-omni-fl2va-release'
PIN = '1c2a81f6d84aea4fff53bd2f894c2a287c237245'
PACKAGE = SOURCE / 'vllm_omni/diffusion/models/minimax_h3'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = ROOT / 'fp16_source_adaptation122b.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old['completed']
        for name, row in old['files'].items():
            assert sha((PACKAGE / name).read_bytes()) == row['adapted_sha256']
        print('Existing isolated dtype-only adaptation re-verified; no edit replay.')
        return
    pin = json.loads((ROOT / 'omni_final_checkout_pin122.json').read_text())
    assert pin['commit'] == PIN and pin['source_root'] == str(SOURCE)
    import urllib.request
    with urllib.request.urlopen(f'https://api.github.com/repos/vllm-project/vllm-omni/git/trees/{PIN}?recursive=1', timeout=60) as response:
        tree = json.load(response)
    assert not tree.get('truncated')
    blobs = {row['path']: row['sha'] for row in tree['tree'] if row['type'] == 'blob'}
    result = {'scope': __doc__, 'completed': False, 'commit': PIN, 'source_archive_sha256': pin['archive_sha256'], 'original_sources_verified_against_official_Git_blob_SHA1': True,
              'official_checkpoint_modified': False, 'global_environment_modified': False,
              'FP32_exceptions_retained': ['time/patch/final-output projections', 'video/audio VAE original FP32 parameters',
                                          'normalization/softmax/rotary accumulation where defined by upstream'],
              'files': {}}
    backup = ROOT / 'original_cuda_sources122b'
    backup.mkdir(exist_ok=False)
    for name in ['encoder.py', 'minimax_h3_transformer.py', 'pipeline_minimax_h3.py']:
        path = PACKAGE / name
        relative = path.relative_to(SOURCE)
        original = path.read_bytes()
        git_blob = hashlib.sha1(f'blob {len(original)}\0'.encode() + original).hexdigest()
        assert git_blob == blobs[str(relative)], f'Refusing unverified upstream source {name}'
        text = original.decode('utf-8')
        count = text.count('torch.bfloat16')
        assert count > 0, name
        adapted = text.replace('torch.bfloat16', 'torch.float16').encode('utf-8')
        assert b'torch.bfloat16' not in adapted
        with (backup / name).open('xb') as f:
            f.write(original)
        path.write_bytes(adapted)
        compile(adapted, str(path), 'exec')
        result['files'][name] = {'original_sha256': sha(original), 'adapted_sha256': sha(adapted),
                                 'replacement': 'torch.bfloat16 -> torch.float16', 'occurrences': count}
    result['completed'] = True
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    import difflib
    diff = ''.join(''.join(difflib.unified_diff((backup / name).read_text().splitlines(keepends=True), (PACKAGE / name).read_text().splitlines(keepends=True), fromfile='original/' + name, tofile='fp16/' + name)) for name in result['files'])
    (ROOT / 'fp16_dtype_only122b.patch').write_text(diff)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
