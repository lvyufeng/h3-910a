"""Launch helper for R125 candidate full generations on the owned 910A host.

Run this on ts-217 (it is not a model script; it only spawns the pinned R124
driver with a fresh private IPC key, work directory and output label).

Only the environment differs from the R124/FA baseline run, which keeps the
comparison honest: same driver, same sources, same checkpoint view, same request.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

BASELINE = Path('/tmp/h3_align12401')
ROOT = Path('/tmp/h3_accel12501')


def launch(label: str, extra_env: dict, note: str) -> int:
    assert not (BASELINE / label).exists(), label
    key = BASELINE / f'private_IPC_key_{label}.bin'
    if key.exists():
        key.unlink()
    handle = os.open(key, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(handle, 'wb') as stream:
        stream.write(os.urandom(32))
    work = ROOT / f'cann_work_{label}'
    work.mkdir(exist_ok=False)
    raw = BASELINE / f'{label}_raw'
    assert not raw.exists()
    raw.mkdir()
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': str(BASELINE),
        'VLLM_OMNI_910A_TE_DTYPE': 'fp16',
        'VLLM_OMNI_910A_TEXT_HF_MATH': '1',
        'H3_NPU_ALIGNMENT124_BOOTSTRAP': '0',
        'H3_NPU_ALIGNMENT124_SELECTIVE_BOOTSTRAP': '1',
        'H3_NPU_ALIGNMENT124_IPC_KEY_FILE': str(key),
        'CANN_KNOWLEDGE_BANK_PROCESS_NUM': '2',
        'ASCEND_WORK_PATH': str(work),
        'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT': '7200',
        'H3_NPU_SAVE_FINAL_LATENTS124': '1',
        'H3_NPU_NAN_COMPAT124': '1',
        'H3_NPU_CAPTURE_NATIVE_MEDIA124': '1',
        'H3_NPU_RAW_OUTPUT_DIR124': str(raw),
        'VLLM_OMNI_910A_FLASH_ATTN': '1',
        'VLLM_OMNI_910A_PERF': '1',
        'HF_HUB_OFFLINE': '1',
        'OMP_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '1',
    })
    env.update(extra_env)
    command = ['/bin/bash', '-c', 'source /home/mseco/Ascend/cann-9.2.0/set_env.sh; '
               'exec /home/mseco/vllm910a/bin/python -u /tmp/h3_align12401/generate_full_NPU124g.py '
               '--output-label ' + label]
    log = ROOT / f'{label}.log'
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(BASELINE), env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT / f'{label}_process.json').write_text(json.dumps({
        'pid': process.pid, 'group': process.pid, 'command': command,
        'output_label': label, 'note': note, 'extra_env': extra_env,
        'flash_attention': 'VLLM_OMNI_910A_FLASH_ATTN=1',
        'weights_copied_or_downloaded': False}, indent=2) + '\n')
    print(json.dumps({'launched': label, 'pid': process.pid, 'extra_env': extra_env}, indent=2), flush=True)
    code = process.wait()
    print(json.dumps({'finished': label, 'exit_code': code}), flush=True)
    return code


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: launch_candidate125.py <label>')
    label = sys.argv[1]
    if label.endswith('_vae_fast'):
        # The only difference from the FA baseline: the R124 hook stops forcing the
        # original FP32 VAE parameters, so the native installer rewrites the decoder
        # Linear parameters to Half and enables its fused decoder path.
        driver_label = label[: -len('_vae_fast')]
        raise SystemExit(launch(driver_label, {'H3_NPU_VAE_PRESERVE_FP32_PARAMS124': '0'},
                                'VAE Half parameters + fused decoder fast path'))
    raise SystemExit(f'unknown candidate label {label}')