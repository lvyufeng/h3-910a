#!/usr/bin/env python3
"""Wait without busy-polling, then supervise one complete H3 FP16 generation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

ROOT = Path('/home/mseco/h3_gpu12201')


def wait_owned_download() -> None:
    deadline = time.monotonic() + 4 * 3600
    owned = [
        ('download_process122b.json', 'download_full_on_gpu122b.py'),
        ('first_shard_recovery_process122.json', 'recover_first_dit_shard122.py'),
        ('later_shards_prefetch_process122.json', 'prefetch_later_dit_shards122.py'),
    ]
    descriptors = []
    with selectors.DefaultSelector() as selector:
        for receipt_name, script in owned:
            pid = int(json.loads((ROOT / receipt_name).read_text())['pid'])
            proc = Path('/proc') / str(pid)
            if not proc.is_dir():
                continue
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            if not any(cmd):
                continue  # An exited zombie owns no ongoing transfer.
            assert b'python' in cmd[0] and str(ROOT / script).encode() in cmd
            assert (proc / 'cwd').resolve() == ROOT
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            descriptors.append(fd)
            selector.register(fd, selectors.EVENT_READ)
        while selector.get_map():
            ready = selector.select(timeout=max(0, deadline - time.monotonic()))
            if not ready:
                raise TimeoutError('Pinned GPU-only checkpoint preparation exceeded the run budget')
            for key, _ in ready:
                selector.unregister(key.fd)
    for fd in descriptors:
        os.close(fd)

    verified_path = ROOT / 'checkpoint_verified122.json'
    if not verified_path.exists():
        # The primary records its failed early first-shard future only after
        # draining its other futures. Consolidate the independently recovered
        # first shard with all completed primary/prefetched files, checking
        # every official file again. Never treat a partially downloaded path
        # or a successful HTTP response alone as a valid checkpoint.
        rows = [row for row in json.loads((ROOT / 'fl2va_tree_hf-mirror_com.json').read_text()) if row['type'] == 'file']
        primary = json.loads((ROOT / 'checkpoint_download_progress122.json').read_text())
        verified = {'completed': False, 'repo_id': 'MiniMaxAI/MiniMax-H3',
                    'revision': '0919fb684102ef115126612f3fbb84d1027e7317',
                    'checkpoint_root': str(ROOT / 'checkpoint'), 'download_executed_on_GPU_only': True,
                    'no_local_weight_download_or_relay': True,
                    'expected_total_bytes': sum(row['size'] for row in rows),
                    'primary_failure_preserved': primary.get('failure'), 'files': {}}
        for row in rows:
            name = row['path']
            path = ROOT / 'checkpoint' / name
            assert path.is_file() and path.stat().st_size == row['size'], 'Missing/incomplete checkpoint file: ' + name
            if 'lfs' in row:
                algorithm = 'sha256'
                expected = row['lfs'].get('oid', row['lfs'].get('sha256')).removeprefix('sha256:')
                git_blob = False
            else:
                algorithm, expected, git_blob = 'sha1', row['oid'], True
            h = hashlib.new(algorithm)
            if git_blob:
                h.update(f'blob {path.stat().st_size}\0'.encode())
            with path.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024**2), b''):
                    h.update(block)
            assert h.hexdigest() == expected, 'Official checkpoint hash mismatch: ' + name
            verified['files'][name] = {'path': name, 'bytes': path.stat().st_size,
                                      'hash_algorithm': algorithm, 'git_blob': git_blob,
                                      'expected_digest': expected, 'actual_digest': h.hexdigest()}
        assert len(verified['files']) == 81
        verified['completed'] = True
        verified['verified_unix'] = time.time()
        with verified_path.open('x') as f:
            f.write(json.dumps(verified, indent=2) + '\n')
    verified = json.loads(verified_path.read_text())
    assert verified['completed'] and verified['revision'] == '0919fb684102ef115126612f3fbb84d1027e7317'


def main():
    receipt = ROOT / 'full_generation_wide_output_supervisor122.json'
    if receipt.exists():
        raise RuntimeError('Reconcile the recorded supervisor; never replay uncertain generation')
    r = {'scope': __doc__, 'completed': False, 'generation_started': False, 'supervisor_pid': os.getpid(),
         'created_unix': time.time(), 'case': 'wide_output_fp16', 'checkpoint_download_executed_on_GPU_only': True}
    def publish():
        receipt.write_text(json.dumps(r, indent=2) + '\n')
    publish()
    try:
        assert json.loads((ROOT / 'final_runtime_preflight122.json').read_text())['passed']
        wait_owned_download()
        manifest = json.loads((ROOT / 'runtime_audit_sources122_wide_output.json').read_text())
        for file, expected in [('generate_full_fp16_wide_output122.py', manifest['generation_script_sha256']),
                               ('h3_gpu_audit122.py', manifest['audit_script_sha256'])]:
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected
        env = os.environ.copy()
        env.update({'PYTHONPATH': str(ROOT), 'HF_HUB_OFFLINE': '1', 'HF_ENDPOINT': 'https://hf-mirror.com',
                    'HF_HUB_DISABLE_XET': '1', 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
                    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_LOGGING_LEVEL': 'INFO',
                    'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
                    'VLLM_DIFFUSION_RPC_TIMEOUT': '3600', 'PYTHONUNBUFFERED': '1'})
        command = [str(ROOT / 'venv/bin/python'), '-u', str(ROOT / 'generate_full_fp16_wide_output122.py'), '--case', 'wide_output_fp16']
        with (ROOT / 'logs/full_generation_wide_output_fp16_122.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            r['generation_pid'] = process.pid
            r['generation_started'] = True
            r['generation_started_unix'] = time.time()
            r['command'] = command
            r['log'] = str(ROOT / 'logs/full_generation_wide_output_fp16_122.log')
            publish()
            print(json.dumps({'generation_started': True, 'pid': process.pid, 'case': 'wide_output_fp16'}), flush=True)
            exit_code = process.wait()
        r['generation_exit_code'] = exit_code
        r['finished_unix'] = time.time()
        r['completed'] = True
        publish()
        print(json.dumps({'generation_exited': True, 'exit_code': exit_code, 'receipt': str(receipt)}), flush=True)
        raise SystemExit(exit_code)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            r['supervisor_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
            publish()
        raise


if __name__ == '__main__':
    main()
