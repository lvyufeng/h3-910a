#!/usr/bin/env python3
"""Complete the pinned H3 FL2VA checkpoint ON the leased RTX4090, never locally.

Use hf-mirror.com as transport for the official repo/revision because this machine
cannot route to huggingface.co. Every LFS file must match its official SHA256 and
size. Git-tracked small files must match their Git-blob SHA1. Previously verified
encoder weights may be hard-linked from the GPU-local R114 cache after a fresh hash
check; no model bytes cross the operator workstation.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/mseco/h3_gpu12201')
TARGET = ROOT / 'checkpoint'
CACHE = Path('/home/mseco/h3_gpu_reference114/checkpoint')
REPO = 'MiniMaxAI/MiniMax-H3'
REVISION = '0919fb684102ef115126612f3fbb84d1027e7317'
ENDPOINT = 'https://hf-mirror.com'


def checksum(path: Path, algorithm: str = 'sha256', git_blob: bool = False) -> str:
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(name: str) -> Path:
    path = TARGET / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('FL2VA/'):
        raise RuntimeError('Unexpected checkpoint file path: ' + name)
    if not path.resolve().is_relative_to(TARGET.resolve()):
        raise RuntimeError('Checkpoint path escapes the explicit GPU-only destination')
    return path


def expected_identity(row: dict) -> tuple[str, str, bool]:
    if 'lfs' in row:
        digest = row['lfs'].get('oid', row['lfs'].get('sha256', ''))
        digest = digest.removeprefix('sha256:')
        if len(digest) != 64:
            raise RuntimeError('Missing official LFS SHA256 for ' + row['path'])
        return 'sha256', digest, False
    digest = row.get('oid', row.get('blob_id', ''))
    if len(digest) != 40:
        raise RuntimeError('Missing Git-blob identity for ' + row['path'])
    return 'sha1', digest, True


def main() -> None:
    assert platform.system() == 'Linux', 'Weights may not be downloaded on the local workstation'
    assert ROOT.resolve() == Path('/home/mseco/h3_gpu12201')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    assert len(gpu) == 8 and all('RTX 4090' in x for x in gpu)
    TARGET.mkdir(exist_ok=True)
    receipt = ROOT / 'checkpoint_verified122.json'
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior.get('completed'):
            for row in prior['files'].values():
                path = checked_path(row['path'])
                assert path.stat().st_size == row['bytes']
                assert checksum(path, row['hash_algorithm'], row['git_blob']) == row['expected_digest']
            print('Existing GPU-only checkpoint receipt verified; no download replay.', flush=True)
            return
        raise RuntimeError('An unfinished receipt already exists; inspect it before choosing a new run')
    tree_path = ROOT / 'fl2va_tree_hf_mirror_com.json'
    rows = json.loads(tree_path.read_text())
    rows = [r for r in rows if r['type'] == 'file']
    assert rows and all(r['path'].startswith('FL2VA/') for r in rows)
    total_bytes = sum(row['size'] for row in rows)
    free = __import__('shutil').disk_usage(ROOT).free
    assert free > total_bytes + 30 * 1024**3, 'Insufficient GPU-host disk space'
    os.environ['HF_ENDPOINT'] = ENDPOINT
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    from huggingface_hub import hf_hub_download
    result = {
        'scope': __doc__, 'completed': False, 'repo_id': REPO, 'revision': REVISION,
        'endpoint': ENDPOINT, 'download_executed_on_gpu_host': platform.node(),
        'no_local_weight_download_or_relay': True, 'checkpoint_root': str(TARGET),
        'expected_files': len(rows), 'expected_total_bytes': total_bytes,
        'started_unix': time.time(), 'files': {},
    }
    lock = threading.Lock()
    progress = ROOT / 'checkpoint_download_progress122.json'

    def publish() -> None:
        progress.write_text(json.dumps(result, indent=2) + '\n')

    publish()

    def obtain(row: dict) -> None:
        name = row['path']
        path = checked_path(name)
        algorithm, digest, git_blob = expected_identity(row)
        original = CACHE / name
        reused = False
        if original.is_file() and original.stat().st_size == row['size']:
            if checksum(original, algorithm, git_blob) == digest:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    assert checksum(path, algorithm, git_blob) == digest
                else:
                    os.link(original, path)
                reused = True
        if not reused:
            returned = Path(hf_hub_download(
                repo_id=REPO, filename=name, revision=REVISION, endpoint=ENDPOINT,
                local_dir=str(TARGET), token=False, etag_timeout=30,
            ))
            assert returned.resolve() == path.resolve(), returned
        actual = checksum(path, algorithm, git_blob)
        if actual != digest or path.stat().st_size != row['size']:
            raise RuntimeError(f'Official identity mismatch: {name}: {actual} vs {digest}')
        item = {
            'path': name, 'bytes': path.stat().st_size, 'hash_algorithm': algorithm,
            'git_blob': git_blob, 'expected_digest': digest, 'actual_digest': actual,
            'reused_verified_gpu_local_cache': reused,
        }
        with lock:
            result['files'][name] = item
            publish()
            print(json.dumps({'verified': name, 'bytes': item['bytes'], 'gpu_cache_reused': reused,
                              'completed_files': len(result['files']), 'expected_files': len(rows)}), flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(obtain, row) for row in rows]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        assert len(result['files']) == len(rows)
        result['completed'] = True
        result['finished_unix'] = time.time()
        publish()
        with receipt.open('x') as f:
            f.write(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'completed': True, 'files': len(rows), 'total_GiB': total_bytes / 1024**3}), flush=True)
    except BaseException as exc:
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise


if __name__ == '__main__':
    main()
