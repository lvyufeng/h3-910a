#!/usr/bin/env python3
"""Recover the failed eighth-DiT shard directly ON the RTX4090 host.

The primary downloader has now exited and the native VAE checkpoint is complete.
Only the eighth DiT shard is missing after a truncated mirror response. This does not duplicate a running transfer. A fresh mirror
resolve URL avoids stale redirect metadata; model/revision and official SHA are
unchanged. Do not relay or download weights through the workstation.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path('/home/mseco/h3_gpu12201')
NAME = 'FL2VA/transformer/model-00008-of-00013.safetensors'
REV = '0919fb684102ef115126612f3fbb84d1027e7317'


def main():
    tree = json.loads((ROOT / 'fl2va_tree_hf-mirror_com.json').read_text())
    row = next(x for x in tree if x.get('path') == NAME)
    expected = row['lfs'].get('oid', row['lfs'].get('sha256')).removeprefix('sha256:')
    destination = ROOT / 'checkpoint' / NAME
    receipt = ROOT / 'eighth_shard_recovery122.json'
    assert not receipt.exists()
    assert not destination.exists()
    pid = json.loads((ROOT / 'download_process122b.json').read_text())['pid']
    active = []
    for fd in ((Path('/proc') / str(pid) / 'fd').iterdir() if (Path('/proc') / str(pid) / 'fd').is_dir() else []):
        try:
            active.append(str(fd.resolve()))
        except OSError:
            pass
    assert not any(NAME + '.lock' in name for name in active), 'The primary first-shard transfer is still active'
    partial = ROOT / 'eighth_shard_recovery122.part'
    assert not partial.exists()
    result = {'scope': __doc__, 'completed': False, 'file': NAME, 'revision': REV,
              'source': 'hf-mirror.com transport for official MiniMaxAI/MiniMax-H3',
              'expected_bytes': row['size'], 'expected_sha256': expected, 'download_on_GPU_only': True,
              'started_unix': time.time()}
    def publish():
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    publish()
    url = f'https://hf-mirror.com/MiniMaxAI/MiniMax-H3/resolve/{REV}/{NAME}?download=true&h3_gpu_run=122'
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'H3-GPU-baseline/122'})
        h = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as f:
            print(json.dumps({'recovery_transfer_started': True, 'status': response.status, 'expected_GiB': row['size'] / 1024**3}), flush=True)
            for block in iter(lambda: response.read(8 * 1024**2), b''):
                f.write(block)
                h.update(block)
                count += len(block)
                if count > row['size']:
                    raise RuntimeError('Unexpected oversized checkpoint response')
        assert count == row['size'], (count, row['size'])
        assert h.hexdigest() == expected
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        os.link(partial, destination)
        result.update({'completed': True, 'actual_bytes': count, 'actual_sha256': h.hexdigest(),
                       'finished_unix': time.time(), 'official_file_verified': True})
        publish()
        print(json.dumps({'eighth_DiT_shard_recovered': True, 'sha256': expected}), flush=True)
    except BaseException as exc:
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish()
        raise


if __name__ == '__main__':
    main()
