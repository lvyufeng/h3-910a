#!/usr/bin/env python3
"""GPU-only parallel prefetch of four later shards, disjoint from active primary transfers.

The primary pool downloads 4 files at a time and currently owns early shards. This
sidecar owns only shards 10-13. Hugging Face's per-file lock and local metadata make
later primary visits reuse these completed files, not redownload them. Nothing is
fetched via the workstation; official revision, sizes and LFS SHA remain unchanged.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/home/mseco/h3_gpu12201')
FILES = [f'FL2VA/transformer/model-{i:05d}-of-00013.safetensors' for i in range(10, 14)]
REV = '0919fb684102ef115126612f3fbb84d1027e7317'


def main():
    os.environ.update({'HF_ENDPOINT': 'https://hf-mirror.com', 'HF_HUB_DISABLE_XET': '1', 'HF_HUB_DOWNLOAD_TIMEOUT': '120'})
    from huggingface_hub import hf_hub_download
    receipt = ROOT / 'later_shards_prefetch122.json'
    assert not receipt.exists()
    rows = {x['path']: x for x in json.loads((ROOT / 'fl2va_tree_hf-mirror_com.json').read_text()) if x['type'] == 'file'}
    result = {'scope': __doc__, 'completed': False, 'file_scopes': FILES, 'started_unix': time.time(), 'files': {}}
    def obtain(name):
        row = rows[name]
        path = Path(hf_hub_download(repo_id='MiniMaxAI/MiniMax-H3', revision=REV, filename=name,
                    endpoint='https://hf-mirror.com', local_dir=str(ROOT / 'checkpoint'), token=False))
        assert path.resolve() == (ROOT / 'checkpoint' / name).resolve()
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(block)
        expected = row['lfs'].get('oid', row['lfs'].get('sha256')).removeprefix('sha256:')
        assert h.hexdigest() == expected and path.stat().st_size == row['size'], name
        print(json.dumps({'later_shard_verified': name, 'bytes': row['size']}), flush=True)
        return name, {'bytes': row['size'], 'sha256': expected}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for future in concurrent.futures.as_completed([pool.submit(obtain, name) for name in FILES]):
                name, entry = future.result()
                result['files'][name] = entry
                receipt.write_text(json.dumps(result, indent=2) + '\n')
        result['completed'] = True
        result['finished_unix'] = time.time()
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    except BaseException as exc:
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
