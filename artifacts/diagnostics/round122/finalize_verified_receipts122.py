#!/usr/bin/env python3
"""Consolidate fresh official per-file verification receipts without another 134-GiB rehash.

The primary already hashed every successfully completed file, and two truncated
shards were then downloaded with fresh URLs and independently verified while
streaming. All writer processes have exited. Check every size, identity, and proof,
then record the final all-81-files checkpoint receipt. No checkpoint bytes change.
"""
import json
from pathlib import Path
import time

ROOT = Path('/home/mseco/h3_gpu12201')


def main():
    destination = ROOT / 'checkpoint_verified122.json'
    assert not destination.exists()
    primary = json.loads((ROOT / 'checkpoint_download_progress122.json').read_text())
    receipts = primary['files'].copy()
    recovery_names = ['first_shard_recovery122.json', 'eighth_shard_recovery122.json']
    recoveries = {}
    for name in recovery_names:
        r = json.loads((ROOT / name).read_text())
        assert r['completed'] and r['revision'] == '0919fb684102ef115126612f3fbb84d1027e7317'
        assert r['actual_bytes'] == r['expected_bytes'] and r['actual_sha256'] == r['expected_sha256']
        receipts[r['file']] = {'path': r['file'], 'bytes': r['actual_bytes'], 'hash_algorithm': 'sha256',
                                'git_blob': False, 'expected_digest': r['expected_sha256'],
                                'actual_digest': r['actual_sha256'], 'proof_receipt': name}
        recoveries[name] = r
    for name in ['download_process122b.json', 'first_shard_recovery_process122.json',
                 'later_shards_prefetch_process122.json', 'eighth_shard_recovery_process122.json']:
        pid = json.loads((ROOT / name).read_text())['pid']
        assert not (Path('/proc') / str(pid)).exists(), 'A checkpoint writer is still active'
    attestation = json.loads((ROOT / 'independent_official_metadata_verification122.json').read_text())
    assert attestation['all_mirror_sizes_git_blob_ids_and_LFS_SHA256_match_official']
    rows = [row for row in json.loads((ROOT / 'fl2va_tree_hf-mirror_com.json').read_text()) if row['type'] == 'file']
    assert len(rows) == 81 and len(receipts) == 81
    files = {}
    for row in rows:
        name = row['path']
        proof = receipts[name]
        path = ROOT / 'checkpoint' / name
        assert not Path(name).is_absolute() and name.startswith('FL2VA/') and '..' not in Path(name).parts
        assert path.is_file() and path.stat().st_size == proof['bytes'] == row['size'], name
        if 'lfs' in row:
            expected = row['lfs']['oid']
            assert proof['hash_algorithm'] == 'sha256' and not proof['git_blob']
        else:
            expected = row['oid']
            assert proof['hash_algorithm'] == 'sha1' and proof['git_blob']
        assert proof['actual_digest'] == proof['expected_digest'] == expected, name
        stat = path.stat()
        files[name] = {**proof, 'inode_at_final_attestation': stat.st_ino,
                       'mtime_ns_at_final_attestation': stat.st_mtime_ns}
    result = {'scope': __doc__, 'completed': True, 'repo_id': 'MiniMaxAI/MiniMax-H3',
              'revision': '0919fb684102ef115126612f3fbb84d1027e7317',
              'checkpoint_root': str(ROOT / 'checkpoint'), 'download_executed_on_GPU_only': True,
              'no_local_weight_download_or_relay': True, 'expected_files': len(rows),
              'expected_total_bytes': sum(row['size'] for row in rows), 'files': files,
              'independent_official_metadata_attestation': attestation,
              'primary_transport_failure_preserved': primary.get('failure'),
              'separately_recovered_shards': recovery_names, 'verified_unix': time.time(),
              'redundant_second_whole_checkpoint_hash_avoided': True}
    with destination.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checkpoint_verified': True, 'official_files': len(rows),
                      'total_GiB': result['expected_total_bytes'] / 1024**3}), flush=True)


if __name__ == '__main__':
    main()
