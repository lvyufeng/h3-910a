#!/usr/bin/env python3
"""Verify existing NPU transformer shards against the independently verified GPU checkpoint.

Read-only hash validation on the NPU host; only metadata leaves it. No weights are
downloaded or relayed, no CPU inference is used, and the successful GPU checkpoint
is not unnecessarily rehashed. Original BF16/FP32 checkpoint bytes are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    checkpoint=Path(args.checkpoint).resolve()
    output=Path(args.output)
    assert not output.exists()
    result={'scope':__doc__,'completed':False,'repo':manifest['repo'],'revision':manifest['revision'],
            'existing_NPU_checkpoint_root':str(checkpoint),'no_local_model_weights':True,'CPU_inference_executed':False,
            'files':{}}
    started=time.perf_counter()
    for name,expected in manifest['files'].items():
        assert name==Path(name).name and name not in ['.','..']
        path=(checkpoint/name).resolve();assert path.parent==checkpoint
        before=path.stat();assert before.st_size==expected['bytes'], name
        digest=hashlib.new(expected['hash_algorithm'])
        if expected['git_blob']:
            digest.update(f'blob {before.st_size}\0'.encode())
        with path.open('rb') as file:
            while data:=file.read(8*1024*1024):
                digest.update(data)
        after=path.stat()
        assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
        actual=digest.hexdigest()
        result['files'][name]={'bytes':before.st_size,'hash_algorithm':expected['hash_algorithm'],
                               'actual_digest':actual,'expected_digest':expected['digest'],'matched':actual==expected['digest']}
        output.write_text(json.dumps(result,indent=2)+'\n')
        assert actual==expected['digest'], name
    result['completed']=True
    result['all_13_original_shards_and_config_index_identical']=len(result['files'])==15
    result['elapsed_s_read_only_validation']=time.perf_counter()-started
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'matched_files':len(result['files']),
                      'all_original_transformer_shards_identical':True,'elapsed_s':result['elapsed_s_read_only_validation']}),flush=True)


if __name__=='__main__':
    main()
