#!/usr/bin/env python3
"""Download exact official H3 weights ON the leased 4090 only; never local relay."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    args=argparse.ArgumentParser(description=__doc__);args.add_argument('--root',type=Path,default=Path('/home/mseco/h3_gpu_reference114'))
    args.add_argument('--expected',type=Path,required=True);flags=args.parse_args()
    if platform.system()!='Linux':raise RuntimeError('Weight download is forbidden on the local workstation')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if not devices or not all('RTX 4090'in value for value in devices):raise RuntimeError('Only the explicitly leased 4090 host may download these weights')
    root=flags.root.absolute()
    if not str(root).startswith('/home/mseco/h3_gpu_reference114'):raise RuntimeError('Unexpected GPU-only download destination')
    root.mkdir(parents=True,exist_ok=True);root=root.resolve(strict=True)
    expected=json.loads(flags.expected.read_text());assert expected['no_local_weight_download_or_relay']
    assert expected['repo_id']=='MiniMaxAI/MiniMax-H3'and expected['revision']=='0919fb684102ef115126612f3fbb84d1027e7317'
    target=root/'checkpoint';done=root/'direct_download_verified114.json'
    if done.exists():
        prior=json.loads(done.read_text());assert prior['completed']
        for name,row in expected['files'].items():assert sha(target/name)==row['sha256'],name
        print('Existing directly downloaded GPU checkpoint verified; no transfer replay.');return
    required=expected['known_encoder_bytes']+24*1024**3
    if shutil.disk_usage(root).free<required:raise RuntimeError('Insufficient GPU-host disk capacity before download')
    # huggingface.co is unreachable from this host; the public mirror is used ONLY as a
    # transport.  Every downloaded file must still match the pin derived from the verified
    # official revision, otherwise this script fails without writing the completion receipt.
    os.environ['HF_ENDPOINT']='https://hf-mirror.com'
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=expected['repo_id'],revision=expected['revision'],local_dir=str(target),
                      allow_patterns=list(expected['files']),token=False,max_workers=4)
    verified={}
    for name,row in expected['files'].items():
        path=target/name;actual=sha(path);assert actual==row['sha256'],name
        verified[name]={'sha256':actual,'bytes':path.stat().st_size}
    result={'scope':__doc__,'completed':True,'download_executed_on_GPU_host':True,'hostname':platform.node(),
            'GPU_devices':devices,'repo_id':expected['repo_id'],'revision':expected['revision'],
            'checkpoint_root':str(target),'direct_official_download_not_local_relay':True,
            'download_endpoint':'hf-mirror.com (public mirror; transport only)',
            'official_huggingface_unreachable_on_GPU_host':'ConnectError network unreachable',
            'each_file_SHA256_must_match_pin_from_verified_official_revision':True,
            'original_BF16_weights_no_FP16_roundtrip':True,'files':verified}
    with done.open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'root':str(target),'verified_files':len(verified)},indent=2))


if __name__=='__main__':main()
