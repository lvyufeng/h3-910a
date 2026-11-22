#!/usr/bin/env python3
"""Collect bounded full49 scalar JSON/text only; never raw media, latents or weights."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

SCRIPT=r'''
from pathlib import Path
import io,json,os,select,sys,tarfile
root=Path('/tmp/h3_quality13601');session=root/'engine_quality136a';receipt=root/'engine_quality136a_process.json'
r=json.loads(receipt.read_text());assert r['pid']==int(sys.argv[1])
if r.get('exit_code') is None and not r.get('completion_unknown'):
 proc=Path('/proc',str(r['launcher_pid']))
 if proc.exists():
  assert b'launch_quality136.py' in (proc/'cmdline').read_bytes()
  fd=os.pidfd_open(r['launcher_pid'])
  try:ready,_,_=select.select([fd],[],[],7500);assert ready
  finally:os.close(fd)
 r=json.loads(receipt.read_text());assert r.get('exit_code') is not None or r.get('completion_unknown')
files=[receipt,session/'lifecycle_result136.json',root/'launch_source_pins136a.json']
files+=list(root.glob('quality_worker_rank*136.json'))+list(root.glob('cohort_failure*.json'))+list(root.glob('private_abort_details136*.json'))
files+=list(session.glob('*.jsonl'))+list(session.glob('initialization_crash_pid*127.txt'))
for label in ('r01_original_false','r02_candidate_true','r03_restored_false'):
 files+=list((session/label).glob('audit_complete_pid*127.json'))
assert all(p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.txt') for p in files)
assert sum(p.stat().st_size for p in files)<=64*1024**2
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz',compresslevel=6) as archive:
 for p in files:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    assert not args.output_dir.exists()
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=20',
        '-o','ControlMaster=no','-o','ControlPath=none','ts-217',f'/usr/bin/python3 - {args.pid}']
    result=subprocess.run(command,input=SCRIPT.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    assert len(result.stdout)<=64*1024**2;args.output_dir.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
        members=archive.getmembers();assert sum(m.size for m in members)<=64*1024**2
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).suffix in ('.json','.jsonl','.txt') for m in members)
        archive.extractall(args.output_dir,filter='data')
    process=json.loads((args.output_dir/'engine_quality136a_process.json').read_text());life=json.loads((args.output_dir/'engine_quality136a/lifecycle_result136.json').read_text())
    print(json.dumps({'exit_code':process.get('exit_code'),'completed':life['completed'],'recorded_requests':len(life['requests']),
        'failure':life.get('failure'),'compressed_scalar_evidence_bytes':len(result.stdout),'raw_media_or_model_activation_files_transferred':False},indent=2))


if __name__=='__main__':main()
