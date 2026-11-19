#!/usr/bin/env python3
"""Fetch the pinned official text-encoder subset ONTO this GPU host only.

Source is the already SHA-verified official checkpoint on the leased NPU host over the
tailnet, so no weights are downloaded to the operator workstation and no weights are
relayed through it. Every file is verified against the pin from the verified official
revision before the completion receipt is written. Interrupted files resume; a mismatch
deletes the file and fails.
"""
from pathlib import Path
import hashlib,json,subprocess


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path('/home/mseco/h3_gpu_reference114')
    pins=json.loads(Path(__file__).resolve().parent.joinpath('expected_encoder_subset114.json').read_text())
    target=root/'checkpoint'/'FL2VA'/'text_encoder';target.mkdir(parents=True,exist_ok=True)
    receipt_path=root/'direct_download_verified114.json'
    receipt={'scope':__doc__,'completed':False,'host':'GPU host only','no_local_relay':True,
             'source':'leased NPU host official checkpoint over tailnet','files':{}}
    def publish():
        tmp=receipt_path.with_name(receipt_path.name+'.tmp')
        tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(receipt_path)
    publish()
    for name,row in pins['files'].items():
        path=target/name;url=pins['source_base_url']+'/'+name
        if path.is_file()and sha(path)==row['sha256']:
            receipt['files'][name]={'bytes':path.stat().st_size,'sha256':row['sha256'],'reused_verified_copy':True}
            publish();continue
        for attempt in range(1,9):
            if path.is_file()and sha(path)==row['sha256']:break
            completed=subprocess.run(['curl','-sS','-C','-','--retry','5','--retry-delay','5',
                                       '--connect-timeout','20','--speed-time','120','--speed-limit','1024',
                                       '-o',str(path),url],capture_output=True,text=True)
            if path.is_file()and sha(path)==row['sha256']:break
            print(json.dumps({'file':name,'attempt':attempt,'curl_rc':completed.returncode,
                              'stderr':completed.stderr[-300:]}),flush=True)
        actual=sha(path)if path.is_file()else None
        receipt['files'][name]={'bytes':path.stat().st_size if path.is_file()else None,'sha256':actual,
                                'expected_sha256':row['sha256'],'matches_pin':actual==row['sha256']}
        publish()
        if actual!=row['sha256']:
            if path.is_file():path.unlink()
            receipt['failure']={'file':name,'actual_sha256':actual,'expected_sha256':row['sha256']}
            publish();raise SystemExit('SHA mismatch for '+name)
        print(json.dumps({'file':name,'verified':True,'bytes':receipt['files'][name]['bytes']}),flush=True)
    receipt['completed']=True;receipt['verified_file_count']=len(receipt['files'])
    receipt['fixture_path']=str(root/'fixtures'/'encoder_input.pt')
    publish();print(json.dumps({'completed':True,'files':len(receipt['files'])},indent=1))


if __name__=='__main__':main()