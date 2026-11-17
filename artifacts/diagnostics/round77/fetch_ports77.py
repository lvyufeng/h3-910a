#!/usr/bin/env python3
"""Persist pinned FP16 community-port sources without importing,installing or executing them."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parent
PINS={
 'amdur':{'repo':'Amduraznak/minimax-h3-fp16-fix','commit':'b09897c5c3bf4af7262bb43d04a71a94563c1590','files':['README.md','minimax_h3_fp16_fix.py']},
 'star7':{'repo':'star7code/minimax-h3-fp16-exact-star7','commit':'27c44a7c39fc8de2d0e8eac6c5ad8032962a30d2','files':['README.md','__init__.py','nodes.py']},
 'icbears':{'repo':'Icbears/minimax-h3-v100-patch','commit':'d869f4fc4826ed4b1b1b32c862a2193da0cf8daa','files':['README.md','runtime_patch.py']},
 'safe':{'repo':'aaalll12322/ComfyUI-MiniMaxH3-FP16Safe','commit':'73456ad94d92d0aec9be2e392888933ce246e4a8','files':['README.md','nodes.py']},
}


def fetch(item):
    name,filename,info=item;url=f"https://raw.githubusercontent.com/{info['repo']}/{info['commit']}/{filename}"
    try:
        try:response=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'H3-FP16-baseline-audit/1.0'}),timeout=45)
        except urllib.error.HTTPError as error:response=error
        with response:
            data=response.read(1024*1024+1);assert len(data)<=1024*1024;data.decode('utf-8');status=response.status
        path=ROOT/'port_sources'/(name+'_'+filename.replace('/','__')+'.txt')
        with path.open('xb')as f:f.write(data)
        return name+'/'+filename,{'url':url,'repo':info['repo'],'commit':info['commit'],'status':status,'path':str(path.relative_to(ROOT)),
                                  'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source_only_not_executed':True}
    except BaseException as error:return name+'/'+filename,{'url':url,'failure':{'type':type(error).__name__,'message':str(error)}}


if __name__=='__main__':
    (ROOT/'port_sources').mkdir(exist_ok=False)
    tasks=[(name,filename,info)for name,info in PINS.items()for filename in info['files']]
    with ThreadPoolExecutor(max_workers=4)as workers:results=dict(workers.map(fetch,tasks))
    path=ROOT/'port_source_receipts77.json';assert not path.exists();path.write_text(json.dumps({'scope':__doc__,'pins':PINS,'sources':results},indent=2)+'\n')
    print({name:r.get('status',r.get('failure'))for name,r in results.items()})
