#!/usr/bin/env python3
"""Final streaming source/original-reference fingerprint proof after complete HF jobs."""
import hashlib
import json
from pathlib import Path
from evidence_io74 import EvidenceWriter

ROOT=Path('/tmp/h3_round74_verify01');writer=EvidenceWriter(ROOT/'final_verification.json')
expected=json.loads((ROOT/'expected_sources.json').read_text())
r={'production_modified':False,'production':{},'diagnostics':{},'installed_sources':{},'fixtures':{},
   'local_rehash_scope':'Streaming hashes are executed on the remote CPU for actual remote source/checkpoint/fixture paths. Local archived sources/manifests are separately compared;absent remote tensors are not represented as locally rehashed.'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


for category in('production','diagnostics','installed_sources','fixtures'):
    for path,digest in expected[category].items():
        got=sha(path);r[category][path]={'sha256':got,'expected':digest,'matches':got==digest}
r['all_match']=all(p['matches']for category in('production','diagnostics','installed_sources','fixtures')for p in r[category].values())
writer.write(r)
print(json.dumps({'all_match':r['all_match'],**{category:len(r[category])for category in('production','diagnostics','installed_sources','fixtures')}},indent=2))
assert r['all_match']
