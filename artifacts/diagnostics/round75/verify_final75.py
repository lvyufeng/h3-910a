#!/usr/bin/env python3
"""CPU-only final source/actual-fixture fingerprint verification for column/transport probes."""
import hashlib
import json
from pathlib import Path
from evidence_io75 import EvidenceWriter

ROOT=Path('/tmp/h3_round75_verify01');writer=EvidenceWriter(ROOT/'final_verification.json')
expected=json.loads((ROOT/'expected_sources.json').read_text())
r={'production_modified':False,'production':{},'diagnostics':{},'fixtures':{},
   'local_rehash_scope':'Remote CPU streaming hashes of actual remote files. Local manifests/sources are separately checked in regression tests;absent remote tensor blobs are not claimed locally rehashed.'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


for category in('production','diagnostics','fixtures'):
    for path,wanted in expected[category].items():
        got=sha(path);r[category][path]={'sha256':got,'expected':wanted,'matches':got==wanted}
r['all_match']=all(p['matches']for category in('production','diagnostics','fixtures')for p in r[category].values())
writer.write(r)
print(json.dumps({'all_match':r['all_match'],'production':len(r['production']),'diagnostics':len(r['diagnostics']),'fixtures':len(r['fixtures'])},indent=2))
assert r['all_match']
