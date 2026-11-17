#!/usr/bin/env python3
"""CPU-only final source/reference/object fingerprint receipt after attention casts."""
import hashlib
import json
from pathlib import Path
import re
from evidence_io73 import EvidenceWriter

ROOT=Path('/tmp/h3_round73_cast_ab02');writer=EvidenceWriter(ROOT/'final_verification.json')
expected=json.loads((ROOT/'expected_sources.json').read_text())
r={'production_modified':False,'production':{},'diagnostics':{},'fixtures':{},'compiled_casts':{}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


for category in('production','diagnostics','fixtures'):
    for path,wanted in expected[category].items():
        observed=sha(path);r[category][path]={'sha256':observed,'expected':wanted,'matches':observed==wanted}
r['all_match']=all(item['matches']for category in('production','diagnostics','fixtures')for item in r[category].values())
for p in sorted(Path('/tmp/h3_round73_cast02/kernels').glob('visioncast73_*/*.o')):
    m=re.match(r'visioncast73_n(\d+)_p(\d+)_c(\d+)_',p.stem)
    assert m and p.is_file()and p.stat().st_size<4000000
    meta=json.loads(p.with_suffix('.json').read_text());assert meta['blockDim']==int(m.group(3))
    r['compiled_casts'][str(p)]={'sha256':sha(p),'metadata_sha256':sha(p.with_suffix('.json')),
                               'tokens':int(m.group(1)),'pitch':int(m.group(2)),'cores':int(m.group(3)),'block_dim':meta['blockDim']}
assert {(v['tokens'],v['pitch'])for v in r['compiled_casts'].values()}=={(n,p)for n in(1,5,61,4032)for p in(1152,3456)}
writer.write(r)
print(json.dumps({'all_match':r['all_match'],'production':len(r['production']),'diagnostics':len(r['diagnostics']),
                  'fixtures':len(r['fixtures']),'compiled_casts':len(r['compiled_casts'])},indent=2))
assert r['all_match']
