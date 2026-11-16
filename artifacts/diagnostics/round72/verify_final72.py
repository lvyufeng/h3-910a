#!/usr/bin/env python3
"""CPU-only source, fullpacked fixtures and compiled tail object receipts."""
import hashlib
import json
from pathlib import Path
import re
from evidence_io72 import EvidenceWriter

ROOT=Path('/tmp/h3_round72_ab01')
writer=EvidenceWriter(ROOT/'final_verification.json')
expected=json.loads((ROOT/'expected_sources.json').read_text())
r={'production_modified':False,'production':{},'diagnostics':{},'fixtures':{},'compiled_kernels':{}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


for category in ('production','diagnostics','fixtures'):
    for path,wanted in expected[category].items():
        observed=sha(path)
        r[category][path]={'sha256':observed,'expected':wanted,'matches':observed==wanted}
r['all_sources_and_fixtures_match']=all(value['matches'] for category in ('production','diagnostics','fixtures') for value in r[category].values())
for path in sorted(Path('/tmp/h3_round72_tail01/kernels').glob('vision_rope_tail72_*/*.o')):
    m=re.match(r'vision_rope_tail72_n(\d+)_d72_h16_c(\d+)_',path.stem)
    assert m and path.is_file() and path.stat().st_size<4000000
    meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['blockDim']==int(m.group(2))
    r['compiled_kernels'][str(path)]={'sha256':sha(path),'metadata_sha256':sha(path.with_suffix('.json')),
                                    'tokens':int(m.group(1)),'cores':int(m.group(2)),'block_dim':meta['blockDim']}
boundary=json.loads(Path('/tmp/h3_round72_tail01/micro02.json').read_text())
expected_tokens={1,5,8,8064,*map(int,boundary['cases'])}
assert {row['tokens'] for row in r['compiled_kernels'].values()}==expected_tokens
writer.write(r)
print(json.dumps({'all_match':r['all_sources_and_fixtures_match'],'production_files':len(r['production']),
                  'diagnostic_files':len(r['diagnostics']),'fixture_files':len(r['fixtures']),'compiled_kernels':len(r['compiled_kernels'])},indent=2))
assert r['all_sources_and_fixtures_match']
