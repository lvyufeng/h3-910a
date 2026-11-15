#!/usr/bin/env python3
"""Publish equivalent native model metadata beside persistent converted storage."""
import hashlib
import json
from pathlib import Path

home=Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53')
private=Path('/tmp/h3_round53_native49_fp16_model')
assert home.resolve(strict=True)==home
partition=home/'FL2VA'
assert partition.resolve(strict=True)==partition
assert (partition/'transformer').is_dir() and not (partition/'transformer').is_symlink()
meta=(private/'model_index.json').read_bytes()
assert meta==(private/'FL2VA/model_index.json').read_bytes()
index=json.loads(meta)
assert index['_class_name']=='MiniMaxH3Pipeline' and 'base_schedule' not in index['_minimax_h3']
for name in ('model_index.json','FL2VA/model_index.json','runtime_storage_conversion.json'):
    if (home/name).exists(): raise RuntimeError('Refuse existing deployment metadata: '+name)
for entry in (private/'FL2VA').iterdir():
    if entry.name in ('model_index.json','transformer'): continue
    target=partition/entry.name
    if target.exists() or target.is_symlink(): raise RuntimeError('Refuse existing component link: '+str(target))
    target.symlink_to(entry.resolve(strict=True),target_is_directory=entry.is_dir())
for path in (home/'model_index.json',partition/'model_index.json'):
    with path.open('xb') as f:f.write(meta)
conversion=Path('/tmp/h3_round53_validation01/native_storage_conversion.json').read_bytes()
with (home/'runtime_storage_conversion.json').open('xb') as f:f.write(conversion)
components=[]
for entry in (private/'FL2VA').iterdir():
    if entry.name=='model_index.json':continue
    assert (partition/entry.name).resolve(strict=True)==entry.resolve(strict=True)
    components.append(entry.name)
report={'persistent_model_path':str(home),'tested_temporary_model_path':str(private),
        'metadata_sha256':hashlib.sha256(meta).hexdigest(),'same_metadata_and_physical_components':True,
        'components':sorted(components),'runtime_dtype':'half','native_task':'fl2va','request_intervals':49,
        'scope':'Persistent metadata/component publication; completed inference used byte-identical metadata and same physical component files through the temporary alias'}
Path('/tmp/h3_round53_validation01/persistent_model.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
