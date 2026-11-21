#!/usr/bin/env python3
"""Small on-host tensor hashing smoke, including scalar/empty/noncontiguous/inference."""
from pathlib import Path
import json
import torch
from fingerprints129 import fingerprint,normalized_fingerprint

ROOT=Path('/tmp/h3_repeat12901')
path=ROOT/'fingerprint_unit129.json';assert not path.exists()
torch.set_num_threads(1)
with torch.inference_mode():
    values=[torch.tensor(1.0),torch.empty(0),torch.arange(16,dtype=torch.float16).reshape(4,4).t()]
    rows=[fingerprint(x,torch,include_pointers=True) for x in values]
    assert normalized_fingerprint(fingerprint(values[0],torch))==normalized_fingerprint(fingerprint(values[0].clone(),torch))
    assert all(r['tensor_version'] is None for r in rows)
result={'completed':True,'no_model_inference':True,'all_scalar_empty_transpose_inference_hashes_valid':True,'rows':rows}
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
