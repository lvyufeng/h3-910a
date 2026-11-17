#!/usr/bin/env python3
"""Read-only HF reference provenance: original checkpoint, installed code and inputs."""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import torch
import transformers
from evidence_io74 import EvidenceWriter

ROOT=Path('/tmp/h3_round74_hf01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
writer=EvidenceWriter(ROOT/'provenance01.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


r={'python_executable':sys.executable,'versions':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__},
   'platform':platform.platform(),'cpu_reference_only':True,'installed_sources':{},'checkpoint_files':{},'reference_fixtures':{},'executed_diagnostics':{}}
for name in('transformers.models.qwen3_vl.modeling_qwen3_vl','transformers.models.qwen3_vl.configuration_qwen3_vl',
             'transformers.models.qwen3_vl.processing_qwen3_vl','transformers.masking_utils','transformers.integrations.sdpa_attention',
             'torch.nn.modules.linear','torch.nn.functional'):
    module=importlib.import_module(name);path=Path(module.__file__)
    r['installed_sources'][name]={'path':str(path),'sha256':sha(path)}
for name in('config.json','model.safetensors.index.json','preprocessor_config.json'):
    path=MODEL/name
    if path.is_file():r['checkpoint_files'][str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
for path in(Path('/tmp/h3_round54_capture01/encoder_input.pt'),Path('/tmp/h3_round70_components01/fresh_encoder_reference.pt')):
    r['reference_fixtures'][str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
for directory,names in((ROOT,('hf_multimodal74.py','evidence_io74.py','provenance74.py')),
                        (Path('/tmp/h3_round74_capture01'),('capture_text74.py','candidate_padded_text_encoder.py','evidence_io74.py'))):
    for name in names:
        path=directory/name;r['executed_diagnostics'][str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
r['checkpoint_shards_used_for_selected50']=sorted({v for k,v in index.items()if k.startswith('model.visual.')or k.startswith('model.language_model.')and('.layers.'not in k or int(k.split('.layers.')[1].split('.')[0])<50)})
r['excluded_checkpoint_language_layers']=sorted({int(k.split('.layers.')[1].split('.')[0])for k in index if k.startswith('model.language_model.layers.')and int(k.split('.layers.')[1].split('.')[0])>=50})
r['exclusions_are_selected_hidden_state_boundary_not_missing_weights']=r['excluded_checkpoint_language_layers']==list(range(50,64))
writer.write(r);print(json.dumps({'installed_sources':len(r['installed_sources']),'diagnostic_sources':len(r['executed_diagnostics']),
                               'excluded_layers':r['excluded_checkpoint_language_layers']},indent=2))
