#!/usr/bin/env python3
"""Read bounded reference progress/provenance without restarting any work."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/tmp/h3_round78_hf_fp1601')
KEYS=('source_sha256','versions','loaded_parameter_tensors','complete_parameter_bytes','original_selected_parameter_sha256','parameter_dtypes','buffer_dtypes','Linear_input_dtypes_observed','Linear_output_dtypes_observed','full_multimodal_reference_completed','first_full_reference_completed_repeat_pending','repeat_pre_final_norm_bits_equal','repeat_post_final_norm_bits_equal','CPU_reference_compute_wall_s','CPU_full_repeat_wall_s','failure','first_nonfinite_Linear')
r={'root':str(ROOT),'files':{},'model_cpu_processes':[]}
for name in('run_started.json','run_finished.json','reference.json'):
    path=ROOT/name
    if path.exists():
        data=path.read_bytes();assert len(data)<4*1024*1024
        value=json.loads(data);r['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        if name=='reference.json':
            r['reference']={key:value[key]for key in KEYS if key in value}
            r['reference']['completed_vision_layers']=len(value.get('vision_layers',[]));r['reference']['completed_language_layers']=len(value.get('language_layers',[]))
            r['reference']['stages']={name:info['file_sha256']for name,info in value.get('stages',{}).items()}
        else:r[name]=value
for process in Path('/proc').iterdir():
    if not process.name.isdecimal():continue
    try:
        if process.stat().st_uid!=os.getuid():continue
        cmd=(process/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        if str(ROOT)in cmd and 'python'in cmd:r['model_cpu_processes'].append({'pid':int(process.name),'cmdline':cmd[:1024]})
    except(FileNotFoundError,PermissionError,ProcessLookupError):continue
print(json.dumps(r,indent=2))
