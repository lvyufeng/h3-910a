#!/usr/bin/env python3
"""Read-only source/capability/NUMA preflight, no model weights loaded."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/tmp/h3_runtime12701')


def main():
    assert ROOT.is_dir()
    result={'scope':__doc__,'weights_loaded':False,'hostname':os.uname().nodename,'affinity':sorted(os.sched_getaffinity(0))}
    for name,cmd in [('npu',['/usr/local/sbin/npu-smi','info']),('topology',['/usr/local/sbin/npu-smi','info','-t','topo']),('numa',['numactl','--hardware'])]:
        try:
            p=subprocess.run(cmd,capture_output=True,text=True);result[name]={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except FileNotFoundError:result[name]={'available':False,'fallback':'sysfs cpulist and /proc NUMA maps'}
    assert result['npu']['exit_code']==result['topology']['exit_code']==0
    result['numa_sysfs']={str(p):p.read_text() for p in Path('/sys/devices/system/node').glob('node*/cpulist')}
    import torch
    import torch_npu
    import vllm_omni.diffusion.offloader.module_residency as residency
    import vllm_omni.diffusion.offloader.layerwise_backend as layerwise
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline
    import vllm_omni.diffusion.models.minimax_h3.vae as vae
    from vllm_omni.platforms import current_omni_platform
    result.update({'torch_version':torch.__version__,'torch_npu_version':torch_npu.__version__,
                   'sources':{},'shared_stream_environment':os.environ.get('VLLM_OMNI_910A_SHARED_STREAM','<default>')})
    for name,module in [('residency',residency),('layerwise',layerwise),('pipeline',pipeline),('vae',vae),('platform_class',__import__(type(current_omni_platform).__module__,fromlist=['x']))]:
        path=Path(module.__file__).resolve();result['sources'][name]={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    # Main process capability only. Actual worker handles will be sampled in their own thread.
    torch.npu.set_device(0)
    current=torch.npu.current_stream();factory=current_omni_platform.Stream()
    result['parent_only_stream_probe']={'current':int(current.npu_stream),'platform_Stream':int(factory.npu_stream),
        'platform_factory_is_current':current==factory,'not_worker_execution_evidence':True}
    result['runtime_device_name']=torch.npu.get_device_name(0)
    result['completed']=True
    path=ROOT/'preflight127.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
