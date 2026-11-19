#!/usr/bin/env python3
"""Save unchanged 910A Half projection outputs for direct GPU comparison."""
from pathlib import Path
import json,hashlib


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch,torch_npu
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.npu.set_device(0)
    root=Path('/tmp/h3_round114_transfer01/projection')
    target=root/'NPU_original_projection.safetensors';assert not target.exists()
    expected=json.loads(Path('/tmp/h3_round114_transfer01/expected_half_parameters111.json').read_text())
    inputs=json.loads(Path('/tmp/h3_round114_transfer01/expected_payload_inputs111.json').read_text())['inputs']
    old=json.loads(Path('/tmp/h3_round113_projection01/projection_range_npu113.json').read_text())
    params={}
    with safe_open('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/transformer/model-00001-of-00013.safetensors',framework='pt',device='cpu')as f:
        for name in('condition_proj.weight','condition_proj.bias'):
            t=f.get_tensor(name);assert t.dtype==torch.float16
            assert hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==expected['parameters'][name]['reference_sha256']
            params[name]=t.to('npu')
    outputs={}
    with torch.inference_mode():
        for name,pin in inputs.items():
            assert sha(pin['path'])==pin['file_SHA256']
            x=torch.load(pin['path'],weights_only=True,map_location='cpu')[pin['tensor_key']]
            assert hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==pin['tensor_SHA256']
            y=torch.nn.functional.linear(x.squeeze(0).to('npu'),params['condition_proj.weight'],params['condition_proj.bias']).cpu().contiguous()
            assert hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()==old['cases'][name]['original_unmodified_Half_projection']['sha256']
            outputs[name]=y
    save_file(outputs,str(target))
    manifest={'path':str(target),'file_SHA256':sha(target),'original_projection_R113_SHAs_reproduced':True,
              'cases':{name:old['cases'][name]['original_unmodified_Half_projection']for name in outputs},'no_source_weights_or_math_changed':True}
    with(root/'NPU_projection_manifest.json').open('x')as f:f.write(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'completed':True,'file_SHA256':manifest['file_SHA256']}))


if __name__=='__main__':main()
