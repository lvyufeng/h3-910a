#!/usr/bin/env python3
"""Prepare isolated NPU vLLM-Omni continuation without changing shared installed code.

Reuse the executed R123 range-safe FP16 transformer plus tested patch/time guards.
The local encoder copy changes only the three hardcoded BF16 media/output casts to
its actual FP16 parameter/input dtype; FP16 storage was already supported by env.
No weight files are copied, downloaded or rounded through BF16 on the NPU.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/tmp/h3_align12401')
OLD=Path('/tmp/h3_align12301')


def main():
    receipt=ROOT/'isolated_integration_sources124.json';assert not receipt.exists()
    package=ROOT/'h3_replay123';assert not package.exists()
    sources={}
    package.mkdir();(package/'__init__.py').write_text('')
    for name in ['transformer.py','denoise_loop.py','scheduling_minimax_h3_euler_ancestral.py']:
        path=OLD/'h3_replay123'/name
        data=path.read_bytes()
        (package/name).write_bytes(data);sources[name]=hashlib.sha256(data).hexdigest()
    assert sources['transformer.py']=='5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b'
    for name in ['fp32_patch_residual_guard123.py','fp32_time_residual_guard123.py']:
        data=(OLD/name).read_bytes();(ROOT/name).write_bytes(data);sources[name]=hashlib.sha256(data).hexdigest()
    original=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py').read_text()
    text=original
    for old,new in [('pixel_values.to(device, torch.bfloat16)','pixel_values.to(device, inputs_embeds.dtype)'),
                    ('pixel_values_videos.to(device, torch.bfloat16)','pixel_values_videos.to(device, inputs_embeds.dtype)'),
                    ('return hidden.to(torch.bfloat16)','return hidden.to(inputs_embeds.dtype)')]:
        assert text.count(old)==1;text=text.replace(old,new)
    path=ROOT/'NPU_encoder_FP16_124.py';assert not path.exists();path.write_text(text)
    compile(text,str(path),'exec')
    report={'scope':__doc__,'completed':True,'source_digests':sources,
            'original_NPU_encoder_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
            'isolated_FP16_encoder_source_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'encoder_delta':'Only3 hardcoded BF16 media/output casts become the actual FP16 inputs_embeds dtype; no FP16-to-BF16 roundtrip.',
            'shared_installed_source_modified':False,'model_weights_downloaded_copied_or_modified':False}
    receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
