#!/usr/bin/env python3
"""Launch corrected full49 regression after exact gates; no uncertain-work replay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/tmp/h3_budget149')
LABEL='NPU_full_generation126b'


def main():
    out=ROOT/LABEL;process_receipt=ROOT/(LABEL+'_process.json')
    assert not out.exists() and not process_receipt.exists(), 'Refuse replay of a prior or uncertain launch'
    unit=json.loads((ROOT/'cache_nested_inference_unit126.json').read_text())
    cache_source=ROOT/'vae/autocast_operand_cache126.py'
    assert unit['completed'] and unit['outer_inference_mode_stayed_enabled'] and unit['private_copies_track_version']
    assert unit['all_words_equal'] and unit['all_original_FP32_parameters_unchanged']
    assert unit['cache_source_sha256']==hashlib.sha256(cache_source.read_bytes()).hexdigest()
    safety=json.loads((ROOT/'equivalent_safety_unit126.json').read_text())
    assert safety['verification_passed'] and safety['every_65536_Half_bit_pattern_tested_on_NPU']
    for rank in range(8):
        gate=json.loads((ROOT/'VAE_operand_cache126c'/f'rank{rank}.json').read_text())
        assert gate['completed'] and gate['all_prequantized_tile_pairs_equal']
        assert gate['cache_after_trial']['original_parameters_unchanged']
    pins={'/tmp/h3_align12401/NPU_encoder_FP16_124.py':'0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026',
          '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b'}
    for path,digest in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    work=ROOT/('cann_'+LABEL);work.mkdir(exist_ok=False)
    assignments={
        'H3_R126_IPC_KEY_FILE':str(ROOT/'private_IPC_key126.bin'),
        'H3_R126_FULL_SELECTIVE_BOOTSTRAP':'1','H3_R126_VAE_OPERAND_CACHE':'1',
        'H3_NPU_ALIGNMENT124_BOOTSTRAP':'0','H3_NPU_ALIGNMENT124_SELECTIVE_BOOTSTRAP':'0',
        'H3_R126_GENERATION_OUT':str(out),'H3_NPU_RAW_OUTPUT_DIR124':str(out),
        'H3_NPU_CAPTURE_NATIVE_MEDIA124':'1','H3_NPU_SAVE_FINAL_LATENTS124':'1',
        'H3_NPU_NAN_COMPAT124':'1','H3_NPU_VAE_PRESERVE_FP32_PARAMS124':'1',
        'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT':'7200','VLLM_OMNI_910A_FLASH_ATTN':'1',
        'VLLM_OMNI_910A_FP16_SEQ':'1','VLLM_OMNI_910A_TE_DTYPE':'fp16','VLLM_OMNI_910A_TEXT_HF_MATH':'1',
        'VLLM_OMNI_910A_NO_FUSED_ACT':'1','VLLM_OMNI_910A_NO_FUSED_NORM':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
        'ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','ASCEND_WORK_PATH':str(work)}
    assert all(' ' not in value and '\n' not in value for value in assignments.values())
    shell='source /home/mseco/Ascend/cann-9.2.0/set_env.sh; export PYTHONPATH=/tmp/h3_budget149/vae:/tmp/h3_budget149:/tmp/h3_align12401:$PYTHONPATH; export '
    shell+=' '.join(k+'='+v for k,v in assignments.items())
    shell+='; exec /home/mseco/vllm910a/bin/python -u /tmp/h3_budget149/generate_full_NPU126.py --output-label '+LABEL
    command=['/bin/bash','-c',shell]
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'integration_hooks126.py',ROOT/'generate_full_NPU126.py',ROOT/'full_bootstrap126.py',ROOT/'capture_NPU_media126.py',ROOT/'equivalent_safety126.py',cache_source]}
    with (ROOT/(LABEL+'.log')).open('xb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':process.pid,'command':command,'source_sha256':sources,'all_original_sources_pinned':pins,
             'nested_inference_lifecycle_unit_passed':True,'new_quality_automatically_accepted':False}
    process_receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'started':LABEL,'pid':process.pid}),flush=True)
    rc=process.wait();receipt['exit_code']=rc;process_receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'label':LABEL,'exit_code':rc}),flush=True)
    raise SystemExit(rc)

if __name__=='__main__':main()
