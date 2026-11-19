#!/usr/bin/env python3
"""Package exact original BF16 weights/fixtures for GPU reference; I/O only, no inference."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--kind',choices=['projection','conditioning','encoder'],required=True);args=parser.parse_args()
    root=args.root.absolute();assert not root.exists();root.mkdir(parents=True);root=root.resolve(strict=True)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    base=Path('/home/mseco/models/MiniMax-H3/FL2VA')
    reference_path=Path('/tmp/h3_round78_hf_fp1601/reference.json')
    assert sha(reference_path)=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
    reference=json.loads(reference_path.read_text())
    if args.kind=='encoder':
        checkpoint=base/'text_encoder';wanted=list(reference['parameter_conversion'])
        source_names={name:'model.'+name for name in wanted};expected=reference['parameter_conversion']
        assert len(wanted)==903
        expected_original_aggregate='a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e'
    else:
        checkpoint=base/'transformer';control_pins=json.loads(Path('/tmp/h3_round114_transfer01/expected_half_parameters111.json').read_text());expected=control_pins['parameters']
        wanted=['condition_proj.weight','condition_proj.bias']if args.kind=='projection'else list(expected)
        source_names={name:name for name in wanted};expected_original_aggregate=None
    config=checkpoint/'config.json';index_path=checkpoint/'model.safetensors.index.json'
    index=json.loads(index_path.read_text())['weight_map']
    if args.kind=='encoder':
        assert sha(config)=='d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427'
        assert sha(index_path)=='06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a'
    else:
        assert sha(config)==control_pins['checkpoint_config_SHA256']and sha(index_path)==control_pins['checkpoint_index_SHA256']
    shutil.copy2(config,root/'config.json')
    manifest={'scope':__doc__,'kind':args.kind,'io_packaging_only':True,'no_cpu_reference_inference':True,
              'original_checkpoint':str(checkpoint),'config_SHA256':sha(config),'source_index_SHA256':sha(index_path),
              'R78_reference_SHA256':sha(reference_path),'parameters':{},'files':{},'source_shards':{},'completed':False}
    aggregate=hashlib.sha256();pending={};size=0;part=0;max_bytes=2*1024**3
    def publish():
        nonlocal pending,size,part
        part+=1;filename='original_bf16_%s_%02d.safetensors'%(args.kind,part);path=root/filename
        assert not path.exists();save_file(pending,str(path),metadata={'source':'original undistilled MiniMaxAI/MiniMax-H3','dtype':'original BF16; never FP16-roundtripped'})
        manifest['files'][filename]={'bytes':path.stat().st_size,'sha256':sha(path)}
        for name in pending:manifest['parameters'][name]['packed_file']=filename
        pending={};size=0
    for name in wanted:
        source=source_names[name];shard=index[source];path=checkpoint/shard
        if shard not in manifest['source_shards']:
            digest=sha(path)
            if args.kind=='encoder':assert digest==reference['checkpoint_shards'][shard]['sha256']
            else:assert digest==control_pins['source_checkpoint_firstshard_sha256']
            manifest['source_shards'][shard]={'sha256':digest,'bytes':path.stat().st_size}
        with safe_open(str(path),framework='pt',device='cpu')as f:value=f.get_tensor(source)
        assert value.dtype==torch.bfloat16 and list(value.shape)==expected[name]['shape'],name
        payload=value.contiguous().view(torch.uint8).numpy()
        bf_sha=hashlib.sha256(memoryview(payload)).hexdigest();aggregate.update(name.encode());aggregate.update(memoryview(payload))
        n=value.numel()*2
        if pending and size+n>max_bytes:publish()
        pending[name]=value.contiguous();size+=n
        manifest['parameters'][name]={'shape':list(value.shape),'original_dtype':'torch.bfloat16','original_BF16_SHA256':bf_sha,
                                     'FP16_converted_SHA256':expected[name]['reference_sha256'],'source_key':source,'source_shard':shard}
    if pending:publish()
    manifest['original_parameter_aggregate_SHA256']=aggregate.hexdigest()
    if expected_original_aggregate is not None:assert aggregate.hexdigest()==expected_original_aggregate
    if args.kind=='projection':
        inputs=root/'inputs';inputs.mkdir()
        pins=json.loads(Path('/tmp/h3_round114_transfer01/expected_payload_inputs111.json').read_text())
        manifest['precision_inputs']={}
        for name,pin in pins['inputs'].items():
            path=Path(pin['path']);assert sha(path)==pin['file_SHA256'];dst=inputs/path.name;shutil.copy2(path,dst)
            assert sha(dst)==pin['file_SHA256'];manifest['precision_inputs'][name]={**pin,'relative_path':'inputs/'+dst.name}
        fixture=Path('/tmp/h3_round54_capture01/encoder_input.pt')
        assert sha(fixture)=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
        shutil.copy2(fixture,inputs/'encoder_input.pt')
        manifest['encoder_fixture']={'relative_path':'inputs/encoder_input.pt','file_SHA256':sha(fixture)}
        shutil.copy2(reference_path,root/'R78_reference.json')
    manifest['completed']=True
    with(root/'package_manifest.json').open('x')as f:f.write(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'completed':True,'kind':args.kind,'root':str(root),'tensor_count':len(wanted),'packed_bytes':sum(x['bytes']for x in manifest['files'].values())},indent=2),flush=True)


if __name__=='__main__':main()
