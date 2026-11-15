#!/usr/bin/env python3
"""Exercise the exact converter on 13 tiny shards, including BF16 underflow."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='h3_native_conversion_test_') as work:
    tmp=Path(work).resolve(); assert tmp.name.startswith('h3_native_conversion_test_')
    source=tmp/'source/FL2VA'; weights=source/'transformer'; weights.mkdir(parents=True)
    bf16=torch.tensor([1.,.25,1e-10],dtype=torch.bfloat16)
    f32=torch.tensor([.123456789,1e-10,70000.],dtype=torch.float32)
    f16=torch.tensor([1.,2.,3.],dtype=torch.float16)
    for i in range(1,14):
        save_file({'bf16':bf16,'f32':f32,'f16':f16},str(weights/f'model-{i:05d}-of-00013.safetensors'))
    (weights/'config.json').write_text('{}')
    (weights/'model.safetensors.index.json').write_text('{}')
    (source/'model_index.json').write_text('{}')
    native_meta=tmp/'native_metadata.json'
    native_meta.write_text(json.dumps({'_minimax_h3':{'partition':'fl2va','tasks':['fl2va']}}))
    dest=tmp/'destination'; alias=tmp/'alias'; report=tmp/'report.json'
    tree=ast.parse((ROOT/'convert_native_storage.py').read_text())
    replacements={
        '/home/mseco/models/MiniMax-H3/FL2VA':str(source),
        '/home/mseco/models/MiniMax-H3-native-fp16-audit53':str(dest),
        '/tmp/h3_round53_native49_fp16_model':str(alias),
        '/tmp/h3_round53_validation01/native_storage_conversion.json':str(report),
        '/tmp/h3_round53_native49_model/FL2VA/model_index.json':str(native_meta)}
    class Paths(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in replacements:
                return ast.copy_location(ast.Constant(value=replacements[node.value]),node)
            return node
    tree=Paths().visit(tree)
    with redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(tree),str(ROOT/'convert_native_storage.py'),'exec'),{'__name__':'__main__'})
    r=json.loads(report.read_text())
    assert r['complete'] and len(r['files'])==13
    assert r['total_bf16_roundtrip_changed_elements']==13
    for i in range(1,14):
        with safe_open(str(dest/'FL2VA/transformer'/f'model-{i:05d}-of-00013.safetensors'),framework='pt') as f:
            assert torch.equal(f.get_tensor('bf16'),bf16.half())
            assert f.get_tensor('bf16').dtype==torch.float16
            assert torch.equal(f.get_tensor('f32'),f32)
            assert f.get_tensor('f32').dtype==torch.float32
            assert torch.equal(f.get_tensor('f16'),f16)
    assert (alias/'FL2VA/transformer').resolve()==(dest/'FL2VA/transformer')
    assert (weights/'config.json').read_text()=='{}'
    summary={'passed':True,'tiny_shards_verified':13,'bf16_underflow_counted':13,
             'all_outputs_match_online_cast':True,'f32_and_existing_f16_unchanged':True,
             'native_alias_uses_converted_storage':True,'source_unchanged':True}
    assert tmp.resolve()==tmp and tmp.name.startswith('h3_native_conversion_test_')
(ROOT/'storage_test.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
