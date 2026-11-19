#!/usr/bin/env python3
"""Execute the original-parameter vision position-only precision probe."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve(strict=True)
    from evidence_io112 import EvidenceWriter
    writer=EvidenceWriter(root/'position_probe112.json');receipt={'scope':__doc__,'completed':False,'full_encoder_precision_accepted':False,'whole_goal_complete':False}
    writer.write(receipt)
    try:
        import torch
        import torch_npu
        import transformers
        from transformers.vision_utils import get_vision_bilinear_indices_and_weights
        from safetensors import safe_open
        import vision_position112 as probe
        assert torch.get_default_dtype()==torch.float32
        torch.npu.set_device(0);torch.set_num_threads(2)
        checkpoint=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
        assert sha(checkpoint/'config.json')=='d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427'
        config=json.loads((checkpoint/'config.json').read_text())['vision_config']
        assert (config['hidden_size'],config['num_position_embeddings'],config['spatial_merge_size'])==(1152,2304,2)
        candidate=root/'candidate_encoder79.py';assert sha(candidate)=='2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78'
        hf_source=Path(get_vision_bilinear_indices_and_weights.__code__.co_filename)
        assert sha(hf_source)=='23828876bb2fe9700d0d68a3374cf9e38086964f97375c9d46d376ab85b265de'
        reference_path=Path('/tmp/h3_round78_hf_fp1601/reference.json')
        assert sha(reference_path)=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        reference=json.loads(reference_path.read_text());pin=reference['parameter_conversion']['visual.pos_embed.weight']
        index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
        names=[name for name in index if name.endswith('visual.pos_embed.weight')];assert len(names)==1
        with safe_open(str(checkpoint/index[names[0]]),framework='pt',device='cpu')as f:
            original=f.get_tensor(names[0]);assert original.dtype==torch.bfloat16
            weight=original.to(torch.float16).contiguous()
        assert probe.info(weight)['sha256']==pin['reference_sha256']
        receipt['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'transformers':transformers.__version__}
        receipt['original_source_pins']={'candidate':sha(candidate),'HF_vision_utils':sha(hf_source),'R78_reference':sha(reference_path)}
        receipt['actual_vision_config']=config
        grid=torch.tensor([[1,48,84],[1,48,84]],dtype=torch.int64)
        receipt['result']=probe.probe(weight.to('npu'),grid,get_vision_bilinear_indices_and_weights,candidate)
        assert sha(candidate)==receipt['original_source_pins']['candidate']and sha(hf_source)==receipt['original_source_pins']['HF_vision_utils']
        assert sha(reference_path)==receipt['original_source_pins']['R78_reference']
        receipt['completed']=True;writer.write(receipt)
    except BaseException as error:
        receipt['failure']={'type':type(error).__name__,'message':str(error)};writer.write(receipt);raise


if __name__=='__main__':main()
