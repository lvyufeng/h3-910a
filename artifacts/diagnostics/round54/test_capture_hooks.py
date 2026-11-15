#!/usr/bin/env python3
"""Capture hooks must preserve outputs, including subsequent write-once skips."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent

def load():
    s=importlib.util.spec_from_file_location('tested_audit',ROOT/'h3_910a_audit.py')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m

class Rotary(nn.Module):
    def forward(self,x,p): return (p.float(),p.float())
class Layer(nn.Module):
    def forward(self,x): return x+1
class Text(nn.Module):
    def __init__(self):
        super().__init__(); self.embed_tokens=nn.Embedding(32,8); self.rotary_emb=Rotary()
        self.layers=nn.ModuleList([Layer() for _ in range(50)])
    def forward(self,x,p,**kw):
        self.rotary_emb(x,p)
        for layer in self.layers:x=layer(x)
        return x
class Vision(nn.Module):
    def forward(self,x): return x,[x+1,x+2]

def execute(m,enabled,root):
    enc=types.SimpleNamespace(text_model=Text(),vision=Vision())
    ids=torch.tensor([1,2,3]); pos=torch.arange(3).reshape(1,1,3).expand(3,1,3)
    with patch.dict(os.environ,{'VLLM_OMNI_910A_AUDIT_DIR':str(root) if enabled else ''}),patch.object(m,'_rank_zero',return_value=True):
        m.install_encoder_hooks(enc,ids,{})
        enc.vision(torch.ones(2,8))
        embedded=enc.text_model.embed_tokens(ids.unsqueeze(0))
        a=enc.text_model(embedded,pos)
        b=enc.text_model(embedded*2,pos)
        ra=embedded.clone(); rb=embedded*2
        for _ in range(50):ra=ra+1; rb=rb+1
        assert torch.equal(a,ra) and torch.equal(b,rb)
        if enabled:
            old=(root/'text_final.pt').read_bytes()
            assert not m.save_once('text_final',b)
            assert (root/'text_final.pt').read_bytes()==old
            saved=torch.load(root/'text_inputs.pt',weights_only=True)
            assert torch.equal(saved['inputs_embeds'],embedded)
            assert not saved['inputs_embeds'].requires_grad
            for name in ('embedding','text_inputs','text_layer_00','text_layer_01','text_layer_02','text_layer_09','text_layer_24','text_layer_49','text_final','text_rotary','encoder_input','vision_outputs'):
                assert (root/(name+'.json')).exists() and (root/(name+'.pt')).exists()
            m._MAX_FILE_BYTES=1
            try:m.save_once('text_layer_03',torch.ones(2))
            except RuntimeError:pass
            else:raise AssertionError('byte cap was not enforced')
        else:
            assert not list(root.iterdir())

with tempfile.TemporaryDirectory(prefix='h3_round54_capture_test_') as work:
    root=Path(work).resolve(); assert root.name.startswith('h3_round54_capture_test_')
    execute(load(),False,root)
    execute(load(),True,root)
    assert root.resolve()==root and root.name.startswith('h3_round54_capture_test_')
report={'passed':True,'default_disabled_no_writes':True,'first_and_second_hooked_outputs_unchanged':True,
        'write_once_skip_does_not_replace_module_output':True,'captured_tensors_detached':True,'bounded_bytes':True}
(ROOT/'capture_hook_tests.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
