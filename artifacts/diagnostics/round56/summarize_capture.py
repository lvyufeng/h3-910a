#!/usr/bin/env python3
"""Verify actual late capture context and repeatability against round54 tensors."""
import json
from pathlib import Path
import torch

ROOT=Path('/tmp/h3_round56_audit01');CAP=Path('/tmp/h3_round56_capture01');OLD=Path('/tmp/h3_round54_capture01');OUT=ROOT/'capture_summary.json'
if OUT.exists():raise RuntimeError('Refuse capture summary overwrite')
new=torch.load(CAP/'text_input.pt',weights_only=True);old=torch.load(OLD/'text_inputs.pt',weights_only=True)
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
r={'context':new['context'],'inputs_embeds_shape':list(new['args'][0].shape),'inputs_embeds_dtype':str(new['args'][0].dtype),
   'input_embeddings_bits_equal_round54':exact(new['args'][0],old['inputs_embeds']),
   'positions_bits_equal_round54':exact(new['args'][1],old['positions']),
   'deepstack_bits_equal_round54':[exact(a,b) for a,b in zip(new['kwargs']['deepstack_visual_embeds'],old['deepstack_visual_embeds'])]}
a=torch.load(CAP/'layer49_output.pt',weights_only=True)['output'];b=torch.load(OLD/'text_layer_49.pt',weights_only=True)
r['last_output_bits_equal_round54']=exact(a,b);r['last_output_max_abs_difference_round54']=float((a.float()-b.float()).abs().max())
r['last_context']=torch.load(CAP/'layer49_input.pt',weights_only=True)['context']
r['scope']='Actual runtime context and same-input repeatability; one-step capture is not generation-quality benchmark'
OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
