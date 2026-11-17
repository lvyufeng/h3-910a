"""Independent original/wrapped/profile/restored control and full54-call transcription proof."""
import torch
import torch_npu
from attention_stages73 import AttentionStages


class AttentionControls(AttentionStages):
    def __init__(self,encoder,candidate,root,rank,tensor_identity,metric):
        super().__init__(encoder,candidate,root,rank,tensor_identity)
        self.metric=metric;self.verify=False;self.checks=[]

    def route(self,original=False):
        self.candidate._vision_sdpa_bf16_compatible=self.original if original else self.forward

    def forward(self,q,k,v):
        index=self.call_count
        output=super().forward(q,k,v)
        if self.verify:
            reference=self.original(q,k,v)
            proof=self.metric(output,reference)
            candidate_layout={'shape':list(output.shape),'stride':list(output.stride()),'dtype':str(output.dtype),'physical_format':torch_npu.get_npu_format(output)}
            original_layout={'shape':list(reference.shape),'stride':list(reference.stride()),'dtype':str(reference.dtype),'physical_format':torch_npu.get_npu_format(reference)}
            row={'call':index,'layer':index//2,'image_segment':index%2,'proof':proof,
                 'output_layout':candidate_layout,'original_output_layout':original_layout}
            self.checks.append(row)
            if not proof['raw_bits_equal'] or not proof['finite'] or candidate_layout!=original_layout:
                raise AssertionError('Transcribed actual attention math changed from original function')
        return output
