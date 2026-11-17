"""Vision attention input-cast factor only; QK/scale/softmax/PV remain original."""
import torch
import torch_npu
from vision_cast_guard73 import cast


class CastAttention:
    def __init__(self,original,metric,tensor_identity):
        self.original,self.metric,self.identity=original,metric,tensor_identity
        self.enabled=False;self.verify=False;self.calls=0;self.checks=[]

    @staticmethod
    def layout(value):
        return {'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),'physical_format':torch_npu.get_npu_format(value)}

    def forward(self,q,k,v):
        index=self.calls;self.calls+=1
        if not self.enabled:return self.original(q,k,v)
        before=self.identity({'q':q,'k':k,'v':v})if self.verify else None
        converted=[];proofs=[]
        for name,value in zip(('q','k','v'),(q,k,v)):
            candidate=cast(value)
            if self.verify:
                reference=value.float().contiguous()
                proof=self.metric(candidate,reference)
                row={'operand':name,'input_layout':self.layout(value),'storage_offset':value.storage_offset(),
                     'input_storage_bytes':value.untyped_storage().nbytes(),'proof':proof,
                     'candidate_layout':self.layout(candidate),'native_layout':self.layout(reference)}
                proofs.append(row)
                if not proof['raw_bits_equal']or not proof['finite']or row['candidate_layout']!=row['native_layout']:raise AssertionError('Actual vision cast bits or native contiguous layout changed')
            converted.append(candidate)
        qf,kf,vf=converted
        scores=torch.matmul(qf,kf.transpose(-2,-1))*(q.shape[-1]**-0.5)
        probabilities=torch.softmax(scores,dim=-1)
        output=torch.matmul(probabilities,vf).to(q.dtype)
        if self.verify:
            reference=self.original(q,k,v)
            proof=self.metric(output,reference)
            row={'call':index,'layer':index//2,'image_segment':index%2,'input_cast_proofs':proofs,
                 'attention_output':proof,'output_layout':self.layout(output),'original_output_layout':self.layout(reference),
                 'input_values_readonly':before==self.identity({'q':q,'k':k,'v':v})}
            self.checks.append(row)
            if not proof['raw_bits_equal']or not proof['finite']or not row['input_values_readonly']or row['output_layout']!=row['original_output_layout']:raise AssertionError('Actual complete attention output/layout/input values changed')
        return output
