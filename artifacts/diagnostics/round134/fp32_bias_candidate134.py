"""R134 local-test-only Half operands -> FP32 output/bias -> one Half narrowing.

Changes original F.linear rounding/bias path. Not adopted precision policy,
not automatically lossless, never installed in the formal full-model worker.
"""


class ExplicitFP32Bias134:
    def __init__(self,torch,target):
        self.torch=torch;self.target=target;self.calls=0;self.last_contract=None

    def apply(self,layer,x,bias=None):
        torch=self.torch
        assert layer is self.target and x.shape==(3,2688) and layer.weight.shape==(5376,2688)
        assert x.dtype==layer.weight.dtype==torch.float16
        assert bias is not None and bias.dtype==torch.float16 and bias.shape==(5376,)
        assert x.device==layer.weight.device==bias.device and x.device.type=='npu'
        output=torch.mm(x,layer.weight.t(),out_dtype=torch.float32)
        assert output.dtype==torch.float32 and output.shape==(3,5376)
        output.add_(bias.float())
        result=output.to(torch.float16)
        self.calls+=1
        self.last_contract={'operand_dtype':str(x.dtype),'weight_dtype':str(layer.weight.dtype),
            'GEMM_output_dtype':str(output.dtype),'bias_add_dtype':'torch.float32','returned_dtype':str(result.dtype),
            'input_shape':list(x.shape),'local_output_shape':list(result.shape),'same_parameters_no_rebinding':True,
            'one_final_narrowing':True,'arithmetic_rounding_changes_are_not_accepted':True}
        return result
