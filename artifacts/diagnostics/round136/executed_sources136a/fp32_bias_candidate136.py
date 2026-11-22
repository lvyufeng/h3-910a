"""R136 approved full49 diagnostic only: same Half operands -> FP32 out/bias -> Half.

Same R134/135 arithmetic, allowing the original first-step two time rows and
remaining three time rows. Row count follows original inputs, never padding,
subsetting, or replacing weights. Rounding changes are measured, not adopted.
"""


class ExplicitFP32Bias136:
    def __init__(self,torch,target):
        self.torch=torch;self.target=target;self.calls=0;self.last_contract=None;self.row_count_calls={}

    def apply(self,layer,x,bias=None):
        torch=self.torch
        assert layer is self.target and x.ndim==2 and x.shape[0] in (2,3) and x.shape[1]==2688 and layer.weight.shape==(5376,2688)
        assert x.dtype==layer.weight.dtype==torch.float16
        assert bias is not None and bias.dtype==torch.float16 and bias.shape==(5376,)
        assert x.device==layer.weight.device==bias.device and x.device.type=='npu'
        output=torch.mm(x,layer.weight.t(),out_dtype=torch.float32)
        assert output.dtype==torch.float32 and output.shape==(x.shape[0],5376)
        output.add_(bias.float())
        result=output.to(torch.float16)
        self.calls+=1;self.row_count_calls[str(x.shape[0])]=self.row_count_calls.get(str(x.shape[0]),0)+1
        self.last_contract={'operand_dtype':str(x.dtype),'weight_dtype':str(layer.weight.dtype),
            'GEMM_output_dtype':str(output.dtype),'bias_add_dtype':'torch.float32','returned_dtype':str(result.dtype),
            'input_shape':list(x.shape),'local_output_shape':list(result.shape),'same_parameters_no_rebinding':True,
            'one_final_narrowing':True,'arithmetic_rounding_changes_are_not_accepted':True}
        return result
