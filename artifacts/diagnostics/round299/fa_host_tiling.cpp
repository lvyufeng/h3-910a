
#include "../op_kernel/flash_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "/home/mseco/Ascend/cann-9.2.0/asc/include/adv_api/matmul/bmm_tiling.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  FlashAttentionCustomTilingData *tiling = context->GetTilingData<FlashAttentionCustomTilingData>();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling->size = data_sz;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  matmul_tiling::MultiCoreMatmulTiling t(platform);
  t.SetDim(1);
  t.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
  t.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
  t.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
  t.SetShape(64, 512, 128);
  t.SetOrgShape(64, 512, 128);
  t.SetBias(false);
  if (t.GetTiling(tiling->mm) == -1) { return ge::GRAPH_FAILED; }
  context->SetBlockDim(1);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 64 * 512 * 4 + platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class FlashAttentionCustom : public OpDef {
public:
    explicit FlashAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(FlashAttentionCustom);
}
