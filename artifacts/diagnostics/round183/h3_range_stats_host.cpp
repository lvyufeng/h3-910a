#include "../op_kernel/h3_range_stats_tiling.h"
#include "register/op_def_registry.h"

// One output slot per core: each core reduces its slice and writes its own maximum there,
// and the caller takes the maximum across the slots. The count is a fixed constant because
// InferShape and TilingFunc must agree on it exactly.
static constexpr int64_t H3_CORE_NUM = 8;
// each core publishes into its own aligned 32-byte region (see the kernel)
static constexpr int64_t H3_OUT_STRIDE = 8;
static constexpr int64_t H3_OUT_ELEMS = H3_CORE_NUM * H3_OUT_STRIDE;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    H3RangeStatsTilingData *tiling = context->GetTilingData<H3RangeStatsTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    tiling->size = data_sz;
    // the kernel reads this to split the tensor; the generated skeleton left it unset
    tiling->coreNum = (unsigned int)H3_CORE_NUM;
    context->SetBlockDim((uint32_t)H3_CORE_NUM);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // the generated skeleton copied the input shape, which is wrong for a reduction:
    // the output is one int32 per core
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, H3_OUT_ELEMS);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // the skeleton propagated the input dtype; the reduction result is int32
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class H3RangeStats : public OpDef {
public:
    explicit H3RangeStats(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
    }
};
OP_ADD(H3RangeStats);
}
