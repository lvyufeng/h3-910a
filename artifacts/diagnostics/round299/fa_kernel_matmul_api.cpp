#include "kernel_operator.h"
#include "flash_attention_custom_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

constexpr int32_t kM = 64, kN = 512, kK = 128;

extern "C" __global__ __aicore__ void flash_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FlashAttentionCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0) { return; }
    TPipe pipe;
    using aType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<aType, bType, cType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.mm);
    GlobalTensor<half> qGm, kGm, oGm;
    GlobalTensor<float> tGm;
    qGm.SetGlobalBuffer((__gm__ half *)q, kM * kK);
    kGm.SetGlobalBuffer((__gm__ half *)k, kN * kK);
    oGm.SetGlobalBuffer((__gm__ half *)o, kM * kN);
    tGm.SetGlobalBuffer((__gm__ float *)workspace, kM * kN);
    mm.SetTensorA(qGm);
    mm.SetTensorB(kGm);
    mm.IterateAll(tGm);
    TQue<TPosition::VECIN, 1> inQ;
    TQue<TPosition::VECOUT, 1> outQ;
    pipe.InitBuffer(inQ, 1, kM * kN * 4);
    pipe.InitBuffer(outQ, 1, kM * kN * 2);
    LocalTensor<float> cf = inQ.AllocTensor<float>();
    DataCopy(cf, tGm, kM * kN);
    LocalTensor<half> ch = outQ.AllocTensor<half>();
    Cast(ch, cf, RoundMode::CAST_NONE, kM * kN);
    DataCopy(oGm, ch, kM * kN);
}
