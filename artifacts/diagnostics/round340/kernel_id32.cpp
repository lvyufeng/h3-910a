#include "kernel_operator.h"
#include "flash_attention_custom_tiling.h"
using namespace AscendC;

constexpr int32_t kM = 32, kN = 32, kK = 32;
constexpr int32_t kElems = kM * kN;

extern "C" __global__ __aicore__ void flash_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FlashAttentionCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0) { return; }

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    TPipe pipe;
    matmul::Matmul<aType, bType, cType, biasType> mm;
    REGIST_MATMUL_OBJ(&pipe, (GM_ADDR)workspace, mm, &tilingData.mm);

    GlobalTensor<half> aGm, bGm;
    GlobalTensor<float> cGm;
    aGm.SetGlobalBuffer((__gm__ half *)q, kElems);
    bGm.SetGlobalBuffer((__gm__ half *)k, kElems);
    cGm.SetGlobalBuffer((__gm__ float *)((GM_ADDR)workspace + 16 * 1024 * 1024), kElems);

    mm.SetTensorA(aGm, false);
    mm.SetTensorB(bGm, false);
    mm.IterateAll(cGm, false);
    mm.End();

    // copy raw C (float) back to o as float->half
    TBuf<TPosition::VECCALC> fBuf, hBuf;
    pipe.InitBuffer(fBuf, kElems * 4);
    pipe.InitBuffer(hBuf, kElems * 2);
    LocalTensor<float> cf = fBuf.Get<float>();
    LocalTensor<half> ch = hBuf.Get<half>();
    GlobalTensor<half> oGm;
    oGm.SetGlobalBuffer((__gm__ half *)o, kElems);
    DataCopy(cf, cGm, kElems);
    PipeBarrier<PIPE_ALL>();
    Cast(ch, cf, RoundMode::CAST_NONE, kElems);
    PipeBarrier<PIPE_ALL>();
    DataCopy(oGm, ch, kElems);
}
