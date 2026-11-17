/* Minimal C ABI for launching the Round82 prototype GEMM kernel on 910A.
 *
 * Same launch path already proven on this chip in
 * artifacts/design/patches/fa910a_lib.c and documented in
 * artifacts/design/ascendc_flash_attn/KERNEL_RUNS.md:
 *
 *   rtDevBinaryRegister -> rtFunctionRegister -> rtKernelLaunch
 *
 * The aclrtBinary* family (aclrtBinaryLoadFromFile / aclrtBinaryGetFunction /
 * aclrtLaunchKernelWithHostArgs) loads and submits without error and then dies
 * with ACL_ERROR_RT_AICORE_EXCEPTION (507015) on the very same .o, so it is
 * deliberately not used here.
 *
 * Python drives this through ctypes with device pointers taken from torch_npu
 * tensors, which avoids needing a pybind11 build.
 *
 * NOT COMPILED AND NOT EXECUTED as part of this round's evidence.
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "acl/acl.h"
#include "rt_external_kernel.h"
#include "runtime/kernel.h"

static int g_inited = 0;

/* rtKernelLaunch resolves the function by the pointer identity of the stub, so
 * the name registered by rtFunctionRegister must be the exact pointer reused at
 * launch. Keep a small table keyed by the returned handle. */
#define GEMM82_MAX 32
static struct { void* hdl; char* name; void* obj; } g_tab[GEMM82_MAX];
static int g_tab_n = 0;

/* Inside a torch_npu process ACL is already up; a second aclInit returns
 * ACL_ERROR_REPEAT_INITIALIZE, which is success for our purposes. */
int gemm82_init(int force) {
    if (g_inited) return 0;
    aclError rc = aclInit(NULL);
    if (rc != ACL_SUCCESS && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        if (force) return -1;
    }
    if (force) {
        if (aclrtSetDevice(0) != ACL_SUCCESS) return -1;
    }
    g_inited = 1;
    return 0;
}

/* Register a kernel_meta/<name>.o and return an opaque handle, or NULL. */
void* gemm82_register(const char* obj_path, const char* kernel_name) {
    if (!obj_path || !kernel_name) return NULL;
    if (g_tab_n >= GEMM82_MAX) return NULL;
    FILE* f = fopen(obj_path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n <= 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    void* buf = malloc((size_t)n);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);

    rtDevBinary_t bin;
    bin.magic = RT_DEV_BINARY_MAGIC_ELF;
    bin.version = 0;
    bin.data = buf;
    bin.length = (uint64_t)n;
    void* hdl = NULL;
    if (rtDevBinaryRegister(&bin, &hdl) != 0) { free(buf); return NULL; }

    char* kn = (char*)malloc(strlen(kernel_name) + 1);
    if (!kn) { free(buf); return NULL; }
    strcpy(kn, kernel_name);
    if (rtFunctionRegister(hdl, kn, kn, NULL, 0) != 0) { free(kn); free(buf); return NULL; }

    g_tab[g_tab_n].hdl = hdl;
    g_tab[g_tab_n].name = kn;
    g_tab[g_tab_n].obj = buf;
    g_tab_n++;
    return hdl;
}

static const char* gemm82_stub(void* hdl) {
    for (int i = 0; i < g_tab_n; i++) {
        if (g_tab[i].hdl == hdl) return g_tab[i].name;
    }
    return NULL;
}

/* No-bias variant: a fp16, b fp16, out (fp32 for fp32_out). */
int gemm82_launch3(void* hdl, uint32_t num_blocks, void* a, void* b, void* out,
                   void* stream) {
    const char* stub = gemm82_stub(hdl);
    if (!stub) return -1;
    void* args[3] = { a, b, out };
    rtError_t e = rtKernelLaunch((const void*)stub, num_blocks, args,
                                 (uint32_t)sizeof(args), NULL, (rtStream_t)stream);
    return (int)e;
}

/* Bias variant: a fp16, b fp16, bias fp32 (L1-resident operand), out fp16. */
int gemm82_launch4(void* hdl, uint32_t num_blocks, void* a, void* b, void* bias,
                   void* out, void* stream) {
    const char* stub = gemm82_stub(hdl);
    if (!stub) return -1;
    void* args[4] = { a, b, bias, out };
    rtError_t e = rtKernelLaunch((const void*)stub, num_blocks, args,
                                 (uint32_t)sizeof(args), NULL, (rtStream_t)stream);
    return (int)e;
}

int gemm82_sync(void* stream) {
    return (int)aclrtSynchronizeStream((aclrtStream)stream);
}

/* Return the current torch_npu stream so the launch is ordered with torch work.
 * The caller passes the stream pointer it already owns; this helper only exists
 * for the standalone case where no stream has been created yet. */
void* gemm82_default_stream(void) {
    aclrtStream s = NULL;
    if (aclrtCreateStream(&s) != ACL_SUCCESS) return NULL;
    return (void*)s;
}

/* Torch passes its validated npu_stream pointer directly;there is no guessed
 * C aclrtGetCurrentStream ABI in this launcher. */