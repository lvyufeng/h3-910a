/* Minimal C ABI for launching a prebuilt flash-attention kernel on 910A.
 * Python calls this via ctypes with device pointers taken from torch_npu
 * tensors, which avoids needing a pybind11 build. */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "acl/acl.h"
#include "rt_external_kernel.h"
#include "runtime/kernel.h"

static int g_inited = 0;

/* rtKernelLaunch resolves the function by pointer identity of the stub, so the
 * string passed to rtFunctionRegister must be the exact pointer reused at
 * launch. Keep a small table mapping the returned handle to it. */
#define FA_MAX 64
static struct { void* hdl; char* name; void* obj; } g_tab[FA_MAX];
static int g_tab_n = 0;

int fa910a_init(int force) {
    /* Inside a torch_npu process ACL is already up. Calling aclInit again
     * returns an error, so treat "already initialised" as success and only
     * force a device bind when the caller explicitly asks for it. */
    if (g_inited) return 0;
    aclError rc = aclInit(NULL);
    if (rc != ACL_SUCCESS && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        /* ACL_ERROR_REPEAT_INITIALIZE means someone else owns the context,
         * which is the normal case under torch_npu. */
        if (force) return -1;
    }
    if (force) {
        if (aclrtSetDevice(0) != ACL_SUCCESS) return -1;
    }
    g_inited = 1;
    return 0;
}

/* Register a .o and return an opaque handle, or NULL. */
void* fa910a_register(const char* obj_path, const char* kernel_name) {
    FILE* f = fopen(obj_path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    void* buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);

    rtDevBinary_t bin;
    bin.magic = RT_DEV_BINARY_MAGIC_ELF;
    bin.version = 0;
    bin.data = buf;
    bin.length = (uint64_t)n;
    void* hdl = NULL;
    if (rtDevBinaryRegister(&bin, &hdl) != 0) { free(buf); return NULL; }
    /* kernel_name must outlive the handle, so copy it */
    char* kn = (char*)malloc(strlen(kernel_name) + 1);
    strcpy(kn, kernel_name);
    if (rtFunctionRegister(hdl, kn, kn, NULL, 0) != 0) { free(kn); free(buf); return NULL; }
    if (g_tab_n < FA_MAX) {
        g_tab[g_tab_n].hdl = hdl;
        g_tab[g_tab_n].name = kn;
        g_tab[g_tab_n].obj = buf;
        g_tab_n++;
    }
    return hdl;
}

/* Launch with 5 GM pointers (q,k,v,o,rowsum) on an existing stream. */
int fa910a_launch(void* hdl, const char* kernel_name, uint32_t num_blocks,
                  void* q, void* k, void* v, void* o, void* l, void* stream) {
    (void)kernel_name;
    if (!hdl) return -1;
    const char* stub = NULL;
    for (int i = 0; i < g_tab_n; i++) {
        if (g_tab[i].hdl == hdl) { stub = g_tab[i].name; break; }
    }
    if (!stub) return -1;              /* unregistered handle */
    void* args[5] = { q, k, v, o, l };
    rtError_t e = rtKernelLaunch((const void*)stub, num_blocks, args,
                                 (uint32_t)sizeof(args), NULL, (rtStream_t)stream);
    return (int)e;
}

int fa910a_sync(void* stream) {
    aclError e = aclrtSynchronizeStream((aclrtStream)stream);
    return (int)e;
}

/* Current stream for the default device, as an opaque pointer. */
void* fa910a_default_stream(void) {
    aclrtStream s = NULL;
    if (aclrtCreateStream(&s) != ACL_SUCCESS) return NULL;
    return (void*)s;
}
