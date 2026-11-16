#!/usr/bin/env python3
"""Portable 65-registration regression with fake ACL/runtime, no hardware."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent


class LauncherCapacity(unittest.TestCase):
    def test_65th_kernel_fails_before_runtime_registration(self):
        cc=shutil.which('cc')
        if not cc:self.skipTest('Portable C compiler unavailable')
        target=Path(tempfile.mkdtemp(prefix='h3_vec61_stub_')).resolve()
        self.assertTrue(target.is_absolute() and target.name.startswith('h3_vec61_stub_'))
        try:
            (target/'acl').mkdir();(target/'runtime').mkdir()
            (target/'acl'/'acl.h').write_text('''#include <stdint.h>
typedef int aclError;
typedef void* aclrtStream;
#define ACL_SUCCESS 0
#define ACL_ERROR_REPEAT_INITIALIZE 1
int aclInit(void* p);
int aclrtSetDevice(int p);
int aclrtSynchronizeStream(aclrtStream s);
int aclrtCreateStream(aclrtStream* s);
''')
            (target/'rt_external_kernel.h').write_text('''#include <stdint.h>
typedef int rtError_t;
typedef void* rtStream_t;
typedef struct {int magic;int version;void* data;uint64_t length;} rtDevBinary_t;
#define RT_DEV_BINARY_MAGIC_ELF 42
''')
            (target/'runtime'/'kernel.h').write_text('''int rtDevBinaryRegister(const rtDevBinary_t* b,void** h);
int rtFunctionRegister(void* h,const void* f,const char* n,void* d,int t);
int rtKernelLaunch(const void* f,uint32_t blocks,void* args,uint32_t size,void* smem,rtStream_t stream);
''')
            (target/'fake.c').write_text('''#include "acl/acl.h"
#include "rt_external_kernel.h"
#include "runtime/kernel.h"
static uintptr_t calls=0;
int aclInit(void* p){return 0;}
int aclrtSetDevice(int p){return 0;}
int aclrtSynchronizeStream(aclrtStream s){return 0;}
int aclrtCreateStream(aclrtStream* s){*s=(void*)1;return 0;}
int rtDevBinaryRegister(const rtDevBinary_t* b,void** h){*h=(void*)(++calls);return 0;}
int rtFunctionRegister(void* h,const void* f,const char* n,void* d,int t){return 0;}
int rtKernelLaunch(const void* f,uint32_t blocks,void* args,uint32_t size,void* smem,rtStream_t stream){return 0;}
int fake_calls(void){return (int)calls;}
''')
            so=target/'stub.so'
            p=subprocess.run([cc,'-shared','-fPIC','-I'+str(target),str(ROOT/'vec61_launcher.c'),str(target/'fake.c'),'-o',str(so)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            lib=ctypes.CDLL(str(so));lib.fa910a_register.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.fa910a_register.restype=ctypes.c_void_p
            lib.fake_calls.restype=ctypes.c_int
            obj=target/'fake.o';obj.write_bytes(b'ELF-stub')
            handles=[lib.fa910a_register(str(obj).encode(),('stub%d'%i).encode()) for i in range(64)]
            self.assertTrue(all(handles));self.assertEqual(len(set(handles)),64)
            self.assertEqual(lib.fake_calls(),64)
            self.assertIsNone(lib.fa910a_register(str(obj).encode(),b'stub64'))
            self.assertEqual(lib.fake_calls(),64)
        finally:
            if target.resolve()!=target or not target.name.startswith('h3_vec61_stub_'):
                raise RuntimeError('Unexpected stub cleanup path')
            shutil.rmtree(target)


if __name__=='__main__':unittest.main(verbosity=2)
