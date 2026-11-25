// aclnn harness for the custom FlashAttentionCustom op (S1a-0: identity kernel o = q).
#include <acl/acl.h>
#include "aclnn_flash_attention_custom.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#define CK(x) do { aclError e_=(x); if(e_!=ACL_SUCCESS){printf("ACL_FAIL %d line %d\n",(int)e_,__LINE__);return 1;} } while(0)
int main(){
  const int64_t M=64, K=128; const size_t bytes=(size_t)M*K*2;
  CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
  aclrtStream st=nullptr; CK(aclrtCreateStream(&st));
  std::vector<__fp16> hq(M*K), hk(M*K), hv(M*K), ho(M*K,(__fp16)0.f);
  srand(11);
  for(size_t i=0;i<M*K;i++){ hq[i]=(__fp16)(((rand()%2000)-1000)/1000.0f); hk[i]=(__fp16)(((rand()%2000)-1000)/1000.0f); hv[i]=(__fp16)(((rand()%2000)-1000)/1000.0f); }
  void *dq,*dk,*dv,*dout;
  CK(aclrtMalloc(&dq,bytes,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dk,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
  CK(aclrtMalloc(&dv,bytes,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dout,bytes,ACL_MEM_MALLOC_HUGE_FIRST));
  CK(aclrtMemcpy(dq,bytes,hq.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));
  CK(aclrtMemcpy(dk,bytes,hk.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));
  CK(aclrtMemcpy(dv,bytes,hv.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));
  int64_t dims[2]={M,K}; int64_t strides[2]={K,1};
  aclTensor *tq=aclCreateTensor(dims,2,ACL_FLOAT16,strides,0,ACL_FORMAT_ND,dims,2,dq);
  aclTensor *tk=aclCreateTensor(dims,2,ACL_FLOAT16,strides,0,ACL_FORMAT_ND,dims,2,dk);
  aclTensor *tv=aclCreateTensor(dims,2,ACL_FLOAT16,strides,0,ACL_FORMAT_ND,dims,2,dv);
  aclTensor *to=aclCreateTensor(dims,2,ACL_FLOAT16,strides,0,ACL_FORMAT_ND,dims,2,dout);
  printf("tensors q=%p k=%p v=%p o=%p\n",(void*)tq,(void*)tk,(void*)tv,(void*)to);
  uint64_t ws=0; aclOpExecutor *ex=nullptr;
  int rc=(int)aclnnFlashAttentionCustomGetWorkspaceSize(tq,tk,tv,to,&ws,&ex);
  if(rc!=0){ printf("GWS rc=%d msg=%s\n",rc,aclGetRecentErrMsg()); return 1; }
  printf("workspace=%llu\n",(unsigned long long)ws);
  void *wsp=nullptr; if(ws>0) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
  rc=(int)aclnnFlashAttentionCustom(wsp,ws,ex,st);
  if(rc!=0){ printf("EXEC rc=%d msg=%s\n",rc,aclGetRecentErrMsg()); return 1; }
  CK(aclrtSynchronizeStream(st));
  CK(aclrtMemcpy(ho.data(),bytes,dout,bytes,ACL_MEMCPY_DEVICE_TO_HOST));
  float maxd=0; int bad=0;
  for(size_t i=0;i<M*K;i++){ float d=fabsf((float)ho[i]-(float)hq[i]); if(d>maxd)maxd=d; if(d>1e-3f)++bad; }
  printf("RESULT M=%lld K=%lld maxdiff_vs_q=%g bad=%d head=(%.4f,%.4f,%.4f)->%.4f\n",
         (long long)M,(long long)K,maxd,bad,(float)hq[0],(float)hk[0],(float)hv[0],(float)ho[0]);
  return bad==0?0:2;
}
