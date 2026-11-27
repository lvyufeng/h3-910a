"""Combined forms on both slow GEMM shapes, with exact-equality checks."""
import torch, torch_npu, statistics

def bench(fn, n=4):
    fn(); torch.npu.synchronize()
    ts=[]
    for _ in range(n):
        s,e=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.npu.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

M=10448
for K,N,label in [(5376,14336,"fc1"),(5376,10752,"qkv")]:
    x=torch.randn(M,K,dtype=torch.float16,device="npu")*0.05
    w=torch.randn(N,K,dtype=torch.float16,device="npu")*0.05
    ref=torch.mm(x,w.t())
    wt_nz=torch_npu.npu_format_cast(w.t().contiguous(),2)
    flops=2*M*K*N
    forms={
      "baseline fp16": lambda: torch.mm(x,w.t()),
      "NZ w + fp16": lambda: torch.mm(x,wt_nz),
      "fp32out+cast": lambda: torch.mm(x,w.t(),out_dtype=torch.float32).half(),
      "NZ w + fp32out+cast": lambda: torch.mm(x,wt_nz,out_dtype=torch.float32).half(),
    }
    print(f"\n=== {label}: M={M} K={K} N={N} ===")
    for name,fn in forms.items():
        try:
            t=bench(fn); out=fn()
            if out.dtype==torch.float16:
                same=bool(torch.equal(out,ref))
                d=(out.float()-ref.float()).abs(); rel=float(d.norm()/ref.float().norm())
                chk=f" bit_equal={same} relRMS={rel:.2e}"
            else:
                chk=""
            print(f"  {name:22s} {t:7.2f} ms  {flops/(t/1e3)/1e12:6.1f} TFLOPS{chk}")
        except Exception as e:
            print(f"  {name:22s} FAILED {str(e).splitlines()[0][:70]}")
