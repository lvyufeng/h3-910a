"""Grid over (K,N) at M=10448 to find which shapes unlock the fast kernel."""
import torch, torch_npu, statistics

def bench(fn, n=4):
    fn(); torch.npu.synchronize()
    ts=[]
    for _ in range(n):
        s,e=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.npu.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

M=10448
def run(K,N,fp32out):
    x=torch.randn(M,K,dtype=torch.float16,device="npu")*0.05
    w=torch.randn(N,K,dtype=torch.float16,device="npu")*0.05
    fn=(lambda: torch.mm(x,w.t(),out_dtype=torch.float32)) if fp32out else (lambda: torch.mm(x,w.t()))
    t=bench(fn); f=2*M*K*N
    print(f"K={K:5d} N={N:6d} out={'fp32' if fp32out else 'fp16'}  {t:8.2f} ms  {f/(t/1e3)/1e12:6.1f} TFLOPS")

for K,N in [(3584,5376),(5376,5376),(3584,10752),(3584,14336),(5376,10752),(5376,14336),(7168,5376),(7168,7168)]:
    run(K,N,True)
print()
for K,N in [(3584,5376),(5376,5376),(3584,14336),(5376,14336)]:
    run(K,N,False)
print("\n--- chunked fc1: N=14336 as chunks (fp32 out) ---")
K=5376; N=14336
x=torch.randn(M,K,dtype=torch.float16,device="npu")*0.05
w=torch.randn(N,K,dtype=torch.float16,device="npu")*0.05
for chunk in [14336,7168,5376,3584,1792]:
    n_chunks=N//chunk
    def fn(chunk=chunk,n_chunks=n_chunks):
        outs=[torch.mm(x, w[i*chunk:(i+1)*chunk].t(), out_dtype=torch.float32) for i in range(n_chunks)]
        return torch.cat(outs, dim=1)
    t=bench(fn); f=2*M*K*N
    print(f"  chunk={chunk:6d} x{n_chunks:2d}  {t:8.2f} ms  {f/(t/1e3)/1e12:6.1f} TFLOPS")
