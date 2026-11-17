"""Read-only numeric proof helpers; private FP32 vs BF16 interfaces are distinct."""
import hashlib
import time
import torch


def raw(x):return hashlib.sha256(memoryview(x.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def metric(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    x,y=a.detach().cpu().contiguous().reshape(-1),b.detach().cpu().contiguous().reshape(-1)
    different=0;err=0.;ref=0.;maximum=0.;finite=True
    word=torch.int32 if a.dtype==torch.float32 else torch.int16
    for s in range(0,x.numel(),1048576):
        xx,yy=x[s:s+1048576],y[s:s+1048576];different+=int((xx.view(word)!=yy.view(word)).sum())
        xd,yd=xx.double(),yy.double();valid=bool(torch.isfinite(xd).all()and torch.isfinite(yd).all());finite=finite and valid
        if valid:
            e=xd-yd;err+=float(e.square().sum());ref+=float(yd.square().sum());maximum=max(maximum,float(e.abs().max()))
    return {'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.numel(),'different_words':different,'raw_bits_equal':different==0,
            'finite':finite,'max_abs_error':maximum if finite else None,'relative_rms':(err/max(ref,1e-30))**.5 if finite else None}


def measure(fn):
    torch.npu.synchronize();start=time.perf_counter();out=fn();torch.npu.synchronize()
    return out,time.perf_counter()-start
