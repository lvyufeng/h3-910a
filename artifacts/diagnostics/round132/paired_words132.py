"""Whole-range CPU FP32 word comparison in bounded chunks, not CPU model inference."""
import math


def compare_fp32(actual,reference,torch,*,chunk_words=262144):
    import numpy as np
    assert actual.device.type==reference.device.type=='cpu'
    assert actual.dtype==reference.dtype==torch.float32 and actual.shape==reference.shape
    a=actual.contiguous().numpy().reshape(-1);b=reference.contiguous().numpy().reshape(-1)
    au=a.view(np.uint32);bu=b.view(np.uint32)
    different=0
    for start in range(0,a.size,chunk_words):different+=int(np.count_nonzero(au[start:start+chunk_words]!=bu[start:start+chunk_words]))
    max_abs=0.;delta_sq=0.;reference_sq=0.
    if different:
        for start in range(0,a.size,chunk_words):
            x=a[start:start+chunk_words].astype(np.float64);y=b[start:start+chunk_words].astype(np.float64);d=x-y
            if d.size:max_abs=max(max_abs,float(np.abs(d).max()))
            delta_sq+=float(np.dot(d,d));reference_sq+=float(np.dot(y,y))
    relative=math.sqrt(delta_sq)/max(math.sqrt(reference_sq),1e-30) if different else 0.
    return {'shape':list(actual.shape),'dtype':str(actual.dtype),'total_FP32_words':int(a.size),'different_FP32_words':different,
        'all_words_equal':different==0,'max_abs_difference':max_abs,'relative_RMS_ratio':relative,'relative_RMS_percent':100*relative,
        'complete_logical_word_range_compared':True,'chunk_words':chunk_words}
