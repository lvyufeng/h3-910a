"""Read-only tensor/parameter fingerprints for native repeat diagnostics.

Hashing device tensors causes D2H synchronization, so leaf traces are expressly
nonclean and never latency comparisons. All bytes stay on-host; only scalar
fingerprints/metadata are returned. Metadata hashes are not tensor byte proofs.
"""
from __future__ import annotations
import hashlib
import json


def cpu_tensor_sha(tensor,torch):
    cpu=tensor.detach().to('cpu').contiguous()
    return hashlib.sha256(memoryview(cpu.reshape(-1).view(torch.uint8).numpy())).hexdigest()


def fingerprint(value,torch,*,include_pointers=False):
    if isinstance(value,torch.Tensor):
        try:version=value._version
        except RuntimeError:version=None  # Inference tensors deliberately have no version counter.
        row={'kind':'tensor','shape':list(value.shape),'dtype':str(value.dtype),'stride':list(value.stride()),
             'storage_offset':value.storage_offset(),'logical_bytes':value.numel()*value.element_size(),
             'device':str(value.device),'sha256_logical_words':cpu_tensor_sha(value,torch),'tensor_version':version}
        if include_pointers:row.update({'data_pointer':value.data_ptr(),'storage_base_pointer':value.untyped_storage().data_ptr(),
            'storage_bytes':value.untyped_storage().nbytes(),'storage_identity':value.untyped_storage()._cdata})
        return row
    if isinstance(value,dict):return {str(k):fingerprint(v,torch,include_pointers=include_pointers) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [fingerprint(v,torch,include_pointers=include_pointers) for v in value]
    assert value is None or isinstance(value,(str,int,float,bool)),('Unsupported diagnostic value',type(value))
    return value


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def normalized_fingerprint(value):
    if isinstance(value,dict):return {k:normalized_fingerprint(v) for k,v in value.items() if k not in ('data_pointer','storage_base_pointer','storage_bytes','storage_identity','tensor_version','device')}
    if isinstance(value,list):return [normalized_fingerprint(v) for v in value]
    return value


def compare_outputs(actual,reference,torch):
    assert len(actual)==len(reference)==2
    rows=[]
    for x,y in zip(actual,reference):
        assert x.shape==y.shape and x.dtype==y.dtype
        changed=int((x.contiguous().view(torch.int32)!=y.contiguous().view(torch.int32)).sum())
        delta=x.double()-y.double();norm=float(y.double().square().sum().sqrt().clamp_min(1e-30))
        rows.append({'shape':list(x.shape),'dtype':str(x.dtype),'different_FP32_words':changed,'all_words_equal':changed==0,
            'max_abs_difference':float(delta.abs().max()),'relative_RMS_error':float(delta.square().sum().sqrt())/norm})
    return rows
