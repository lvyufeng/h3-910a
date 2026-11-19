"""Exact same-weight position-only CPU/NPU diagnostic, not a full encoder benchmark.

Compare the unchanged R79 interpolation method to original HF Float-weight bilinear
interpolation and one Half output. No model/source patch, hidden conditioning cache,
Float parameter shadow, BF16 output oracle, or new encoder norm is introduced.
"""
import ast
import hashlib
import types
from pathlib import Path


def merge_row_order(h,w,merge,t=1):
    if h<=0 or w<=0 or t<=0 or h%merge or w%merge:raise ValueError('Invalid merge geometry')
    return [i*w+j for _ in range(t) for hi in range(0,h,merge) for wi in range(0,w,merge)
            for i in range(hi,hi+merge) for j in range(wi,wi+merge)]


def extract_original_method(path):
    """Compile only the exact unmodified method body, avoiding full model construction."""
    import torch
    tree=ast.parse(Path(path).read_text());classes=[c for c in tree.body if isinstance(c,ast.ClassDef)and c.name=='MiniMaxH3Qwen3VLVisionModel']
    if len(classes)!=1:raise ValueError('Missing actual candidate vision class')
    methods=[m for m in classes[0].body if isinstance(m,ast.FunctionDef)and m.name=='fast_pos_embed_interpolate']
    if len(methods)!=1:raise ValueError('Missing actual candidate position method')
    namespace={'torch':torch};exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['fast_pos_embed_interpolate']


def reconstruct_candidate_coordinates(grid,side,merge):
    import torch
    assert grid.device.type=='cpu'and torch.get_default_dtype()==torch.float32
    idx=[[]for _ in range(4)];weights=[[]for _ in range(4)]
    for t,h,w in grid.tolist():
        a=torch.linspace(0,side-1,h);b=torch.linspace(0,side-1,w)
        af=a.int();bf=b.int();ac=(af+1).clip(max=side-1);bc=(bf+1).clip(max=side-1)
        da=a-af;db=b-bf
        corners=[(af[None].T*side+bf[None]).flatten(),(af[None].T*side+bc[None]).flatten(),
                 (ac[None].T*side+bf[None]).flatten(),(ac[None].T*side+bc[None]).flatten()]
        factors=[((1-da)[None].T*(1-db)[None]).flatten(),((1-da)[None].T*db[None]).flatten(),
                 (da[None].T*(1-db)[None]).flatten(),(da[None].T*db[None]).flatten()]
        order=torch.tensor(merge_row_order(h,w,merge,t),dtype=torch.long)
        for i in range(4):idx[i].extend(corners[i][order].tolist());weights[i].extend(factors[i][order].tolist())
    return torch.tensor(idx,dtype=torch.long),torch.tensor(weights,dtype=torch.float32)


def info(x):
    import torch
    y=x.detach().cpu().contiguous()
    return {'shape':list(y.shape),'dtype':str(y.dtype),'sha256':hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'finite':bool(torch.isfinite(y).all())}


def equal_bits(a,b):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype:return False
    return bool(torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8)))


def compare_half(a,b):
    import torch
    aa=a.detach().cpu().contiguous();bb=b.detach().cpu().contiguous();assert aa.shape==bb.shape and aa.dtype==bb.dtype==torch.float16
    assert info(aa)['finite']and info(bb)['finite'];d=aa.double()-bb.double();den=bb.double().square().mean().sqrt().item()
    return {'npu':info(aa),'cpu':info(bb),'raw_bits_equal':equal_bits(aa,bb),'relative_rms':d.square().mean().sqrt().item()/max(den,1e-30),
            'max_abs_error':d.abs().max().item()}


def probe(weight,grid,original_hf_helper,candidate_source):
    import torch
    assert weight.dtype==torch.float16 and weight.shape==(2304,1152)and weight.device.type in('npu','privateuseone')
    assert grid.device.type=='cpu'and grid.tolist()==[[1,48,84],[1,48,84]]
    weight_before=info(weight);grid_before=info(grid);cpu_weight=weight.detach().cpu().contiguous()
    assert weight_before['sha256']=='e304bab6e8671ebeb9c08e6db9d0e51c3dcca394c78c3e40dd9130a3f6a23b07'
    hf_idx,hf_w=original_hf_helper(grid,num_grid_per_side=48,spatial_merge_size=2)
    idx,w=reconstruct_candidate_coordinates(grid,48,2)
    assert hf_idx.shape==idx.shape==(4,8064)and hf_w.shape==w.shape==(4,8064)
    assert hf_w.dtype==w.dtype==torch.float32
    address_equal=equal_bits(hf_idx.to(torch.int64),idx);weight_equal=equal_bits(hf_w,w)
    if not address_equal or not weight_equal:raise ValueError('Candidate CPU coordinates differ from the pinned original HF helper')
    gold=(torch.nn.functional.embedding(hf_idx,cpu_weight)*hf_w[:,:,None]).sum(0).to(torch.float16).contiguous()
    assert gold.shape==(8064,1152)
    cpu_reconstructed=(torch.nn.functional.embedding(idx,cpu_weight)*w[:,:,None]).sum(0).to(torch.float16).contiguous()
    assert equal_bits(gold,cpu_reconstructed)
    cpu_repeat=(torch.nn.functional.embedding(hf_idx,cpu_weight)*hf_w[:,:,None]).sum(0).to(torch.float16).contiguous()
    assert equal_bits(gold,cpu_repeat)
    embedding=torch.nn.Embedding(2304,1152,dtype=torch.float16,device=weight.device)
    with torch.no_grad():embedding.weight.copy_(weight)
    context=types.SimpleNamespace(pos_embed=embedding,num_grid_per_side=48,spatial_merge_size=2)
    method=types.MethodType(extract_original_method(candidate_source),context)
    with torch.inference_mode():
        actual=method(grid);repeat=method(grid)
        assert equal_bits(actual,repeat)
        samples=torch.nn.functional.embedding(idx.to(weight.device),embedding.weight).float()
        products=samples*w.to(weight.device)[:,:,None]
        sum0=products.sum(0).to(torch.float16)
        explicit=((products[0]+products[1])+products[2])+products[3]
        explicit=explicit.to(torch.float16)
    result={'scope':__doc__,'CPU_original_and_reconstructed_coordinates_and_output_bits_exact':True,
            'original_index_raw_dtype':str(hf_idx.dtype),'candidate_index_raw_dtype':str(idx.dtype),
            'index_int64_address_values_equal':address_equal,'Float32_bilinear_weights_equal_bits':weight_equal,
            'original_CPU_repeat_bits_equal':True,'candidate_actual_method_repeat_bits_equal':True,
            'actual_method_vs_original_CPU':compare_half(actual,gold),
            'original_final_order_NPU_sum0_vs_CPU':compare_half(sum0,gold),
            'original_final_order_NPU_explicit4add_vs_CPU':compare_half(explicit,gold),
            'actual_method_vs_NPU_final_order_sum0':compare_half(actual,sum0),
            'original_CPU_output':info(gold),'original_Half_parameter':weight_before,
            'parameter_unchanged':info(weight)==weight_before and info(embedding.weight)==weight_before,
            'grid_unchanged':info(grid)==grid_before,'no_model_or_source_patch':True,
            'not_full_encoder_time':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None}
    assert result['parameter_unchanged']and result['grid_unchanged']
    return result
