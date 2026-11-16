"""Default-off 910A VAE finite-select. Never changes global Torch/SDPA or weights.

Enable only with VLLM_OMNI_910A_VAE_FINITE_SELECT=1. This selects zero for NaN/Inf;
it is bit-identical to the current native path on the finite measured workload,
not a promise of native exceptional-value equivalence. Instance-local cloned
functions keep the encoder, DiT and other VAE instances untouched.
"""
import ast
from contextlib import contextmanager
import copy
import ctypes
import inspect
import os
from types import FunctionType, MethodType, SimpleNamespace


def enabled():
    return os.environ.get('VLLM_OMNI_910A_VAE_FINITE_SELECT', '0') == '1'


def clone(function, **replacements):
    function = inspect.unwrap(function)
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def replace_return(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Expected exactly the checked SDPA function')
    function = tree.body[0]
    returns = [x for x in ast.walk(function) if isinstance(x, ast.Return)]
    if len(returns) != 1 or function.body[-1] is not returns[0]:
        raise ValueError('Unexpected SDPA control flow')
    expected = ast.parse('torch.where(torch.isfinite(out), out, torch.zeros((), dtype=out.dtype, device=out.device))', mode='eval').body
    if ast.dump(returns[0].value) != ast.dump(expected):
        raise ValueError('Unknown native output handling; refuse the opt-in replacement')
    prior = copy.deepcopy(tree)
    returns[0].value = ast.Call(func=ast.Name(id='_finite_select64', ctx=ast.Load()), args=[ast.Name(id='out', ctx=ast.Load())], keywords=[])
    before, after = copy.deepcopy(prior), copy.deepcopy(tree)
    before.body[0].body.pop()
    after.body[0].body.pop()
    assert ast.dump(before) == ast.dump(after)
    return ast.fix_missing_locations(tree)


def install(vae, device):
    if not enabled() or device.type not in ('npu', 'privateuseone') or not hasattr(vae.model, 'decoder'):
        return False
    if hasattr(vae, '_h3_910a_finite_select'):
        return True
    import torch
    from vllm_omni import h3_910a_sdpa_shim as shim
    from vllm_omni.diffusion.attention.backends import fa910a as fa
    # An A2/A3-targeted binary must never be confused with this Ascend910A one.
    lib = fa._lib()
    soc_query = lib.aclrtGetSocName
    soc_query.argtypes = []
    soc_query.restype = ctypes.c_char_p
    soc = soc_query().decode()
    if soc not in ('Ascend910A', 'Ascend910', 'Ascend910B'):
        raise RuntimeError('VAE finite-select is validated for genuine 910A only: ' + soc)
    if torch.nn.functional.scaled_dot_product_attention is not shim._shim:
        raise RuntimeError('Opt-in VAE expected the verified 910A SDPA shim')
    chunk = int(os.environ.get('VLLM_OMNI_910A_VAE_QUERY_CHUNK', '256'))
    if chunk not in (128, 256, 512):
        raise ValueError('VAE query chunk must be a measured 128/256/512')
    from .h3_910a_vae_kernel import sanitize
    def finite_select(out):
        if out.device.type in ('npu', 'privateuseone') and out.dtype == torch.float16 and out.numel() > 0 and out.numel() % 128 == 0:
            return sanitize(out)
        return torch.where(torch.isfinite(out), out, torch.zeros((), dtype=out.dtype, device=out.device))
    # Query chunking is private to these cloned VAE functions, not an env change
    # affecting Qwen/DiT or another instance of the remote-code module.
    local_sdpa = clone(shim._matmul_sdpa, _chunk_rows=lambda: chunk)
    originals = []
    controller = SimpleNamespace(active=True, query_chunk=chunk, soc=soc, originals=originals, calls=0)
    prepared = []
    for block in vae.model.decoder.transformer_blocks:
        attention = block.attn
        original = attention._perform_attention
        unbound = original.__func__
        native_flash = inspect.unwrap(unbound.__globals__['flash_attn'])
        native_sdpa = native_flash.__globals__['_sdpa_attention']
        tree = replace_return(inspect.getsource(native_sdpa))
        namespace = dict(native_sdpa.__globals__)
        namespace.update(_finite_select64=finite_select, F=SimpleNamespace(scaled_dot_product_attention=local_sdpa))
        exec(compile(tree, '<instance-local 910A VAE SDPA>', 'exec'), namespace)
        local_flash = clone(native_flash, _sdpa_attention=namespace['_sdpa_attention'])
        local_flash = torch.compiler.disable(local_flash)
        optimized = clone(unbound, flash_attn=local_flash)
        def dispatch(owner, q, k, v, pack_info, original=original, optimized=optimized):
            if not controller.active:
                return original(q, k, v, pack_info)
            controller.calls += 1
            return optimized(owner, q, k, v, pack_info)
        prepared.append((attention, original, dispatch))
    if len(prepared) != 36:
        raise ValueError('Unexpected VAE decoder block count')
    # Commit only after every function's source/contract has passed validation.
    for attention, original, dispatch in prepared:
        originals.append((attention, original))
        attention._perform_attention = MethodType(dispatch, attention)
    vae._h3_910a_finite_select = controller
    print('[H3VAE64] installed instance_only=1 default_off=1 finite_select=1 chunk=%d soc=%s exceptional_values=zero' % (chunk, soc), flush=True)
    return True


@contextmanager
def mode(vae, active):
    controller = getattr(vae, '_h3_910a_finite_select', None)
    if active and controller is None:
        raise RuntimeError('Candidate requested without explicitly enabled VAE integration')
    previous = controller.active if controller is not None else None
    if controller is not None:
        controller.active = bool(active)
    try:
        yield
    finally:
        if controller is not None:
            controller.active = previous
