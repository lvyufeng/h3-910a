"""Isolated/default-off replacement of only VAE half finite-select output.

Preserve the exact loaded remote SDPA function's q/k/v transpose, mask shape,
backend context and functional SDPA call. Replace its single final where/isfinite
return, not the attention algorithm. Never writes a production file.
"""
import ast
import copy
import hashlib
import importlib
import inspect
import torch
from sanitize_vector63 import sanitize


def finite_select(out):
    if out.device.type in ('npu', 'privateuseone') and out.dtype == torch.float16 and out.numel() > 0 and out.numel() % 128 == 0:
        return sanitize(out)
    return torch.where(torch.isfinite(out), out, torch.zeros((), dtype=out.dtype, device=out.device))


def replace_final_return(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Expected exactly the native SDPA function definition')
    function = tree.body[0]
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or function.body[-1] is not returns[0]:
        raise ValueError('Unexpected SDPA return/control-flow contract')
    expected = ast.parse('torch.where(torch.isfinite(out), out, torch.zeros((), dtype=out.dtype, device=out.device))', mode='eval').body
    if ast.dump(returns[0].value) != ast.dump(expected):
        raise ValueError('Loaded SDPA finite-select differs from the checked contract')
    original = copy.deepcopy(tree)
    returns[0].value = ast.Call(func=ast.Name(id='_h3_sanitize63', ctx=ast.Load()),
                               args=[ast.Name(id='out', ctx=ast.Load())], keywords=[])
    ast.fix_missing_locations(tree)
    # Remove only the final return in both ASTs to certify all earlier logic.
    before, after = copy.deepcopy(original), copy.deepcopy(tree)
    before.body[0].body.pop()
    after.body[0].body.pop()
    assert ast.dump(before) == ast.dump(after)
    return tree


def install(vae):
    package = type(vae.model.decoder).__module__.rsplit('.', 1)[0]
    module = importlib.import_module(package + '.flash')
    original = module._sdpa_attention
    source = inspect.getsource(original)
    tree = replace_final_return(source)
    namespace = dict(original.__globals__)
    namespace['_h3_sanitize63'] = finite_select
    exec(compile(tree, '<isolated H3 VAE finite-select63>', 'exec'), namespace)
    replacement = namespace['_sdpa_attention']
    module._sdpa_attention = replacement
    receipt = {'target_module': module.__name__, 'original_function_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'only_final_return_changed': True, 'attention_algorithm_changed': False,
               'replacement_ast_sha256': hashlib.sha256(ast.dump(tree).encode()).hexdigest()}
    def restore():
        if module._sdpa_attention is not replacement:
            raise RuntimeError('VAE SDPA target changed while diagnostic replacement was installed')
        module._sdpa_attention = original
        return module._sdpa_attention is original
    return receipt, restore
