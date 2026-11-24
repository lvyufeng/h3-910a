from tbe import tik
import inspect

for module_name, name in [('MatMulImpl', 'execute'), ('MatmulApiCls', 'execute')]:
    obj = tik.Tik.matmul.__globals__[module_name]
    print(module_name, 'module', obj.__module__, 'file', inspect.getfile(obj))
    member = getattr(obj, name, None)
    print(name, member)
    if member is not None:
        print(inspect.getfile(member), inspect.getsourcelines(member)[1])
print('globals:', [n for n in tik.Tik.matmul.__globals__ if 'Matmul' in n or 'MatMul' in n])
