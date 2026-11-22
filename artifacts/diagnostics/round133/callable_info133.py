import hashlib
import inspect
from pathlib import Path


def callable_info(function):
    row={'module':getattr(function,'__module__',None),'qualified_name':getattr(function,'__qualname__',type(function).__qualname__),
        'type':type(function).__module__+'.'+type(function).__qualname__}
    try:path=inspect.getsourcefile(function)
    except TypeError:path=None
    row['source_path']=path
    if path and Path(path).is_file():row['source_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    else:row['implementation_source_unknown']=True
    return row
