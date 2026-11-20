#!/usr/bin/env python3
"""Read agent-created CPU tensor-only torch.save ZIP archives with NumPy.

For activation comparison only, not model inference. The restricted unpickler
allows only torch tensor rebuilding and typed storages, with no arbitrary globals.
No torch installation, model weights, or BF16-to-Half conversions are involved.
"""
from __future__ import annotations
import collections
import io
import pickle
from pathlib import Path
import zipfile
import numpy as np

_STORAGE_DTYPES = {
    'HalfStorage': np.dtype('<f2'), 'FloatStorage': np.dtype('<f4'),
    'DoubleStorage': np.dtype('<f8'), 'LongStorage': np.dtype('<i8'),
    'IntStorage': np.dtype('<i4'), 'ShortStorage': np.dtype('<i2'),
    'CharStorage': np.dtype('i1'), 'ByteStorage': np.dtype('u1'),
    'BoolStorage': np.dtype('?'),
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _STORAGE_DTYPES[name]


def _rebuild_tensor(storage, offset, shape, stride, *unused):
    shape = tuple(int(x) for x in shape)
    stride = tuple(int(x) for x in stride)
    if any(x < 0 for x in (*shape, *stride)):
        raise ValueError('Reject negative tensor shape or stride')
    maximum = int(offset) + sum((length - 1) * step for length, step in zip(shape, stride) if length)
    if np.prod(shape, dtype=np.int64) and maximum >= storage.size:
        raise ValueError('Tensor storage view is out of bounds')
    return np.ndarray(shape=shape, dtype=storage.dtype, buffer=storage,
                      offset=int(offset) * storage.dtype.itemsize,
                      strides=tuple(step * storage.dtype.itemsize for step in stride)).copy()


class _TensorOnlyUnpickler(pickle.Unpickler):
    def __init__(self, handle, archive, prefix):
        super().__init__(handle)
        self.archive = archive
        self.prefix = prefix
        self.storages = {}

    def find_class(self, module, name):
        if module == 'torch' and name in _STORAGE_DTYPES:
            return _StorageType(name)
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return _rebuild_tensor
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        raise pickle.UnpicklingError(f'Reject non-tensor global {module}.{name}')

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != 'storage':
            raise pickle.UnpicklingError('Reject unknown persistent object')
        _, storage_type, key, location, numel = pid
        if not isinstance(storage_type, _StorageType) or location != 'cpu':
            raise pickle.UnpicklingError('Only typed CPU tensor storage is allowed')
        if key not in self.storages:
            data = self.archive.read(self.prefix + 'data/' + str(key))
            if len(data) != int(numel) * storage_type.dtype.itemsize:
                raise ValueError('Storage byte count mismatch')
            self.storages[key] = np.frombuffer(data, dtype=storage_type.dtype)
        return self.storages[key]


def load_tensor_archive(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        pickle_names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(pickle_names) != 1:
            raise ValueError('Expected one torch ZIP tensor archive pickle')
        name = pickle_names[0]
        prefix = name[:-len('data.pkl')]
        byteorder = prefix + 'byteorder'
        if byteorder in archive.namelist() and archive.read(byteorder).strip() != b'little':
            raise ValueError('Only little-endian storage is accepted')
        return _TensorOnlyUnpickler(io.BytesIO(archive.read(name)), archive, prefix).load()


if __name__ == '__main__':
    import sys
    value = load_tensor_archive(sys.argv[1])
    print({name: {'shape': list(x.shape), 'dtype': str(x.dtype)} for name, x in value.items() if isinstance(x, np.ndarray)})
