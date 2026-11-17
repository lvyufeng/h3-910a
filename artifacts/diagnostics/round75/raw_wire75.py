"""Metadata-only BF16/raw-int32 views for genuine ND collective probes."""
import torch
import torch_npu


def pack(value):
    if value.dtype!=torch.bfloat16 or value.ndim!=2 or value.device.type not in('npu','privateuseone'):raise ValueError('Require2DdeviceBF16')
    if value.shape[-1]%2:raise ValueError('RequireevenBF16columns')
    if value.storage_offset()%2:raise ValueError('RequireevenBF16storageoffset')
    if not value.is_contiguous()or value.stride(-1)!=1:raise ValueError('RequirecontiguousBF16')
    if value.device.index!=torch.npu.current_device()or value.data_ptr()%32:raise ValueError('Requirecurrentdeviceand32Bpointer')
    if torch_npu.get_npu_format(value)not in(0,2):raise ValueError('RequiregenuineNDformat')
    return value.view(torch.int32)


def unpack(value):
    if value.dtype!=torch.int32 or value.ndim!=2 or value.device.type not in('npu','privateuseone')or not value.is_contiguous():raise ValueError('Requiredevicecontiguous2DINT32')
    if value.device.index!=torch.npu.current_device()or value.data_ptr()%32:raise ValueError('RequirecurrentINT32deviceand32Bpointer')
    if torch_npu.get_npu_format(value)not in(0,2):raise ValueError('RequireINT32ND')
    return value.view(torch.bfloat16)
