"""Default-off, process-local exact FP32 conversion reuse for text diagnostics.

No GEMM shape/order, accumulation, padding, all-reduce or output cast changes.
Cache slots are explicit module/slice identities; tensors retain the source
storage and include version/storage/offset/layout in their descriptor. Unknown
version counters fail closed to an uncached float conversion. Call clear() after
any .data write or external loader update: .data bypasses PyTorch version counters
and cannot be detected by a cheap descriptor. This diagnostic freezes/byte-hashes
parameters after loading; it is NOT a production offload/load-hook integration.
Activation reuse
lasts only for one attention scope and cannot leak into later layers/forwards.
"""
from contextlib import contextmanager
import torch


class ShadowWeights:
    def __init__(self, max_bytes=14 * 1024**3, max_entries=256):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = {}
        self.bytes = 0
        self.hits = self.misses = self.invalidations = self.bypasses = 0

    @staticmethod
    def descriptor(weight):
        if torch.is_grad_enabled():
            return None
        try:
            version = weight._version
        except RuntimeError:
            return None
        storage = weight.untyped_storage()
        return (version, weight.data_ptr(), storage._cdata, storage.nbytes(), weight.storage_offset(),
                tuple(weight.shape), tuple(weight.stride()), weight.dtype, weight.device)

    def get(self, slot, weight):
        descriptor = self.descriptor(weight)
        if descriptor is None:
            self.bypasses += 1
            return weight.float()
        prior = self.entries.get(slot)
        if prior is not None and prior[0] == descriptor:
            self.hits += 1
            return prior[2]
        if prior is not None:
            self.bytes -= prior[2].numel() * prior[2].element_size()
            del self.entries[slot]
            self.invalidations += 1
        needed = weight.numel() * 4
        if len(self.entries) >= self.max_entries or self.bytes + needed > self.max_bytes:
            raise RuntimeError('Explicit FP32 shadow budget exceeded; do not silently benchmark a different cache policy')
        value = weight.float()
        # No contiguous, shape, packing or parameter reassignment here: match the
        # original weight.float() result. New GEMM dispatch would be a new test.
        self.entries[slot] = (descriptor, weight, value)
        self.bytes += needed
        self.misses += 1
        return value

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def stats(self):
        return {'entries': len(self.entries), 'cached_fp32_bytes': self.bytes,
                'hits': self.hits, 'misses': self.misses, 'invalidations': self.invalidations,
                'uncacheable_bypasses': self.bypasses, 'max_bytes': self.max_bytes}


class SharedActivation:
    @staticmethod
    def signature(value):
        try:
            version = value._version
        except RuntimeError:
            # Layer outputs created in inference_mode have no version counter.
            # Reuse is strictly limited to the same object in one inspected
            # read-only attention call; never persist it across layer/forward.
            version = None
        return (version, value.data_ptr(), value.untyped_storage()._cdata, value.storage_offset(),
                tuple(value.shape), tuple(value.stride()), value.dtype, value.device)

    def __init__(self):
        self.active = None
        self.conversions = self.hits = 0

    @contextmanager
    def scope(self, value):
        if self.active is not None:
            raise RuntimeError('Nested shared-activation scopes are unsupported')
        version = self.signature(value)
        converted = value.float()
        self.conversions += 1
        self.active = (value, version, converted)
        try:
            yield
        finally:
            self.active = None

    def get(self, value):
        if self.active is not None:
            original, descriptor, converted = self.active
            if value is original:
                if self.signature(value) != descriptor:
                    raise RuntimeError('Attention input changed during exact activation reuse')
                self.hits += 1
                return converted
        self.conversions += 1
        return value.float()
