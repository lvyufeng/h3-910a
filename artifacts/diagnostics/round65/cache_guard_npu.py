#!/usr/bin/env python3
"""Real-NPU cache invalidation controls; no weights or production files touched."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from shadow_cache65 import ShadowWeights, SharedActivation
from evidence_io65 import EvidenceWriter

ROOT = Path('/tmp/h3_round65_shadow01')
writer = EvidenceWriter(ROOT / 'cache_guard_npu.json')
r = {'checks': [], 'scope': 'Small actual-NPU lifecycle tests; not a text benchmark or arithmetic substitute for full-stack identity.',
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def check(name, observed):
    r['checks'].append({'name': name, 'passed': bool(observed)})
    if not observed:
        raise AssertionError(name)


try:
    torch.npu.set_device(0)
    # Construct ordinary versioned Parameter outside inference_mode, like the
    # actual model loader. Keep small real BF16 values away from rounding ties.
    x = torch.nn.Parameter(torch.arange(64, dtype=torch.float32, device='npu').to(torch.bfloat16).reshape(8, 8), requires_grad=False)
    cache = ShadowWeights(max_entries=8, max_bytes=65536)
    with torch.no_grad():
        prior = cache.get('weight', x)
        check('initial_conversion_bits', torch.equal(prior.cpu(), x.float().cpu()))
        check('same_parameter_hit', cache.get('weight', x) is prior)
        x.add_(1)
        current = cache.get('weight', x)
        check('ordinary_mutation_invalidates', current is not prior and torch.equal(current.cpu(), x.float().cpu()))
        before_version = x._version
        prior = current
        x.data.add_(2)
        r['data_write_version_changed'] = x._version != before_version
        check('data_write_explicit_clear_rebuild', (cache.clear() is None))
        current = cache.get('weight', x)
        check('after_clear_correct', current is not prior and torch.equal(current.cpu(), x.float().cpu()))
        prior = current
        x.data = x.data.clone()
        current = cache.get('weight', x)
        check('storage_rebind_invalidates', current is not prior and torch.equal(current.cpu(), x.float().cpu()))
        view = x[:4]
        prior = cache.get('view', view)
        check('fresh_same_storage_view_hit', cache.get('view', x[:4]) is prior)
        current = cache.get('view', x[4:])
        check('offset_view_invalidates', current is not prior and torch.equal(current.cpu(), x[4:].float().cpu()))
        prior = cache.get('stride', x)
        current = cache.get('stride', x.t())
        check('stride_changes_invalidate', current is not prior and torch.equal(current.cpu(), x.t().float().cpu()))
    with torch.inference_mode():
        unversioned = torch.ones(8, 8, dtype=torch.bfloat16, device='npu')
        a, b = cache.get('inference', unversioned), cache.get('inference', unversioned)
        check('unversioned_weights_bypass', a is not b and 'inference' not in cache.entries)
        activation = SharedActivation()
        with activation.scope(unversioned):
            a = activation.get(unversioned)
            check('unversioned_activation_same_scope', activation.get(unversioned) is a)
        check('activation_released_after_scope', activation.active is None)
    r['stats_before_clear'] = cache.stats()
    cache.clear()
    check('clear_releases_all_entries_and_accounted_bytes', cache.stats()['entries'] == 0 and cache.stats()['cached_fp32_bytes'] == 0)
    torch.npu.synchronize()
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print(json.dumps(r, indent=2), flush=True)
