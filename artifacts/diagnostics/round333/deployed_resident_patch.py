"""Keep the text encoder resident on device after the first encode (math unchanged).

Sequence:
  1. first encode runs exactly as before (prefetch -> forward -> offload per block);
  2. after it, every block is restored to device once (restore_next_block);
  3. from then on prefetch_layer/offload_layer short-circuit, so later encodes do
     no H2D and no GPU storage churn.
Guards: abort residency if host memory headroom is too small (R331/R332 lessons).
"""
import time


def _free_gib():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return -1.0


def _find_offloaded(module, depth=0, seen=None):
    import torch.nn as nn
    if seen is None:
        seen = set()
    if id(module) in seen or depth > 6:
        return None
    seen.add(id(module))
    if getattr(module, '_omni_layerwise_enabled', False):
        return module
    if isinstance(module, nn.Module):
        for child in module.children():
            found = _find_offloaded(child, depth + 1, seen)
            if found is not None:
                return found
    if isinstance(module, (list, tuple, dict)):
        for child in (module.values() if isinstance(module, dict) else module):
            found = _find_offloaded(child, depth + 1, seen)
            if found is not None:
                return found
    return None


def install(module_ns):
    import torch
    from vllm_omni.diffusion.offloader import layerwise_backend as lb

    hook_cls = lb.LayerwiseOffloadHook
    orig_prefetch = hook_cls.prefetch_layer
    orig_offload = hook_cls.offload_layer
    state = {'resident': False}

    def prefetch(self, non_blocking=True):
        if state['resident']:
            return
        return orig_prefetch(self, non_blocking)

    def offload(self):
        if state['resident']:
            return
        return orig_offload(self)

    hook_cls.prefetch_layer = prefetch
    hook_cls.offload_layer = offload
    print('[RESIDENT4] prefetch/offload short-circuit installed', flush=True)

    for obj in list(vars(module_ns).values()):
        if not isinstance(obj, type) or 'encode_prompt' not in obj.__dict__:
            continue
        orig = obj.encode_prompt

        def wrapper(self, *a, __orig=orig, **k):
            t0 = time.perf_counter()
            out = __orig(self, *a, **k)
            torch.npu.synchronize()
            first = time.perf_counter() - t0
            if not state['resident']:
                free = _free_gib()
                print('[RESIDENT4] first encode %.2fs free=%.1f GiB' % (first, free), flush=True)
                target = _find_offloaded(self)
                hooks = list(getattr(target, '_omni_layerwise_hooks', [])) if target is not None else []
                print('[RESIDENT4] hooks=%d' % len(hooks), flush=True)
                if hooks and free >= 0 and free >= 70.0:
                    t1 = time.perf_counter()
                    ok = 0
                    for hook in hooks:
                        try:
                            hook.prefetch_layer(non_blocking=False)
                            ok += 1
                        except Exception as exc:  # noqa: BLE001
                            print('[RESIDENT4] restore failed: %r' % (exc,), flush=True)
                            break
                    torch.npu.synchronize()
                    # non-block state (embeddings / norms) must be resident too, otherwise
                    # forward mixes CPU weights with NPU activations
                    try:
                        from vllm_omni.diffusion.offloader.component_utils import move_non_block_state_to_device
                        groups = list(getattr(target, '_omni_layerwise_block_groups', []))
                        dev = torch.device('npu', torch.npu.current_device())
                        move_non_block_state_to_device(target, groups, dev)
                        torch.npu.synchronize()
                        print('[RESIDENT4] non-block state moved to %s (groups=%d)' % (dev, len(groups)), flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print('[RESIDENT4] non-block move failed: %r' % (exc,), flush=True)
                    # sweep any remaining CPU tensors on the pipeline instance (vision tower etc.)
                    try:
                        cpu_named = [(n, pm) for n, pm in target.named_parameters() if pm.device.type == 'cpu']
                        total_bytes = sum(pm.numel() * pm.element_size() for _, pm in cpu_named)
                        print('[RESIDENT4] leftover cpu params=%d bytes=%.2f GiB names=%s'
                              % (len(cpu_named), total_bytes / 1024.0 ** 3,
                                 [n for n, _ in cpu_named[:5]]), flush=True)
                        if len(cpu_named) and total_bytes < 20.0 * 1024 ** 3:
                            for _, pm in cpu_named:
                                pm.data = pm.data.to(dev)
                            torch.npu.synchronize()
                            print('[RESIDENT4] leftover moved to %s' % dev, flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print('[RESIDENT4] leftover sweep failed: %r' % (exc,), flush=True)
                    state['resident'] = (ok == len(hooks))
                    print('[RESIDENT4] resident=%s blocks=%d/%d restore=%.2fs free_after=%.1f GiB hbm=%.1f GiB'
                          % (state['resident'], ok, len(hooks), time.perf_counter() - t1,
                             _free_gib(), torch.npu.memory_allocated() / 1024.0 ** 3), flush=True)
                else:
                    print('[RESIDENT4] skip residency (hooks=%d free=%.1f)' % (len(hooks), free), flush=True)
            else:
                print('[RESIDENT4] warm encode %.3fs hbm=%.1f GiB'
                      % (first, torch.npu.memory_allocated() / 1024.0 ** 3), flush=True)
            return out

        setattr(obj, 'encode_prompt', wrapper)
        print('[RESIDENT4] patched %s.encode_prompt' % obj.__name__, flush=True)
