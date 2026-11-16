"""Explicit diagnostic-only, synchronized COMPLETE service decode timing.

Called only through VLLM_OMNI_910A_DECODE_TIMING_DIR. The native pipeline decode
includes stage-cache release, video load/decode/offload, crop, audio load/decode/
offload and its CPU stage-output. Excludes subsequent frame D2H/MP4 mux.
A paired request repeats this complete stage on exactly the same input tensors;
it is output-identity evidence, NOT a normal request/E2E timing measurement.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from .h3_910a_vae import mode
from .h3_evidence64 import EvidenceWriter

_COUNT = 0


def digest(x):
    x = x.detach().cpu().contiguous()
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'bytes': x.numel() * x.element_size(),
            'sha256': hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()}


def synchronized_call(owner, native, video_latent, audio_latent, active, **kwargs):
    with mode(owner.video_vae, active):
        torch.distributed.barrier()
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        output = native(video_latent, audio_latent, **kwargs)
        end.record()
        torch.npu.synchronize()
        wall = time.perf_counter() - t
        return output, {'wall_s': wall, 'stream_s': start.elapsed_time(end) / 1000,
                        'candidate_enabled': active}


def decode_stage(owner, native, video_latent, audio_latent, **kwargs):
    global _COUNT
    _COUNT += 1
    rank = torch.distributed.get_rank()
    root = Path(os.environ['VLLM_OMNI_910A_DECODE_TIMING_DIR'])
    control = json.loads((root / 'control.json').read_text())
    label = control['label']
    if not label.isalnum() or len(label) > 40:
        raise ValueError('Unsafe decode label')
    if control['mode'] not in ('baseline', 'candidate', 'paired'):
        raise ValueError('Unknown service decode mode')
    writer = EvidenceWriter(root / ('%s_rank%d.json' % (label, rank)))
    row = {'rank': rank, 'label': label, 'request_index': _COUNT, 'mode': control['mode'], 'calls': [],
           'scope': 'Complete service decode: cache release + video/audio load, native decode, offload and crop, synchronized per rank. Excludes MP4 and subsequent video D2H.',
           'timing_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    controller = getattr(owner.video_vae, '_h3_910a_finite_select', None)
    calls_before = controller.calls if controller is not None else 0
    result = None
    try:
        if control['mode'] == 'paired':
            row['input_before'] = {'video': digest(video_latent), 'audio': digest(audio_latent)}
            # First baseline/candidate warm calls excluded from comparative medians.
            # Native component lifecycle still executes every time, no resident bypass.
            for active in (False, True, False, True, False, True, False, True, False):
                output, timing = synchronized_call(owner, native, video_latent, audio_latent, active, **kwargs)
                timing['output'] = {'video': digest(output[0]), 'audio': digest(output[1])}
                row['calls'].append(timing)
                del output
                writer.write(row)
            row['input_after'] = {'video': digest(video_latent), 'audio': digest(audio_latent)}
            if row['input_before'] != row['input_after']:
                raise RuntimeError('Decode mutated a shared latent input')
            row['all_complete_outputs_equal'] = all(call['output'] == row['calls'][0]['output'] for call in row['calls'])
            if not row['all_complete_outputs_equal']:
                raise AssertionError('Paired same-input complete service outputs differ')
            # Return a fresh native decode; large paired outputs are not retained
            # across later staging cycles and are not mistaken for the normal call.
            result, row['return_call'] = synchronized_call(owner, native, video_latent, audio_latent, True, **kwargs)
            row['return_output'] = {'video': digest(result[0]), 'audio': digest(result[1])}
            assert row['return_output'] == row['calls'][0]['output']
        else:
            active = control['mode'] == 'candidate'
            result, timing = synchronized_call(owner, native, video_latent, audio_latent, active, **kwargs)
            row['calls'].append(timing)
        row['controller_calls_delta'] = controller.calls - calls_before if controller is not None else 0
        row['completed'] = True
        return result
    except BaseException as exc:
        row['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        writer.write(row)
        print('[H3DECODE64] ' + json.dumps({'rank': rank, 'label': label, 'mode': control['mode'],
                                           'wall_s': [x['wall_s'] for x in row['calls']],
                                           'completed': row.get('completed', False)}), flush=True)
