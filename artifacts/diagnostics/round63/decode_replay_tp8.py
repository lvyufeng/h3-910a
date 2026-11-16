#!/usr/bin/env python3
"""Clean complete-video VAE replay on eight owned 910A cards.

Actual captured normalized FL2VA latent; native spatial tiles, temporal blending,
post-quant conv, pixel quantizer and HCCL gathers are all included. Measured decode
is resident: model loading/staging, audio decode and MP4 delivery are excluded and
reported separately. It is NOT the official warmed T2VA service benchmark.
"""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ.update(VLLM_OMNI_910A_NO_FUSED_NORM='1', VLLM_OMNI_910A_NO_FUSED_ACT='1',
                  VLLM_OMNI_910A_ATTN_CHUNK='128')
for key in ('VLLM_OMNI_910A_VAE_AUDIT_DIR', 'VLLM_OMNI_910A_TEXT_AUDIT_DIR',
            'VLLM_OMNI_910A_AUDIT_DIR', 'VLLM_OMNI_910A_ADALN_AUDIT_DIR',
            'VLLM_OMNI_910A_BLOCK_AUDIT_DIR'):
    os.environ.pop(key, None)
import torch
import torch_npu
from vllm_omni.platforms import current_omni_platform
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

ROOT = Path('/tmp/h3_round63_decode01')
CAP = ROOT / 'capture' / 'rank0'
MODEL = Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/video_vae')
rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert size == 8
OUT = ROOT / ('decode_replay01_rank%d.json' % rank)
if OUT.exists():
    raise RuntimeError('Refuse complete decoder proof overwrite')
current_omni_platform.set_device(torch.device('npu', rank))
torch.set_num_threads(2)
r = {'rank': rank, 'world_size': size, 'cases': {},
     'scope': 'Clean complete resident video VAE decode, actual FL2VA latent, all native tile/temporal/pixel/HCCL operations. Excludes model transfer, audio, MP4, DiT/encoder and full-generation quality.',
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'measurement': 'Barrier before each call; synchronized wall and current-stream events; compare CPU outputs outside timer; fixed case order.',
     'construction_default_dtype': 'torch.float16', 'storage_dtype': 'torch.float32'}


def cpu_sha(x):
    return hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()


def output_receipt(out, expected_sha):
    x = out.detach().cpu().contiguous()
    result = {'shape': list(x.shape), 'dtype': str(x.dtype), 'bytes': x.numel() * x.element_size(), 'sha256': cpu_sha(x)}
    result['raw_bits_equal_to_actual_capture'] = result['sha256'] == expected_sha
    if rank == 0 and not result['raw_bits_equal_to_actual_capture']:
        y = torch.load(CAP / 'decoded_video_reference.pt', weights_only=True)
        assert x.shape == y.shape and x.dtype == y.dtype == torch.uint8
        different, absolute, max_error = 0, 0, 0
        for start in range(0, x.shape[2], 4):
            a, b = x[:, :, start:start + 4], y[:, :, start:start + 4]
            error = (a.to(torch.int16) - b.to(torch.int16)).abs()
            different += int(a.ne(b).sum())
            absolute += int(error.sum())
            max_error = max(max_error, int(error.max()))
        result.update(different_pixels=different, max_byte_error=max_error, mean_byte_abs_error=absolute / x.numel())
    return result


def once(vae, latent, reference_sha):
    torch.distributed.barrier()
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    t = time.perf_counter()
    start.record()
    with torch.autocast('npu', dtype=torch.float16):
        out = vae.decode_latent(latent)
    end.record()
    torch.npu.synchronize()
    wall = time.perf_counter() - t
    stream = start.elapsed_time(end) / 1000
    # No capture hooks/CPU download/hash work is inside the timed interval.
    proof = output_receipt(out, reference_sha)
    del out
    return {'wall_s': wall, 'stream_s': stream, 'output': proof}


vae = None
try:
    torch.distributed.init_process_group('hccl', timeout=datetime.timedelta(seconds=900))
    meta = json.loads((CAP / 'decode_profile.json').read_text())
    reference_sha = meta['full_output_sha256']
    r['reference_output_sha256'] = reference_sha
    t = time.perf_counter()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        # Native remote loader's state-dict copies use the ambient dtype. Keep
        # the actual service construction precision, not a default-FP32 oracle.
        vae = MiniMaxH3VideoVAE(str(MODEL), device=torch.device('npu', rank), load_device=torch.device('cpu'),
                              decode_only=True, trust_remote_code=True)
    finally:
        torch.set_default_dtype(previous)
    r['cpu_load_s'] = time.perf_counter() - t
    vae.set_parallel_size(size, mode='tile', process_group=torch.distributed.group.WORLD)
    assert vae.model.parallel_tiling and not vae.model.decoder.spatial_parallel
    t = time.perf_counter()
    vae.decoder_component.load_to_device()
    torch.npu.synchronize()
    r['initial_stage_to_device_s'] = time.perf_counter() - t
    decoder = vae.model.decoder
    r['parameter_dtypes'] = dict(collections.Counter(str(p.dtype) for p in decoder.parameters()))
    r['parameter_bytes'] = sum(p.numel() * p.element_size() for p in decoder.parameters())
    latent = torch.load(CAP / 'video_latent.pt', weights_only=True).to(torch.device('npu', rank)).contiguous()
    r['latent'] = {'shape': list(latent.shape), 'dtype': str(latent.dtype), 'sha256': cpu_sha(latent.cpu())}
    linears = [linear for block in decoder.transformer_blocks
               for linear in (block.attn.to_qkv, block.attn.to_out, block.ff.w1, block.ff.w2)]
    assert len(linears) == 144
    with torch.inference_mode():
        for mode in ('baseline_c128', 'candidate_c256', 'candidate_c256_linear_fp16', 'baseline_restored'):
            os.environ['VLLM_OMNI_910A_ATTN_CHUNK'] = '256' if mode.startswith('candidate') else '128'
            if mode == 'candidate_c256_linear_fp16':
                torch.clear_autocast_cache()
                for linear in linears:
                    linear.to(dtype=torch.float16)
            elif mode == 'baseline_restored':
                torch.clear_autocast_cache()
                for linear in linears:
                    linear.to(dtype=torch.float32)
            row = {'first_call': once(vae, latent, reference_sha), 'warm_calls': []}
            for _ in range(3):
                row['warm_calls'].append(once(vae, latent, reference_sha))
            row['median_local_warm_s'] = statistics.median(call['wall_s'] for call in row['warm_calls'])
            shas = [call['output']['sha256'] for call in [row['first_call'], *row['warm_calls']]]
            row['repeat_exact'] = len(set(shas)) == 1
            row['all_calls_capture_exact'] = all(call['output']['raw_bits_equal_to_actual_capture']
                                                  for call in [row['first_call'], *row['warm_calls']])
            r['cases'][mode] = row
            OUT.write_text(json.dumps(r, indent=2) + '\n')
            print(json.dumps({'rank': rank, 'mode': mode, 'median_local_warm_s': row['median_local_warm_s'],
                              'repeat_exact': row['repeat_exact'], 'all_calls_capture_exact': row['all_calls_capture_exact']}), flush=True)
        r['all_cases_capture_exact'] = all(case['all_calls_capture_exact'] for case in r['cases'].values())
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r, indent=2) + '\n')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print('Evidence:', OUT, flush=True)
