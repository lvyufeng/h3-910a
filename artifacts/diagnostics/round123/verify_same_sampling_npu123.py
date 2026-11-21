#!/usr/bin/env python3
"""910A check that exact GPU noise/sigma tensor bits survive export and local reconstruction.

CPU RNG and scalar schedule construction are setup, NOT a CPU model baseline. The
actual shared GPU-initial-sampling bundle is authoritative for later replay, so a
same-seed implementation mismatch is a recorded failure, not silently tolerated.
"""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path('/tmp/h3_align12301')


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = ROOT / 'NPU_same_initial_sampling_verified123.json'
    assert not p.exists()
    bundle_path = ROOT / 'same_initial_sampling123.pt'
    manifest = json.loads((ROOT / 'same_initial_sampling123.json').read_text())
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == manifest['payload_sha256']
    d = torch.load(bundle_path, map_location='cpu', weights_only=True)
    from h3_replay123.scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0, minimax_h3_euler_eta0_step
    torch.manual_seed(1234)
    vg = torch.Generator(device='cpu').manual_seed(1234)
    video = torch.randn((1, 24, 37, 48, 84), generator=vg, dtype=torch.float32)
    ag = torch.Generator(device='cpu').manual_seed(1234)
    audio = torch.randn((414, 32), generator=ag, dtype=torch.float32)
    base = torch.linspace(1., 0., 50, dtype=torch.float32)
    sigmas = {'video': 12. * base / (1. + 11. * base), 'audio': 3. * base / (1. + 2. * base)}
    r = {'scope': __doc__, 'completed': False, 'torch': torch.__version__, 'checks': {},
         'payload_bytes': bundle_path.stat().st_size, 'payload_sha256': manifest['payload_sha256'],
         'initial_sampling_authoritative_bundle_used': True, 'model_inference_executed': False}
    for name, value in [('video_latent_noise', video), ('initial_audio_rows', audio),
                        ('sigmas_video', sigmas['video']), ('sigmas_audio', sigmas['audio'])]:
        target = d[name]
        equal = torch.equal(value.view(torch.uint8), target.view(torch.uint8))
        r['checks'][name] = {'raw_bits_identical': equal, 'sha256_local_reconstructed': sha(value),
                            'sha256_GPU_exported': sha(target), 'shape': list(target.shape), 'dtype': str(target.dtype)}
        assert equal, name
    for name, row in manifest['tensor_identities'].items():
        assert sha(d[name]) == row['sha256_raw_tensor_bytes'], name
    r['all_shared_bundle_tensor_digests_verified'] = True
    r['sigma_points'] = 50
    r['DiT_evaluations'] = 49
    r['checkpoint_keyframe_indices'] = [0, -1]
    r['packed_rows'] = int(d['packed']['seq_len'])
    r['valid_rows'] = int(d['packed']['cu_seqlens'][1])
    r['completed'] = True
    p.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)


if __name__ == '__main__':
    main()
