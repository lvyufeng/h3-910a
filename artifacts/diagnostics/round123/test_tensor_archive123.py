#!/usr/bin/env python3
"""Regression checks for activation-archive validation and exact shared sampling setup.

These checks inspect saved tensors and scalar contracts only. They do not run H3,
use a CPU inference baseline, accept a model tolerance, or download model weights.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import unittest
import zipfile
import numpy as np
from torch_pt_numpy123 import load_tensor_archive, _TensorOnlyUnpickler
from compare_refiner_numpy123 import metric

ROOT = Path(__file__).resolve().parent


class TensorArchiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'same_initial_sampling123.json').read_text())
        cls.bundle = load_tensor_archive(ROOT / 'same_initial_sampling123.pt')

    def test_bundle_file_digest(self):
        self.assertEqual(hashlib.sha256((ROOT/'same_initial_sampling123.pt').read_bytes()).hexdigest(), self.manifest['payload_sha256'])

    def test_each_exported_tensor_digest(self):
        for name, expected in self.manifest['tensor_identities'].items():
            value = self.bundle[name]
            self.assertEqual(list(value.shape), expected['shape'], name)
            self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(), expected['sha256_raw_tensor_bytes'], name)

    def test_exact_sigmas(self):
        self.assertEqual(self.bundle['sigmas_video'].view(np.uint32).tolist(), self.manifest['sigma_video_FP32_bits'])
        self.assertEqual(self.bundle['sigmas_audio'].view(np.uint32).tolist(), self.manifest['sigma_audio_FP32_bits'])
        self.assertEqual(len(self.bundle['sigmas_video'])-1, 49)
        self.assertEqual(float(self.bundle['sigmas_video'][0]), 1.)
        self.assertEqual(float(self.bundle['sigmas_audio'][-1]), 0.)

    def test_exact_target_and_anchor_initial_rows(self):
        update = self.bundle['packed']['update_mask']
        initial = self.bundle['initial_video_rows_with_keyframes']
        self.assertEqual(initial[update].tobytes(), self.bundle['initial_video_target_rows'].tobytes())
        self.assertEqual(initial[~update].tobytes(), self.bundle['keyframe_noise_aug_rows'].tobytes())
        self.assertEqual(int((~update).sum()), 2016)
        self.assertEqual(int(update.sum()), 37296)

    def test_packed_shape_document_boundary(self):
        self.assertEqual(int(self.bundle['packed']['seq_len']), 41792)
        self.assertEqual(int(self.bundle['packed']['cu_seqlens'][1]), 41769)
        self.assertEqual(len(self.bundle['packed']['text_pos']), 2043)
        self.assertEqual(len(self.bundle['packed']['audio_pos']), 414)
        self.assertEqual(len(self.bundle['packed']['img_pos']), 39312)

    def test_isolated_NPU_source_delta_matches_executed_hash(self):
        original=(ROOT/'transformer_successful_GPU_source123.py').read_text()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),'7cf0c53e6d9199b7944d9026c2bdc1120b1e32084cffe266c434d2d6449b8029')
        text=original.replace('if x.device.type != "cuda" or layer.weight.dtype != torch.float16:', 'if x.device.type not in ("cuda", "npu") or layer.weight.dtype != torch.float16:')
        text=text.replace('peak = float(product.abs().amax())', 'peak = float(product.abs().amax(dim=tuple(range(product.ndim))))')
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b')

    def test_refiner_output_digests_match_receipts(self):
        for host, directory in [('GPU','GPU_refiner_reference'),('NPU','NPU_refiner_partial')]:
            receipt=json.loads((ROOT/directory/'refiner_replay_rank0.json').read_text())
            path=ROOT/'refiner_compare_minimal'/f'{host}_refiner_final_rank0_legacy.pt'
            value=load_tensor_archive(path)['value']
            self.assertEqual(value.shape,(2043,5376))
            self.assertEqual(value.dtype,np.float16)
            self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(),receipt['captures']['refiner_final']['sha256_raw_tensor_bytes'])
            self.assertTrue(np.isfinite(value).all())

    def test_same_row_ids_before_error_comparison(self):
        for name in ['video_patch_proj','audio_patch_proj','block_00_input_local_rows','block_00_output_local_rows','block_01_output_local_rows']:
            gp=ROOT/'GPU_first_blocks_reference'/f'{name}_sample_rank0.pt'
            np_=ROOT/'NPU_first_blocks_reference'/f'{name}_sample_rank0.pt'
            gpu,npu=load_tensor_archive(gp),load_tensor_archive(np_)
            self.assertEqual(gpu['full_shape'],npu['full_shape'])
            self.assertEqual(gpu['row_indices'].tobytes(),npu['row_indices'].tobytes())
            self.assertEqual(gpu['value'].dtype,npu['value'].dtype)

    def test_reject_arbitrary_pickle_globals(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('archive/data.pkl',b'cos\nsystem\n.')
        stream.seek(0)
        with zipfile.ZipFile(stream) as z:
            with self.assertRaises(pickle.UnpicklingError):
                _TensorOnlyUnpickler(io.BytesIO(z.read('archive/data.pkl')),z,'archive/').load()

    def test_relative_error_uses_reference_denominator(self):
        candidate=np.array([[3.,0.]],dtype=np.float16)
        reference=np.array([[1.,1.]],dtype=np.float16)
        self.assertAlmostEqual(metric(candidate,reference)['relative_RMS'],math.sqrt(5./2.))


if __name__=='__main__':
    unittest.main(verbosity=2)
