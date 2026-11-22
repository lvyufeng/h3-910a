#!/usr/bin/env python3
"""Independent post-model CPU reductions, not the primary metric implementation."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT=Path('/tmp/h3_quality13601');SESSION=ROOT/'engine_quality136a'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for data in iter(lambda:file.read(1024**2),b''):digest.update(data)
    return digest.hexdigest()


def floating(actual,reference):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype
    word=np.uint16 if actual.dtype.itemsize==2 else np.uint32
    changed=int(np.count_nonzero(actual.view(word)!=reference.view(word)))
    a=actual.astype(np.float64);b=reference.astype(np.float64);delta=a-b
    return {'different_words':changed,'relative_RMS_ratio':float(np.linalg.norm(delta.reshape(-1))/max(np.linalg.norm(b.reshape(-1)),1e-30)),
        'max_abs_difference':float(np.max(np.abs(delta))),'reference_RMS':float(np.sqrt(np.mean(b*b)))}


def close_scalar(actual,recorded):
    assert math.isclose(actual,recorded,rel_tol=1e-10,abs_tol=1e-12), (actual,recorded)


def main():
    process=json.loads((ROOT/'engine_quality136a_process.json').read_text());assert process['exit_code']==0 and process['all_private_children_exited']
    life=json.loads((SESSION/'lifecycle_result136.json').read_text());assert life['completed'] and len(life['requests'])==3
    primary=json.loads((ROOT/'quality_comparison136.json').read_text());assert primary['completed']
    import torch  # Only CPU-deserialize own saved outputs, never a model/checkpoint.
    snapshots={}
    for row in life['requests']:
        directory=SESSION/row['label'];paths={'rawframes_sha256':directory/'nativeframes136.npy','rawaudio_sha256':directory/'audio136.npy',
            'final_latents_sha256':next(directory.glob('final_latents_pid*127.pt')),'encoder_hidden_file_sha256':next(directory.glob('live_encoder_pid*127.pt'))}
        assert all(path.is_file() and not path.is_symlink() and sha(path)==row[key] for key,path in paths.items())
        latent=torch.load(paths['final_latents_sha256'],map_location='cpu',weights_only=True)
        hidden=torch.load(paths['encoder_hidden_file_sha256'],map_location='cpu',weights_only=True)
        snapshots[row['label']]={'frames':np.load(paths['rawframes_sha256'],mmap_mode='r',allow_pickle=False),
            'audio':np.load(paths['rawaudio_sha256'],mmap_mode='r',allow_pickle=False),
            'video_latent':latent['video_latent'].numpy(),'audio_latent':latent['audio_latent'].numpy(),'hidden':hidden['hidden_states'].numpy()}
    result={'completed':True,'CPU_postprocessing_only_after_model_exit':True,'primary_metrics_independently_crosschecked':True,
        'postprocessing_source_sha256':sha(Path(__file__)),'primary_quality_report_sha256':sha(ROOT/'quality_comparison136.json'),
        'video_reduction_method':'Integer channel deltas, per-frame int64 squared sum, full pixel masks',
        'float_reduction_method':'Full unchanged arrays expanded to FP64, numpy norms; uint16/uint32 complete word comparisons',
        'consistency_tolerance_only_for_scalar_reduction_roundoff_not_model_error':True,'model_tolerance_or_adoption_decided':False,'pairs':[]}
    for pair in primary['comparisons']:
        actual=snapshots[pair['actual_request_label']];reference=snapshots[pair['reference_request_label']]
        row={'pair_id':pair['pair_id'],'float_metrics':{}}
        for key,metric in (('hidden','own_live_encoder_conditioning'),('video_latent','video_latent'),('audio_latent','audio_latent'),('audio','native_decoded_audio')):
            checked=floating(actual[key],reference[key]);target=pair[metric]
            assert checked['different_words']==target['different_words']
            close_scalar(checked['relative_RMS_ratio'],target['relative_RMS_ratio']);close_scalar(checked['max_abs_difference'],target['max_abs_difference'])
            row['float_metrics'][metric]=checked
        sse=absolute=channels=pixels=0;maxabs=-1;location=None;values=None;per_frame=[]
        for frame in range(124):
            a=actual['frames'][frame];b=reference['frames'][frame];d=a.astype(np.int16)-b.astype(np.int16);absd=np.abs(d)
            square=int(np.sum(d.astype(np.int32)**2,dtype=np.int64));sse+=square;absolute+=int(np.sum(absd,dtype=np.int64))
            channels+=int(np.count_nonzero(d));pixels+=int(np.count_nonzero(np.any(d!=0,axis=-1)))
            maximum=int(absd.max())
            if maximum>maxabs:
                maxabs=maximum;position=tuple(int(x) for x in np.unravel_index(int(absd.argmax()),d.shape));location=[frame,*position];values={'actual':int(a[position]),'reference':int(b[position])}
            mse=square/d.size;per_frame.append({'frame':frame,'mse':mse,'PSNR_dB':10*math.log10(255**2/mse) if mse else None})
        target=pair['native_decoded_video'];assert channels==target['different_channel_values'] and pixels==target['different_RGB_pixels'] and maxabs==target['max_abs_difference']
        mse=sse/actual['frames'].size;psnr=10*math.log10(255**2/mse) if sse else None
        if psnr is not None:close_scalar(psnr,target['psnr_db'])
        close_scalar(absolute/actual['frames'].size,target['mean_absolute_error'])
        row['video']={'different_channel_values':channels,'different_RGB_pixels':pixels,'PSNR_dB':psnr,'max_abs_0_255':maxabs,
            'max_abs_coordinate':location,'max_coordinate_pixel_values':values,'worst_frames':sorted([v for v in per_frame if v['PSNR_dB'] is not None],key=lambda v:v['PSNR_dB'])[:5]}
        result['pairs'].append(row)
    path=ROOT/'independent_quality_audit136.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
