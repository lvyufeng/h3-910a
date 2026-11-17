#!/usr/bin/env python3
"""Full-frame video inspection metrics;not a ground-truth generation-quality score."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import av
from evidence_io76 import EvidenceWriter

ROOT=Path('/tmp/h3_round76_generation02')
writer=EvidenceWriter(ROOT/'video_scan.json')
r={'scope':__doc__,'videos':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'PSNR_is_endpoint_reference_reconstruction_only':True,'endpoint_PSNR_includes_lossy_H264_export':True,
   'optical_flow_or_CLIP_or_OCR_score_computed':False,'official_quality_accepted':False}


def endpoint(image,reference):
    error=image.astype(np.float64)-reference.astype(np.float64)
    mse=float(np.square(error).mean())
    return {'MAE_pixels':float(np.abs(error).mean()),'MSE_pixels':mse,'PSNR_db':10*math.log10(255**2/mse)if mse else None,
            'exact_pixels':bool(np.array_equal(image,reference))}


try:
    first_ref=np.asarray(Image.open(ROOT/'first.png').convert('RGB'));last_ref=np.asarray(Image.open(ROOT/'last.png').convert('RGB'))
    for label in('optimized','HF_reference_conditioning'):
        path=ROOT/(label+'.mp4');row={'label':label,'path':str(path),'mp4_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frames':[]}
        previous=None;first=None;last=None;hasher=hashlib.sha256()
        with av.open(str(path))as container:
            stream=container.streams.video[0]
            row['stream']={'width':stream.width,'height':stream.height,'average_rate':str(stream.average_rate),
                           'duration_s':float(stream.duration*stream.time_base)if stream.duration is not None else None,
                           'video_tracks':len(container.streams.video),'audio_tracks':len(container.streams.audio)}
            for index,frame in enumerate(container.decode(video=0)):
                image=frame.to_ndarray(format='rgb24');assert image.shape==(768,1344,3)and image.dtype==np.uint8
                hasher.update(memoryview(image));rgb=image.astype(np.float32)
                red=(rgb[:,:,0]>140)&(rgb[:,:,0]>rgb[:,:,1]*1.6)&(rgb[:,:,0]>rgb[:,:,2]*1.6)
                ys,xs=np.nonzero(red)
                item={'index':index,'pts_seconds':float(frame.pts*frame.time_base)if frame.pts is not None else None,
                      'mean_RGB':rgb.mean((0,1)).tolist(),'std_RGB':rgb.std((0,1)).tolist(),'red_pixel_count':int(red.sum()),
                      'red_centroid_x':float(xs.mean())if len(xs)else None,'red_centroid_y':float(ys.mean())if len(ys)else None,
                      'consecutive_frame_MAE':float(np.abs(rgb-previous).mean())if previous is not None else None}
                if index==0:first=image.copy()
                last=image.copy();previous=rgb
                row['frames'].append(item)
        assert len(row['frames'])==124 and row['stream']['width']==1344 and row['stream']['height']==768
        row['all124_frames_decoded']=True;row['decoded_RGB_sha256']=hasher.hexdigest()
        row['first_vs_input_reference']=endpoint(first,first_ref);row['last_vs_input_reference']=endpoint(last,last_ref)
        original_first=ROOT/(label+'_frame000.png');original_last=ROOT/(label+'_frame123.png')
        if original_first.is_file()and original_last.is_file():
            row['original_exported_RGB_endpoint_metrics_before_MP4_codec']={'first':endpoint(np.asarray(Image.open(original_first).convert('RGB')),first_ref),
                                                                        'last':endpoint(np.asarray(Image.open(original_last).convert('RGB')),last_ref)}
        row['red_centroid_is_not_object_tracking']='Simple global red-color segmentation;can include artifacts/multiple objects. Not optical flow or a verified bouncing-ball trajectory.'
        r['videos'].append(row);writer.write(r)
    r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:writer.write(r)
