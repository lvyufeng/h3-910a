#!/usr/bin/env python3
"""Decode every actual MP4 frame locally;diagnostic metrics are not a video-quality score."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
FFMPEG='/opt/homebrew/bin/ffmpeg'
FFPROBE='/opt/homebrew/bin/ffprobe'
FRAME_BYTES=1344*768*3


def endpoint(image,reference):
    error=image.astype(np.float64)-reference.astype(np.float64);mse=float(np.square(error).mean())
    return {'MAE_pixels':float(np.abs(error).mean()),'MSE_pixels':mse,
            'PSNR_db':10*math.log10(255**2/mse)if mse else None,'exact_pixels':bool(np.array_equal(image,reference))}


def read_frame(pipe):
    pieces=[];size=0
    while size<FRAME_BYTES:
        chunk=pipe.read(FRAME_BYTES-size)
        if not chunk:break
        pieces.append(chunk);size+=len(chunk)
    if not size:return None
    if size!=FRAME_BYTES:raise ValueError('Incomplete raw RGB frame in actual video decode')
    return np.frombuffer(b''.join(pieces),dtype=np.uint8).reshape((768,1344,3))


def scan(label,root=ROOT):
    path=root/(label+'.mp4')
    meta=subprocess.run([FFPROBE,'-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,timeout=30)
    if meta.returncode:raise ValueError('ffprobe failed:'+meta.stderr)
    metadata=json.loads(meta.stdout);video=[s for s in metadata['streams']if s['codec_type']=='video']
    assert len(video)==1 and (video[0]['width'],video[0]['height'])==(1344,768)and video[0]['avg_frame_rate']=='24/1'
    assert int(video[0]['nb_frames'])==124 and abs(float(video[0]['duration'])-124/24)<1e-5
    row={'label':label,'video_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ffprobe':metadata,'frames':[],
         'decoder_command':[FFMPEG,'-nostdin','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-pix_fmt','rgb24','-f','rawvideo','pipe:1']}
    previous=None;first=None;last=None;hasher=hashlib.sha256();proc=None
    log=root/(label+'_scan_ffmpeg.stderr')
    with log.open('xb')as errors:
        proc=subprocess.Popen(row['decoder_command'],stdout=subprocess.PIPE,stderr=errors)
        try:
            while (image:=read_frame(proc.stdout))is not None:
                index=len(row['frames']);assert index<124
                hasher.update(memoryview(image));rgb=image.astype(np.float32)
                red=(rgb[:,:,0]>140)&(rgb[:,:,0]>rgb[:,:,1]*1.6)&(rgb[:,:,0]>rgb[:,:,2]*1.6);ys,xs=np.nonzero(red)
                item={'index':index,'nominal_time_seconds':index/24,'RGB_sha256':hashlib.sha256(memoryview(image)).hexdigest(),
                      'mean_RGB':rgb.mean((0,1)).tolist(),'std_RGB':rgb.std((0,1)).tolist(),'red_pixel_count':int(red.sum()),
                      'red_centroid_x':float(xs.mean())if len(xs)else None,'red_centroid_y':float(ys.mean())if len(ys)else None,
                      'consecutive_frame_MAE':float(np.abs(rgb-previous).mean())if previous is not None else None,
                      'all_black':bool(image.max()==0)}
                if index in(0,12,24,36,48,62,74,86,98,110,123):
                    derived=root/(label+f'_decoded_frame{index:03d}.png')
                    assert not derived.exists();Image.fromarray(image).save(derived)
                    item['derived_full_resolution_PNG']={'filename':derived.name,'sha256':hashlib.sha256(derived.read_bytes()).hexdigest(),'scope':'Decoded from verified lossy MP4;not original returned RGB PNG'}
                if index==0:first=image.copy()
                last=image.copy();previous=rgb;row['frames'].append(item)
            rc=proc.wait(timeout=30)
            if rc:raise ValueError('Actual video RGB decode failed with exit code '+str(rc))
        finally:
            if proc.poll()is None:proc.terminate();proc.wait(timeout=10)
    assert len(row['frames'])==124
    row['all124_frames_decoded']=True;row['decoded_RGB_sha256']=hasher.hexdigest();row['black_frame_count']=sum(i['all_black']for i in row['frames'])
    refs={side:np.asarray(Image.open(root/(side+'.png')).convert('RGB'))for side in('first','last')}
    row['lossy_MP4_endpoint_metrics']={'first':endpoint(first,refs['first']),'last':endpoint(last,refs['last'])}
    row['original_exported_RGB_endpoints_before_MP4_codec']={side:endpoint(np.asarray(Image.open(root/(label+f'_frame{index:03d}.png')).convert('RGB')),refs[side])for side,index in(('first',0),('last',123))if(root/(label+f'_frame{index:03d}.png')).is_file()}
    row['both_original_RGB_endpoints_locally_available']=set(row['original_exported_RGB_endpoints_before_MP4_codec'])=={'first','last'}
    diffs=[i['consecutive_frame_MAE']for i in row['frames'][1:]]
    row['temporal_frame_MAE_diagnostics']={'mean':float(np.mean(diffs)),'max':max(diffs),'max_transition_to_frame':1+int(np.argmax(diffs))}
    row['limits']='Color-segmentation centroid can include multiple red objects/artifacts. Temporal pixel changes include intended motion. Neither is optical flow,verified tracking,OCR,CLIP or human-quality score.'
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',action='append',choices=['optimized','HF_reference_conditioning']);p.add_argument('--output',default='video_scan_local.json');args=p.parse_args()
    output=ROOT/args.output;assert output.parent==ROOT and not output.exists()
    r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'videos':[],
       'endpoint_PSNR_is_reconstruction_diagnostic_only':True,'decoded_MP4_metrics_include_lossy_codec':True,'original_PNG_endpoint_metrics_separate':True,
       'native_BF16_compute_claimed':False,'optical_flow_or_CLIP_or_OCR_score_computed':False,'official_quality_accepted':False}
    try:
        for label in(args.label or['optimized','HF_reference_conditioning']):r['videos'].append(scan(label))
        r['completed']=True
    except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        with output.open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items()if k!='videos'}));print([(i['label'],i['video_sha256'],len(i['frames']),i['original_exported_RGB_endpoints_before_MP4_codec'])for i in r['videos']])


if __name__=='__main__':main()
